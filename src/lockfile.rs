//! GPU claim management for coordination between `with-gpu` processes.
//!
//! Each GPU has a persistent file in a shared temporary directory. On Unix, an
//! advisory lock on that file is the claim; the PID stored inside is diagnostic
//! metadata only. The lock file descriptor is inherited across `exec`, so the
//! operating system releases the claim when the launched command exits.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

const LOCK_DIR_MODE: u32 = 0o1777;
const LOCK_FILE_MODE: u32 = 0o666;

#[derive(Debug, Clone)]
pub struct LockManager {
    dir: PathBuf,
}

impl LockManager {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    pub fn is_gpu_available(&self, gpu_index: usize) -> bool {
        matches!(inspect_claim(&self.dir, gpu_index), Ok(None))
    }

    pub fn claim_gpu(&self, gpu_index: usize) -> Result<GpuClaim, ClaimError> {
        claim_gpu_in(&self.dir, gpu_index)
    }

    /// Claim a set as one logical operation. If any claim fails, earlier claims
    /// are released before the error is returned.
    pub fn claim_gpus(&self, gpu_indices: &[usize]) -> Result<Vec<GpuClaim>, ClaimError> {
        let mut ordered_indices = gpu_indices.to_vec();
        ordered_indices.sort_unstable();
        ordered_indices.dedup();

        let mut claims = Vec::with_capacity(ordered_indices.len());
        for gpu_index in ordered_indices {
            claims.push(self.claim_gpu(gpu_index)?);
        }
        Ok(claims)
    }

    pub fn get_claimed_gpus(&self) -> Vec<(usize, u32)> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(_) => return Vec::new(),
        };

        let mut claimed = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(gpu_index) = path
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(|name| name.strip_prefix("gpu-"))
                .and_then(|name| name.strip_suffix(".lock"))
                .and_then(|index| index.parse::<usize>().ok())
            {
                if let Ok(Some(pid)) = inspect_claim(&self.dir, gpu_index) {
                    claimed.push((gpu_index, pid));
                }
            }
        }

        claimed.sort_by_key(|(index, _)| *index);
        claimed
    }
}

fn lock_path(dir: &Path, gpu_index: usize) -> PathBuf {
    dir.join(format!("gpu-{}.lock", gpu_index))
}

fn ensure_lock_dir(dir: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;

    let metadata = fs::symlink_metadata(dir)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(std::io::Error::other(format!(
            "{} is not a real directory",
            dir.display()
        )));
    }

    #[cfg(unix)]
    {
        let current_mode = metadata.permissions().mode() & 0o7777;
        if current_mode != LOCK_DIR_MODE {
            fs::set_permissions(dir, fs::Permissions::from_mode(LOCK_DIR_MODE)).map_err(
                |error| {
                    std::io::Error::new(
                        error.kind(),
                        format!(
                            "cannot make {} a shared lock directory (mode {:04o}): {}; fix its ownership/permissions or use --lock-dir",
                            dir.display(), current_mode, error
                        ),
                    )
                },
            )?;
        }
    }

    Ok(())
}

fn open_lock_file(dir: &Path, gpu_index: usize) -> std::io::Result<File> {
    let path = lock_path(dir, gpu_index);
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true);

    #[cfg(unix)]
    options.mode(LOCK_FILE_MODE).custom_flags(libc::O_NOFOLLOW);

    let file = options.open(&path)?;

    #[cfg(unix)]
    {
        let metadata = file.metadata()?;
        let current_mode = metadata.permissions().mode() & 0o777;
        if current_mode != LOCK_FILE_MODE {
            file.set_permissions(fs::Permissions::from_mode(LOCK_FILE_MODE))?;
        }
    }

    Ok(file)
}

fn read_pid(file: &mut File) -> Option<u32> {
    file.seek(SeekFrom::Start(0)).ok()?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).ok()?;
    contents.trim().parse().ok()
}

fn write_pid(file: &mut File) -> std::io::Result<()> {
    file.set_len(0)?;
    file.seek(SeekFrom::Start(0))?;
    write!(file, "{}", std::process::id())?;
    file.flush()
}

#[cfg(unix)]
fn try_exclusive_lock(file: &File) -> std::io::Result<bool> {
    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if result == 0 {
        return Ok(true);
    }

    let error = std::io::Error::last_os_error();
    match error.raw_os_error() {
        Some(code) if code == libc::EWOULDBLOCK || code == libc::EAGAIN => Ok(false),
        _ => Err(error),
    }
}

#[cfg(unix)]
fn unlock(file: &File) -> std::io::Result<()> {
    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(unix)]
fn inherit_across_exec(file: &File) -> std::io::Result<()> {
    let descriptor = file.as_raw_fd();
    let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFD) };
    if flags == -1 {
        return Err(std::io::Error::last_os_error());
    }

    let result = unsafe { libc::fcntl(descriptor, libc::F_SETFD, flags & !libc::FD_CLOEXEC) };
    if result == -1 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(unix)]
fn inspect_claim(dir: &Path, gpu_index: usize) -> std::io::Result<Option<u32>> {
    let path = lock_path(dir, gpu_index);
    let mut file = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };

    if try_exclusive_lock(&file)? {
        unlock(&file)?;
        Ok(None)
    } else {
        Ok(Some(read_pid(&mut file).unwrap_or(0)))
    }
}

#[cfg(not(unix))]
fn is_pid_alive(_pid: u32) -> bool {
    true
}

#[cfg(not(unix))]
fn inspect_claim(dir: &Path, gpu_index: usize) -> std::io::Result<Option<u32>> {
    let path = lock_path(dir, gpu_index);
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    Ok(read_pid(&mut file).filter(|pid| is_pid_alive(*pid)))
}

/// A claim remains active while this value, or its inherited file descriptor,
/// remains alive.
pub struct GpuClaim {
    _file: File,
    #[cfg(not(unix))]
    path: PathBuf,
}

fn claim_gpu_in(dir: &Path, gpu_index: usize) -> Result<GpuClaim, ClaimError> {
    ensure_lock_dir(dir).map_err(ClaimError::from_io)?;

    #[cfg(unix)]
    {
        let mut file = open_lock_file(dir, gpu_index).map_err(ClaimError::from_io)?;
        if !try_exclusive_lock(&file).map_err(ClaimError::from_io)? {
            return Err(ClaimError::AlreadyClaimed {
                gpu_index,
                pid: read_pid(&mut file),
            });
        }

        write_pid(&mut file).map_err(ClaimError::from_io)?;
        inherit_across_exec(&file).map_err(ClaimError::from_io)?;
        Ok(GpuClaim { _file: file })
    }

    #[cfg(not(unix))]
    {
        if let Some(pid) = inspect_claim(dir, gpu_index).map_err(ClaimError::from_io)? {
            return Err(ClaimError::AlreadyClaimed {
                gpu_index,
                pid: Some(pid),
            });
        }

        let path = lock_path(dir, gpu_index);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(ClaimError::from_io)?;
        write_pid(&mut file).map_err(ClaimError::from_io)?;
        Ok(GpuClaim { _file: file, path })
    }
}

#[cfg(not(unix))]
impl Drop for GpuClaim {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Debug)]
pub enum ClaimError {
    AlreadyClaimed { gpu_index: usize, pid: Option<u32> },
    IoError(String),
}

impl ClaimError {
    fn from_io(error: std::io::Error) -> Self {
        Self::IoError(error.to_string())
    }
}

impl std::fmt::Display for ClaimError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyClaimed {
                gpu_index,
                pid: Some(pid),
            } if *pid != 0 => write!(formatter, "GPU {} is claimed by process {}", gpu_index, pid),
            Self::AlreadyClaimed { gpu_index, .. } => {
                write!(formatter, "GPU {} is already claimed", gpu_index)
            }
            Self::IoError(message) => write!(formatter, "Lock file error: {}", message),
        }
    }
}

impl std::error::Error for ClaimError {}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("with-gpu-{name}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn claim_is_exclusive_and_released_with_guard() {
        let dir = test_dir("exclusive");
        let claim = claim_gpu_in(&dir, 0).unwrap();

        let descriptor_flags = unsafe { libc::fcntl(claim._file.as_raw_fd(), libc::F_GETFD) };
        assert_ne!(descriptor_flags, -1);
        assert_eq!(descriptor_flags & libc::FD_CLOEXEC, 0);

        assert_eq!(inspect_claim(&dir, 0).unwrap(), Some(std::process::id()));
        assert!(matches!(
            claim_gpu_in(&dir, 0),
            Err(ClaimError::AlreadyClaimed { .. })
        ));

        drop(claim);
        assert_eq!(inspect_claim(&dir, 0).unwrap(), None);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn shared_paths_have_multi_user_permissions() {
        let dir = test_dir("permissions");
        let claim = claim_gpu_in(&dir, 3).unwrap();

        let dir_mode = fs::metadata(&dir).unwrap().permissions().mode() & 0o7777;
        let file_mode = fs::metadata(lock_path(&dir, 3))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(dir_mode, LOCK_DIR_MODE);
        assert_eq!(file_mode, LOCK_FILE_MODE);

        drop(claim);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn failed_set_claim_releases_partial_claims() {
        let dir = test_dir("atomic-set");
        let manager = LockManager::new(&dir);
        let blocking_claim = manager.claim_gpu(2).unwrap();

        assert!(matches!(
            manager.claim_gpus(&[1, 2]),
            Err(ClaimError::AlreadyClaimed { gpu_index: 2, .. })
        ));
        assert!(manager.is_gpu_available(1));

        drop(blocking_claim);
        fs::remove_dir_all(dir).unwrap();
    }
}

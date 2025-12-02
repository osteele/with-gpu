//! Lock file management for GPU coordination.
//!
//! Prevents race conditions when multiple `with-gpu` processes start simultaneously
//! by creating per-GPU lock files that track which process has claimed each GPU.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;

/// Directory for lock files
fn lock_dir() -> PathBuf {
    PathBuf::from("/tmp/with-gpu")
}

/// Path to lock file for a specific GPU
fn lock_path(gpu_index: usize) -> PathBuf {
    lock_dir().join(format!("gpu-{}.lock", gpu_index))
}

/// Ensure the lock directory exists
fn ensure_lock_dir() -> std::io::Result<()> {
    fs::create_dir_all(lock_dir())
}

/// Check if a process with the given PID is still alive
fn is_pid_alive(pid: u32) -> bool {
    // On Unix, sending signal 0 checks if process exists without actually signaling
    #[cfg(unix)]
    {
        let ret = unsafe { libc::kill(pid as libc::pid_t, 0) };
        if ret == 0 {
            return true; // Process exists and we can signal it
        }
        // Check errno to distinguish "no such process" from "permission denied"
        // EPERM = process exists but belongs to another user (treat as alive)
        // ESRCH = no such process (treat as dead)
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        errno == libc::EPERM
    }
    #[cfg(not(unix))]
    {
        // On non-Unix, assume process is alive (conservative)
        let _ = pid;
        true
    }
}

/// Check if a GPU is currently claimed by another process.
/// Returns Some(pid) if claimed, None if available.
pub fn get_gpu_claim(gpu_index: usize) -> Option<u32> {
    let path = lock_path(gpu_index);

    let mut file = match File::open(&path) {
        Ok(f) => f,
        Err(_) => return None, // No lock file = not claimed
    };

    let mut contents = String::new();
    if file.read_to_string(&mut contents).is_err() {
        return None;
    }

    let pid: u32 = match contents.trim().parse() {
        Ok(p) => p,
        Err(_) => {
            // Invalid lock file, remove it
            let _ = fs::remove_file(&path);
            return None;
        }
    };

    if is_pid_alive(pid) {
        Some(pid)
    } else {
        // Stale lock file (process died), clean it up
        let _ = fs::remove_file(&path);
        None
    }
}

/// Check if a GPU is available (not claimed by another process)
pub fn is_gpu_available(gpu_index: usize) -> bool {
    get_gpu_claim(gpu_index).is_none()
}

/// Attempt to claim a GPU. Returns Ok(()) if successful, Err if already claimed.
pub fn claim_gpu(gpu_index: usize) -> Result<(), ClaimError> {
    ensure_lock_dir().map_err(|e| ClaimError::IoError(e.to_string()))?;

    let path = lock_path(gpu_index);

    // First check if there's an existing valid claim
    if let Some(pid) = get_gpu_claim(gpu_index) {
        return Err(ClaimError::AlreadyClaimed { gpu_index, pid });
    }

    // Try to create lock file atomically
    let mut file = match OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            // Race condition: another process claimed it between our check and create
            // Re-check if it's a valid claim
            if let Some(pid) = get_gpu_claim(gpu_index) {
                return Err(ClaimError::AlreadyClaimed { gpu_index, pid });
            }
            // Stale file was cleaned up by get_gpu_claim, try again
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|e| ClaimError::IoError(e.to_string()))?
        }
        Err(e) => return Err(ClaimError::IoError(e.to_string())),
    };

    // Write our PID to the lock file
    let pid = std::process::id();
    write!(file, "{}", pid).map_err(|e| ClaimError::IoError(e.to_string()))?;

    Ok(())
}

/// Get list of GPUs that are currently claimed (for status display)
pub fn get_claimed_gpus() -> Vec<(usize, u32)> {
    let mut claimed = Vec::new();

    // Dynamically enumerate lock files to support any number of GPUs
    let lock_dir = lock_dir();
    let entries = match fs::read_dir(&lock_dir) {
        Ok(e) => e,
        Err(_) => return claimed, // No lock directory = no claims
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
            // Parse "gpu-N.lock" pattern
            if let Some(idx_str) = filename
                .strip_prefix("gpu-")
                .and_then(|s| s.strip_suffix(".lock"))
            {
                if let Ok(gpu_index) = idx_str.parse::<usize>() {
                    if let Some(pid) = get_gpu_claim(gpu_index) {
                        claimed.push((gpu_index, pid));
                    }
                }
            }
        }
    }

    claimed.sort_by_key(|(idx, _)| *idx);
    claimed
}

#[derive(Debug)]
pub enum ClaimError {
    AlreadyClaimed { gpu_index: usize, pid: u32 },
    IoError(String),
}

impl std::fmt::Display for ClaimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClaimError::AlreadyClaimed { gpu_index, pid } => {
                write!(f, "GPU {} is claimed by process {}", gpu_index, pid)
            }
            ClaimError::IoError(msg) => write!(f, "Lock file error: {}", msg),
        }
    }
}

impl std::error::Error for ClaimError {}

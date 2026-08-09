//! CUDA memory queries using the CUDA Driver API.
//!
//! This module provides accurate GPU memory information by querying CUDA directly,
//! bypassing NVML which can return stale data in some scenarios.

use anyhow::{anyhow, Result};
use std::ffi::OsString;
use std::ptr;

use cudarc::driver::{result, sys};

struct RemovedEnvironmentVariable {
    name: &'static str,
    value: Option<OsString>,
}

impl RemovedEnvironmentVariable {
    fn new(name: &'static str) -> Self {
        let value = std::env::var_os(name);
        std::env::remove_var(name);
        Self { name, value }
    }
}

impl Drop for RemovedEnvironmentVariable {
    fn drop(&mut self) {
        if let Some(value) = &self.value {
            std::env::set_var(self.name, value);
        } else {
            std::env::remove_var(self.name);
        }
    }
}

/// Memory information for a single GPU.
#[derive(Debug, Clone)]
pub struct CudaMemoryInfo {
    pub device_uuid: String,
    pub free_bytes: u64,
    pub total_bytes: u64,
}

struct PrimaryContextGuard {
    device: sys::CUdevice,
    previous: Option<sys::CUcontext>,
}

impl PrimaryContextGuard {
    fn retain_and_activate(device: sys::CUdevice, device_index: usize) -> Result<Self> {
        let previous = result::ctx::get_current()
            .map_err(|error| anyhow!("Failed to get current CUDA context: {error:?}"))?;
        // SAFETY: device came from CUDA's device::get.
        let context = unsafe { result::primary_ctx::retain(device) }.map_err(|error| {
            anyhow!("Failed to retain CUDA context for device {device_index}: {error:?}")
        })?;
        let guard = Self { device, previous };
        // SAFETY: context was returned by primary_ctx::retain and remains retained
        // for the lifetime of guard.
        unsafe { result::ctx::set_current(context) }
            .map_err(|error| anyhow!("Failed to set CUDA context as current: {error:?}"))?;
        Ok(guard)
    }
}

impl Drop for PrimaryContextGuard {
    fn drop(&mut self) {
        let previous = self.previous.unwrap_or(ptr::null_mut());
        // SAFETY: the previous context, when present, was returned by CUDA. CUDA
        // accepts a null context to clear the current context.
        let _ = unsafe { result::ctx::set_current(previous) };
        // SAFETY: this balances the successful primary_ctx::retain in the
        // constructor.
        let _ = unsafe { result::primary_ctx::release(self.device) };
    }
}

fn format_uuid(uuid: sys::CUuuid) -> String {
    uuid.bytes
        .iter()
        .map(|byte| format!("{:02x}", *byte as u8))
        .collect()
}

impl CudaMemoryInfo {
    pub fn used_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.free_bytes)
    }

    pub fn used_mb(&self) -> u64 {
        self.used_bytes() / (1024 * 1024)
    }

    pub fn total_mb(&self) -> u64 {
        self.total_bytes / (1024 * 1024)
    }
}

/// Query memory info for a specific GPU using CUDA Driver API.
///
/// This creates a CUDA context on the device, queries memory, then releases the context.
/// More accurate than NVML's memory_info() which can return stale data.
pub fn query_device_memory(device_index: usize) -> Result<CudaMemoryInfo> {
    // Initialize CUDA driver API (safe to call multiple times)
    result::init().map_err(|e| anyhow!("Failed to initialize CUDA driver: {:?}", e))?;

    // Get device handle
    let device = result::device::get(device_index as i32)
        .map_err(|e| anyhow!("Failed to get CUDA device {}: {:?}", device_index, e))?;
    let device_uuid = result::device::get_uuid(device)
        .map(format_uuid)
        .map_err(|error| anyhow!("Failed to get UUID for CUDA device {device_index}: {error:?}"))?;

    let _context = PrimaryContextGuard::retain_and_activate(device, device_index)?;

    // Query memory info using the result module's wrapper
    let (free, total) = result::mem_get_info().map_err(|e| {
        anyhow!(
            "Failed to get memory info for device {}: {:?}",
            device_index,
            e
        )
    })?;

    Ok(CudaMemoryInfo {
        device_uuid,
        free_bytes: free as u64,
        total_bytes: total as u64,
    })
}

/// Query memory info for all GPUs.
pub fn query_all_device_memory() -> Result<Vec<CudaMemoryInfo>> {
    // CUDA ordinals are filtered and reordered by CUDA_VISIBLE_DEVICES, while
    // NVML indices are physical. Query the unfiltered device list so the two
    // APIs describe the same ordinal space. The launched command receives its
    // selected CUDA_VISIBLE_DEVICES value later, after exec.
    let _visible_devices = RemovedEnvironmentVariable::new("CUDA_VISIBLE_DEVICES");

    // Initialize CUDA driver API
    result::init().map_err(|e| anyhow!("Failed to initialize CUDA driver: {:?}", e))?;

    let device_count = result::device::get_count()
        .map_err(|e| anyhow!("Failed to get CUDA device count: {:?}", e))?;

    let mut results = Vec::with_capacity(device_count as usize);
    for i in 0..device_count {
        match query_device_memory(i as usize) {
            Ok(info) => results.push(info),
            Err(e) => {
                // Log warning but continue with other devices
                eprintln!(
                    "Warning: Failed to query CUDA memory for device {}: {}",
                    i, e
                );
            }
        }
    }

    Ok(results)
}

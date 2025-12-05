//! CUDA memory queries using the CUDA Driver API.
//!
//! This module provides accurate GPU memory information by querying CUDA directly,
//! bypassing NVML which can return stale data in some scenarios.

use anyhow::{anyhow, Result};

/// Memory information for a single GPU.
#[derive(Debug, Clone)]
pub struct CudaMemoryInfo {
    pub device_index: usize,
    pub free_bytes: u64,
    pub total_bytes: u64,
}

impl CudaMemoryInfo {
    pub fn used_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.free_bytes)
    }

    pub fn used_mb(&self) -> u64 {
        self.used_bytes() / (1024 * 1024)
    }

    pub fn free_mb(&self) -> u64 {
        self.free_bytes / (1024 * 1024)
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
    use cudarc::driver::result;

    // Initialize CUDA driver API (safe to call multiple times)
    result::init().map_err(|e| anyhow!("Failed to initialize CUDA driver: {:?}", e))?;

    // Get device handle
    let device = result::device::get(device_index as i32)
        .map_err(|e| anyhow!("Failed to get CUDA device {}: {:?}", device_index, e))?;

    // Create/retain a primary context for this device
    // SAFETY: device is a valid device handle obtained from device::get
    let ctx = unsafe {
        result::primary_ctx::retain(device)
            .map_err(|e| anyhow!("Failed to create CUDA context for device {}: {:?}", device_index, e))?
    };

    // Push context to make it current
    // SAFETY: ctx is a valid context obtained from primary_ctx::retain
    unsafe {
        result::ctx::set_current(ctx)
            .map_err(|e| anyhow!("Failed to set CUDA context as current: {:?}", e))?;
    }

    // Query memory info using the result module's wrapper
    let (free, total) = result::mem_get_info()
        .map_err(|e| anyhow!("Failed to get memory info for device {}: {:?}", device_index, e))?;

    // Release the primary context (decrements refcount, doesn't destroy)
    // SAFETY: device is a valid device handle
    unsafe {
        result::primary_ctx::release(device)
            .map_err(|e| anyhow!("Failed to release CUDA context: {:?}", e))?;
    }

    Ok(CudaMemoryInfo {
        device_index,
        free_bytes: free as u64,
        total_bytes: total as u64,
    })
}

/// Query memory info for all GPUs.
pub fn query_all_device_memory() -> Result<Vec<CudaMemoryInfo>> {
    use cudarc::driver::result;

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
                eprintln!("Warning: Failed to query CUDA memory for device {}: {}", i, e);
            }
        }
    }

    Ok(results)
}

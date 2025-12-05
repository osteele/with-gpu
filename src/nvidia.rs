use anyhow::Result;

use crate::GpuInfo;

#[cfg(not(target_os = "macos"))]
use anyhow::Context;

#[cfg(not(target_os = "macos"))]
use nvml_wrapper::Nvml;

#[cfg(not(target_os = "macos"))]
use crate::cuda;

pub fn query_gpus() -> Result<Vec<GpuInfo>> {
    #[cfg(target_os = "macos")]
    {
        // On macOS, there are no NVIDIA GPUs - return empty list
        // This allows the tool to work as a no-op (just execute the command)
        Ok(vec![])
    }

    #[cfg(not(target_os = "macos"))]
    {
        let nvml =
            Nvml::init().context("Failed to initialize NVML (is the NVIDIA driver installed?)")?;

        let device_count = nvml.device_count().context("Failed to get GPU count")?;

        // Query CUDA memory for all devices upfront
        // This gives us accurate memory usage that NVML may miss
        let cuda_memory = cuda::query_all_device_memory().unwrap_or_default();

        let mut gpus = Vec::new();
        for i in 0..device_count {
            let device = nvml
                .device_by_index(i)
                .context(format!("Failed to get GPU {}", i))?;

            // Get NVML memory info as fallback
            let nvml_memory_info = device
                .memory_info()
                .context(format!("Failed to get memory info for GPU {}", i))?;

            // Prefer CUDA memory info if available (more accurate)
            let (memory_used_mb, memory_total_mb) =
                if let Some(cuda_info) = cuda_memory.iter().find(|m| m.device_index == i as usize) {
                    (cuda_info.used_mb(), cuda_info.total_mb())
                } else {
                    // Fallback to NVML if CUDA query failed for this device
                    (
                        nvml_memory_info.used / (1024 * 1024),
                        nvml_memory_info.total / (1024 * 1024),
                    )
                };

            let utilization = device
                .utilization_rates()
                .context(format!("Failed to get utilization for GPU {}", i))?;

            let process_infos = device
                .running_compute_processes()
                .context(format!("Failed to get process info for GPU {}", i))?;

            let index = i as usize;
            let utilization_percent = utilization.gpu as u8;
            let process_count = process_infos.len();

            // Sum memory attributed to visible processes (from NVML)
            let attributed_memory_mb: u64 = process_infos
                .iter()
                .filter_map(|p| match p.used_gpu_memory {
                    nvml_wrapper::enums::device::UsedGpuMemory::Used(bytes) => {
                        Some(bytes / (1024 * 1024))
                    }
                    nvml_wrapper::enums::device::UsedGpuMemory::Unavailable => None,
                })
                .sum();

            // Hidden usage is total used minus attributed (clamp negative/rounding noise to zero)
            // Now uses CUDA memory which is more accurate than NVML
            let hidden_usage_mb = memory_used_mb.saturating_sub(attributed_memory_mb);

            gpus.push(GpuInfo {
                index,
                memory_used_mb,
                memory_total_mb,
                utilization_percent,
                process_count,
                hidden_usage_mb,
            });
        }

        Ok(gpus)
    }
}

use anyhow::Result;

use crate::GpuInfo;

#[cfg(not(target_os = "macos"))]
use anyhow::Context;

#[cfg(not(target_os = "macos"))]
use nvml_wrapper::Nvml;

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

        let mut gpus = Vec::new();
        for i in 0..device_count {
            let device = nvml
                .device_by_index(i)
                .context(format!("Failed to get GPU {}", i))?;

            let memory_info = device
                .memory_info()
                .context(format!("Failed to get memory info for GPU {}", i))?;

            let utilization = device
                .utilization_rates()
                .context(format!("Failed to get utilization for GPU {}", i))?;

            let process_infos = device
                .running_compute_processes()
                .context(format!("Failed to get process info for GPU {}", i))?;

            let index = i as usize;
            let memory_used_mb = memory_info.used / (1024 * 1024);
            let memory_total_mb = memory_info.total / (1024 * 1024);
            let utilization_percent = utilization.gpu as u8;
            let process_count = process_infos.len();

            // Sum memory attributed to visible processes
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

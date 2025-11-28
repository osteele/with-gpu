use std::fmt;

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub index: usize,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub utilization_percent: u8,
    pub process_count: usize,
}

impl GpuInfo {
    pub fn is_idle(&self) -> bool {
        // A GPU is idle if it has no processes AND minimal memory usage
        // We check memory usage because NVML process detection can miss processes
        // in some cases (e.g., persistence mode, MPS, certain driver states)
        const IDLE_MEMORY_THRESHOLD_MB: u64 = 500;
        self.process_count == 0 && self.memory_used_mb < IDLE_MEMORY_THRESHOLD_MB
    }

    pub fn memory_free_mb(&self) -> u64 {
        self.memory_total_mb.saturating_sub(self.memory_used_mb)
    }

    pub fn memory_usage_percent(&self) -> f64 {
        if self.memory_total_mb == 0 {
            0.0
        } else {
            (self.memory_used_mb as f64 / self.memory_total_mb as f64) * 100.0
        }
    }
}

impl fmt::Display for GpuInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.is_idle() { "IDLE" } else { "USED" };
        write!(
            f,
            "GPU {}: {} - {}/{} MB ({:.1}%), {} util, {} processes",
            self.index,
            status,
            self.memory_used_mb,
            self.memory_total_mb,
            self.memory_usage_percent(),
            self.utilization_percent,
            self.process_count
        )
    }
}

#[derive(Debug, Clone)]
pub struct GpuSelection {
    pub gpu_indices: Vec<usize>,
    pub all_idle: bool,
    pub warning: Option<String>,
}

impl GpuSelection {
    pub fn to_cuda_visible_devices(&self) -> String {
        self.gpu_indices
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }
}

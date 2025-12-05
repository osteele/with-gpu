use std::fmt;

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub index: usize,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub utilization_percent: u8,
    pub process_count: usize,
    /// Memory used but not attributed to visible processes (indicates hidden/stale process data)
    pub hidden_usage_mb: u64,
}

/// Threshold for detecting hidden memory usage (driver jitter tolerance)
pub const HIDDEN_USAGE_THRESHOLD_MB: u64 = 512;

impl GpuInfo {
    /// Returns true if unattributed memory usage exceeds the given threshold.
    /// This indicates processes using GPU memory that aren't visible to NVML.
    pub fn has_hidden_usage(&self, threshold_mb: u64) -> bool {
        self.hidden_usage_mb > threshold_mb
    }

    pub fn is_idle(&self) -> bool {
        // A GPU is idle if it has no processes AND minimal memory usage
        // We check memory usage because NVML process detection can miss processes
        // in some cases (e.g., persistence mode, MPS, certain driver states)
        const IDLE_MEMORY_THRESHOLD_MB: u64 = 500;
        self.process_count == 0
            && self.memory_used_mb < IDLE_MEMORY_THRESHOLD_MB
            && !self.has_hidden_usage(HIDDEN_USAGE_THRESHOLD_MB)
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
        )?;
        if self.has_hidden_usage(HIDDEN_USAGE_THRESHOLD_MB) {
            write!(f, " (suspected hidden usage: {} MB)", self.hidden_usage_mb)?;
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gpu(
        index: usize,
        memory_used_mb: u64,
        process_count: usize,
        hidden_usage_mb: u64,
    ) -> GpuInfo {
        GpuInfo {
            index,
            memory_used_mb,
            memory_total_mb: 24000,
            utilization_percent: 0,
            process_count,
            hidden_usage_mb,
        }
    }

    #[test]
    fn test_has_hidden_usage_ignores_small_noise() {
        let gpu = make_gpu(0, 600, 1, 100);
        assert!(!gpu.has_hidden_usage(512));
    }

    #[test]
    fn test_has_hidden_usage_detects_large_discrepancy() {
        let gpu = make_gpu(0, 12000, 0, 11500);
        assert!(gpu.has_hidden_usage(512));
        assert!(!gpu.is_idle());
    }

    #[test]
    fn test_is_idle_with_no_hidden_usage() {
        let gpu = make_gpu(0, 300, 0, 0);
        assert!(gpu.is_idle());
    }

    #[test]
    fn test_is_idle_false_when_has_processes() {
        let gpu = make_gpu(0, 300, 1, 0);
        assert!(!gpu.is_idle());
    }

    #[test]
    fn test_is_idle_false_when_memory_above_threshold() {
        let gpu = make_gpu(0, 600, 0, 0);
        assert!(!gpu.is_idle());
    }

    #[test]
    fn test_display_shows_hidden_usage() {
        let gpu = make_gpu(0, 12000, 0, 11500);
        let display = format!("{}", gpu);
        assert!(display.contains("suspected hidden usage: 11500 MB"));
    }

    #[test]
    fn test_display_hides_small_hidden_usage() {
        let gpu = make_gpu(0, 600, 1, 100);
        let display = format!("{}", gpu);
        assert!(!display.contains("hidden usage"));
    }
}

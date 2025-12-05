use anyhow::{Context, Result};

use crate::lockfile;
use with_gpu::{GpuInfo, GpuSelection, HIDDEN_USAGE_THRESHOLD_MB};

pub struct SelectionCriteria {
    pub min_gpus: usize,
    pub max_gpus: usize,
    pub require_idle: bool,
    pub min_memory_mb: Option<u64>,
    pub max_utilization: Option<u8>,
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            min_gpus: 1,
            max_gpus: 1,
            require_idle: false,
            min_memory_mb: Some(2048),
            max_utilization: None,
        }
    }
}

pub fn select_gpus(gpus: &[GpuInfo], criteria: &SelectionCriteria) -> Result<GpuSelection> {
    if gpus.is_empty() {
        anyhow::bail!("No GPUs detected");
    }

    // Apply threshold filters and exclude claimed GPUs
    let filtered_gpus: Vec<&GpuInfo> = gpus
        .iter()
        .filter(|gpu| {
            // Filter out GPUs claimed by other processes
            if !lockfile::is_gpu_available(gpu.index) {
                return false;
            }
            // Filter out GPUs with hidden memory usage (stale NVML data)
            if gpu.has_hidden_usage(HIDDEN_USAGE_THRESHOLD_MB) {
                return false;
            }
            // Filter by minimum free memory
            if let Some(min_mem) = criteria.min_memory_mb {
                if gpu.memory_free_mb() < min_mem {
                    return false;
                }
            }
            // Filter by maximum utilization
            if let Some(max_util) = criteria.max_utilization {
                if gpu.utilization_percent > max_util {
                    return false;
                }
            }
            true
        })
        .collect();

    // Check if filtering left us with no GPUs
    if filtered_gpus.is_empty() {
        let mut reasons = Vec::new();
        let claimed = lockfile::get_claimed_gpus();
        if !claimed.is_empty() {
            reasons.push(format!(
                "{} GPU(s) claimed by other processes",
                claimed.len()
            ));
        }
        let hidden_count = gpus
            .iter()
            .filter(|g| g.has_hidden_usage(HIDDEN_USAGE_THRESHOLD_MB))
            .count();
        if hidden_count > 0 {
            reasons.push(format!(
                "{} GPU(s) have suspected hidden memory usage",
                hidden_count
            ));
        }
        if let Some(min_mem) = criteria.min_memory_mb {
            reasons.push(format!("{}+ MB free memory required", min_mem));
        }
        if let Some(max_util) = criteria.max_utilization {
            reasons.push(format!("≤{}% utilization required", max_util));
        }
        anyhow::bail!(
            "No GPUs found matching criteria: {} (use --status to see GPU state)",
            reasons.join(", ")
        );
    }

    let (idle_gpus, _used_gpus) = partition_gpus_refs(&filtered_gpus);

    // If --require-idle is set, only consider idle GPUs
    if criteria.require_idle {
        if idle_gpus.len() < criteria.min_gpus {
            anyhow::bail!(
                "Require {} idle GPUs but only {} available (use --status to see GPU state)",
                criteria.min_gpus,
                idle_gpus.len()
            );
        }
        // Sort idle GPUs by available memory (most free first)
        let sorted_idle = sort_by_most_free_refs(&idle_gpus);
        let count = criteria.max_gpus.min(sorted_idle.len());
        let selected: Vec<usize> = sorted_idle.iter().take(count).map(|g| g.index).collect();

        return Ok(GpuSelection {
            gpu_indices: selected,
            all_idle: true,
            warning: None,
        });
    }

    // Sort filtered GPUs by available memory (most free first)
    // This prioritizes available memory over idle status
    let all_gpus_sorted = sort_by_most_free_refs(&filtered_gpus);

    // Select the requested number of GPUs
    let count = criteria.max_gpus.min(all_gpus_sorted.len());
    let selected_gpus: Vec<&GpuInfo> = all_gpus_sorted.iter().take(count).copied().collect();

    // Check if we have enough GPUs
    if selected_gpus.len() < criteria.min_gpus {
        anyhow::bail!(
            "Need {} GPUs but only {} available (use --status to see GPU state)",
            criteria.min_gpus,
            selected_gpus.len()
        );
    }

    // Check if all selected GPUs are idle
    let all_idle = selected_gpus.iter().all(|g| g.is_idle());

    // Generate warning if we're using non-idle GPUs
    let warning = if !all_idle {
        let non_idle_count = selected_gpus.iter().filter(|g| !g.is_idle()).count();
        let idle_count = idle_gpus.len();
        Some(format!(
            "Warning: Using {} non-idle GPU(s) with most available memory (only {} idle GPU(s) available)",
            non_idle_count,
            idle_count
        ))
    } else {
        None
    };

    let gpu_indices: Vec<usize> = selected_gpus.iter().map(|g| g.index).collect();

    Ok(GpuSelection {
        gpu_indices,
        all_idle,
        warning,
    })
}

fn partition_gpus_refs<'a>(gpus: &[&'a GpuInfo]) -> (Vec<&'a GpuInfo>, Vec<&'a GpuInfo>) {
    let mut idle = Vec::new();
    let mut used = Vec::new();

    for &gpu in gpus {
        if gpu.is_idle() {
            idle.push(gpu);
        } else {
            used.push(gpu);
        }
    }

    (idle, used)
}

fn sort_by_most_free_refs<'a>(gpus: &[&'a GpuInfo]) -> Vec<&'a GpuInfo> {
    let mut sorted = gpus.to_vec();
    sorted.sort_by(|a, b| {
        // Primary: Most free memory (descending)
        b.memory_free_mb()
            .cmp(&a.memory_free_mb())
            // Secondary: Fewest processes (ascending)
            .then_with(|| a.process_count.cmp(&b.process_count))
            // Tertiary: Lowest index (ascending)
            .then_with(|| a.index.cmp(&b.index))
    });
    sorted
}

pub fn parse_manual_gpu_selection(input: &str) -> Result<Vec<usize>> {
    input
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .context(format!("Invalid GPU ID: '{}'", s))
        })
        .collect()
}

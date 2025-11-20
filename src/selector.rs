use anyhow::{Context, Result};

use crate::{GpuInfo, GpuSelection};

pub struct SelectionCriteria {
    pub min_gpus: usize,
    pub max_gpus: usize,
    pub require_idle: bool,
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            min_gpus: 1,
            max_gpus: 1,
            require_idle: false,
        }
    }
}

pub fn select_gpus(gpus: &[GpuInfo], criteria: &SelectionCriteria) -> Result<GpuSelection> {
    if gpus.is_empty() {
        anyhow::bail!("No GPUs detected");
    }

    let (idle_gpus, _used_gpus) = partition_gpus(gpus);

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

    // Sort ALL GPUs by available memory (most free first)
    // This prioritizes available memory over idle status
    let all_gpus_sorted = sort_by_most_free(gpus);

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

    let mut gpu_indices: Vec<usize> = selected_gpus.iter().map(|g| g.index).collect();
    gpu_indices.sort_unstable();

    Ok(GpuSelection {
        gpu_indices,
        all_idle,
        warning,
    })
}

fn partition_gpus(gpus: &[GpuInfo]) -> (Vec<&GpuInfo>, Vec<&GpuInfo>) {
    let mut idle = Vec::new();
    let mut used = Vec::new();

    for gpu in gpus {
        if gpu.is_idle() {
            idle.push(gpu);
        } else {
            used.push(gpu);
        }
    }

    (idle, used)
}

fn sort_by_most_free(gpus: &[GpuInfo]) -> Vec<&GpuInfo> {
    let mut sorted: Vec<&GpuInfo> = gpus.iter().collect();
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

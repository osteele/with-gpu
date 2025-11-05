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

    let (idle_gpus, used_gpus) = partition_gpus(gpus);

    let mut selected = Vec::new();
    let mut all_idle = true;
    let mut warning = None;

    if idle_gpus.len() >= criteria.min_gpus {
        let count = criteria.max_gpus.min(idle_gpus.len());
        selected.extend(idle_gpus.iter().take(count).map(|g| g.index));
    } else if criteria.require_idle {
        anyhow::bail!(
            "Require {} idle GPUs but only {} available (use --status to see GPU state)",
            criteria.min_gpus,
            idle_gpus.len()
        );
    } else {
        selected.extend(idle_gpus.iter().map(|g| g.index));

        let needed = criteria.min_gpus.saturating_sub(idle_gpus.len());
        if needed > 0 {
            let sorted_used = sort_by_least_used(&used_gpus);
            let count = needed.min(sorted_used.len());
            selected.extend(sorted_used.iter().take(count).map(|g| g.index));
            all_idle = false;

            if selected.len() < criteria.min_gpus {
                anyhow::bail!(
                    "Need {} GPUs but only {} available (use --status to see GPU state)",
                    criteria.min_gpus,
                    selected.len()
                );
            }

            let used_count = selected.len() - idle_gpus.len();
            warning = Some(format!(
                "Warning: Using {} non-idle GPU(s) because only {} idle GPU(s) available",
                used_count,
                idle_gpus.len()
            ));
        }

        if selected.len() < criteria.max_gpus && !used_gpus.is_empty() {
            let sorted_used = sort_by_least_used(&used_gpus);
            let remaining = criteria.max_gpus - selected.len();
            for gpu in sorted_used.iter().take(remaining) {
                if !selected.contains(&gpu.index) {
                    selected.push(gpu.index);
                }
            }
        }
    }

    selected.sort_unstable();

    Ok(GpuSelection {
        gpu_indices: selected,
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

fn sort_by_least_used<'a>(gpus: &[&'a GpuInfo]) -> Vec<&'a GpuInfo> {
    let mut sorted = gpus.to_vec();
    sorted.sort_by(|a, b| {
        a.memory_used_mb
            .cmp(&b.memory_used_mb)
            .then_with(|| a.process_count.cmp(&b.process_count))
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

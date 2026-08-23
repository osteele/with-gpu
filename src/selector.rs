use anyhow::{Context, Result};
use std::collections::HashSet;

use crate::lockfile::LockManager;
use with_gpu::{GpuInfo, GpuSelection, HIDDEN_USAGE_THRESHOLD_MB};

pub struct SelectionCriteria {
    pub min_gpus: usize,
    pub max_gpus: usize,
    pub require_idle: bool,
    pub min_memory_mb: Option<u64>,
    pub max_utilization: Option<u8>,
    pub gpu_type_pattern: Option<String>,
    pub require_type_match: bool,
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            min_gpus: 1,
            max_gpus: 1,
            require_idle: false,
            min_memory_mb: Some(2048),
            max_utilization: None,
            gpu_type_pattern: None,
            require_type_match: false,
        }
    }
}

pub fn select_gpus(
    gpus: &[GpuInfo],
    criteria: &SelectionCriteria,
    locks: &LockManager,
) -> Result<GpuSelection> {
    if gpus.is_empty() {
        anyhow::bail!("No GPUs detected");
    }

    if let Some(pattern) = &criteria.gpu_type_pattern {
        let matching = gpus
            .iter()
            .filter(|gpu| gpu.matches_type(pattern))
            .collect::<Vec<_>>();
        if criteria.require_type_match {
            if matching.is_empty() {
                anyhow::bail!(
                    "No GPUs found matching type '{}' (use --status to see available GPUs)",
                    pattern
                );
            }
            return select_from_candidates(&matching, criteria, locks, None);
        }
        return select_from_candidates(
            &gpus.iter().collect::<Vec<_>>(),
            criteria,
            locks,
            Some(pattern),
        );
    }

    select_from_candidates(&gpus.iter().collect::<Vec<_>>(), criteria, locks, None)
}

fn select_from_candidates(
    candidate_gpus: &[&GpuInfo],
    criteria: &SelectionCriteria,
    locks: &LockManager,
    preferred_type: Option<&str>,
) -> Result<GpuSelection> {
    // Apply threshold filters and exclude claimed GPUs
    let filtered_gpus: Vec<&GpuInfo> = candidate_gpus
        .iter()
        .copied()
        .filter(|gpu| {
            // Filter out GPUs claimed by other processes
            if !locks.is_gpu_available(gpu.index) {
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
        let claimed = locks.get_claimed_gpus();
        if !claimed.is_empty() {
            reasons.push(format!(
                "{} GPU(s) claimed by other processes",
                claimed.len()
            ));
        }
        let hidden_count = candidate_gpus
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
        let sorted_idle = sort_by_most_free_refs(&idle_gpus, preferred_type);
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
    let all_gpus_sorted = sort_by_most_free_refs(&filtered_gpus, preferred_type);

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

/// Select exactly the requested GPU indices, preserving their order.
///
/// Manual selection bypasses the automatic min/max count and ranking rules, but
/// every requested GPU must still satisfy the availability and threshold filters.
pub fn select_manual_gpus(
    gpus: &[GpuInfo],
    indices: &[usize],
    criteria: &SelectionCriteria,
    locks: &LockManager,
) -> Result<GpuSelection> {
    validate_manual_gpu_selection(gpus, indices)?;

    let selected_gpus: Vec<&GpuInfo> = indices
        .iter()
        .map(|index| {
            gpus.iter()
                .find(|gpu| gpu.index == *index)
                .expect("manual GPU indices were validated")
        })
        .collect();

    for gpu in &selected_gpus {
        if let Some(reason) = gpu_rejection_reason(gpu, criteria, locks) {
            anyhow::bail!(
                "Manually selected GPU {} is unavailable: {} (use --status to see GPU state)",
                gpu.index,
                reason
            );
        }
    }

    let all_idle = selected_gpus.iter().all(|gpu| gpu.is_idle());
    let warning = if all_idle {
        None
    } else {
        let non_idle_count = selected_gpus.iter().filter(|gpu| !gpu.is_idle()).count();
        Some(format!(
            "Warning: Manually selected {} non-idle GPU(s)",
            non_idle_count
        ))
    };

    Ok(GpuSelection {
        gpu_indices: indices.to_vec(),
        all_idle,
        warning,
    })
}

pub fn validate_manual_gpu_selection(gpus: &[GpuInfo], indices: &[usize]) -> Result<()> {
    if indices.is_empty() {
        anyhow::bail!("Manual GPU selection cannot be empty");
    }

    let mut unique_indices = HashSet::with_capacity(indices.len());
    for &index in indices {
        if !unique_indices.insert(index) {
            anyhow::bail!("GPU {} is listed more than once", index);
        }
        if !gpus.iter().any(|gpu| gpu.index == index) {
            let available = gpus
                .iter()
                .map(|gpu| gpu.index.to_string())
                .collect::<Vec<_>>()
                .join(",");
            anyhow::bail!("GPU {} not found (available: {})", index, available);
        }
    }

    Ok(())
}

fn gpu_rejection_reason(
    gpu: &GpuInfo,
    criteria: &SelectionCriteria,
    locks: &LockManager,
) -> Option<String> {
    if !locks.is_gpu_available(gpu.index) {
        return Some("claimed by another with-gpu process".to_string());
    }
    if gpu.has_hidden_usage(HIDDEN_USAGE_THRESHOLD_MB) {
        return Some(format!(
            "{} MB of suspected hidden memory usage",
            gpu.hidden_usage_mb
        ));
    }
    if let Some(min_mem) = criteria.min_memory_mb {
        if gpu.memory_free_mb() < min_mem {
            return Some(format!(
                "{} MB free memory is below the required {} MB",
                gpu.memory_free_mb(),
                min_mem
            ));
        }
    }
    if let Some(max_util) = criteria.max_utilization {
        if gpu.utilization_percent > max_util {
            return Some(format!(
                "{}% utilization exceeds the allowed {}%",
                gpu.utilization_percent, max_util
            ));
        }
    }
    if criteria.require_idle && !gpu.is_idle() {
        return Some("GPU is not idle".to_string());
    }

    None
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

fn sort_by_most_free_refs<'a>(
    gpus: &[&'a GpuInfo],
    preferred_type: Option<&str>,
) -> Vec<&'a GpuInfo> {
    let mut sorted = gpus.to_vec();
    sorted.sort_by(|a, b| {
        let a_is_preferred = preferred_type.is_some_and(|pattern| a.matches_type(pattern));
        let b_is_preferred = preferred_type.is_some_and(|pattern| b.matches_type(pattern));

        // Preferred models form the first group, then each group is memory-first.
        b_is_preferred.cmp(&a_is_preferred).then_with(|| {
            b.memory_free_mb()
                .cmp(&a.memory_free_mb())
                // Secondary: Fewest processes (ascending)
                .then_with(|| a.process_count.cmp(&b.process_count))
                // Tertiary: Lowest index (ascending)
                .then_with(|| a.index.cmp(&b.index))
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gpu(index: usize, memory_used_mb: u64) -> GpuInfo {
        GpuInfo {
            index,
            name: format!("Test GPU {index}"),
            memory_used_mb,
            memory_total_mb: 24_000,
            utilization_percent: 0,
            process_count: usize::from(memory_used_mb >= 500),
            hidden_usage_mb: 0,
        }
    }

    fn locks() -> LockManager {
        LockManager::new(
            std::env::temp_dir().join(format!("with-gpu-selector-tests-{}", std::process::id())),
        )
    }

    #[test]
    fn manual_selection_preserves_all_indices_and_their_order() {
        let gpus = vec![make_gpu(10_001, 1_000), make_gpu(10_002, 2_000)];
        let criteria = SelectionCriteria {
            max_gpus: 1,
            min_memory_mb: Some(0),
            ..SelectionCriteria::default()
        };

        let selection = select_manual_gpus(&gpus, &[10_002, 10_001], &criteria, &locks()).unwrap();

        assert_eq!(selection.gpu_indices, vec![10_002, 10_001]);
    }

    #[test]
    fn manual_selection_rejects_duplicate_indices() {
        let gpus = vec![make_gpu(10_001, 0)];

        let error = select_manual_gpus(
            &gpus,
            &[10_001, 10_001],
            &SelectionCriteria::default(),
            &locks(),
        )
        .unwrap_err();

        assert!(error.to_string().contains("listed more than once"));
    }

    #[test]
    fn preferred_gpu_type_falls_back_when_match_is_unusable() {
        let mut preferred = make_gpu(0, 23_500);
        preferred.name = "RTX 4090".to_string();
        let mut fallback = make_gpu(1, 0);
        fallback.name = "RTX 3090".to_string();
        let criteria = SelectionCriteria {
            gpu_type_pattern: Some("4090".to_string()),
            ..SelectionCriteria::default()
        };

        let selection = select_gpus(&[preferred, fallback], &criteria, &locks()).unwrap();
        assert_eq!(selection.gpu_indices, vec![1]);
    }

    #[test]
    fn strict_gpu_type_does_not_fall_back() {
        let gpu = make_gpu(0, 0);
        let criteria = SelectionCriteria {
            gpu_type_pattern: Some("A100".to_string()),
            require_type_match: true,
            ..SelectionCriteria::default()
        };

        assert!(select_gpus(&[gpu], &criteria, &locks())
            .unwrap_err()
            .to_string()
            .contains("matching type"));
    }

    #[test]
    fn manual_selection_applies_thresholds() {
        let gpus = vec![make_gpu(10_001, 23_000)];

        let error = select_manual_gpus(&gpus, &[10_001], &SelectionCriteria::default(), &locks())
            .unwrap_err();

        assert!(error.to_string().contains("below the required 2048 MB"));
    }

    #[test]
    fn automatic_selection_is_invariant_to_discovery_order() {
        let gpus = [
            make_gpu(10_010, 10_000),
            make_gpu(10_011, 0),
            make_gpu(10_012, 500),
        ];
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let criteria = SelectionCriteria {
            min_gpus: 3,
            max_gpus: 3,
            min_memory_mb: Some(0),
            ..SelectionCriteria::default()
        };

        for permutation in permutations {
            let discovered = permutation.map(|index| gpus[index].clone());
            let selection = select_gpus(&discovered, &criteria, &locks()).unwrap();
            assert_eq!(selection.gpu_indices, vec![10_011, 10_012, 10_010]);
        }
    }

    #[test]
    fn generated_min_max_ranges_select_every_available_gpu_up_to_max() {
        for available in 1..=4 {
            let gpus = (0..available)
                .map(|offset| make_gpu(10_100 + offset, offset as u64 * 100))
                .collect::<Vec<_>>();
            for min_gpus in 1..=4 {
                for max_gpus in min_gpus..=4 {
                    let criteria = SelectionCriteria {
                        min_gpus,
                        max_gpus,
                        min_memory_mb: Some(0),
                        ..SelectionCriteria::default()
                    };
                    let result = select_gpus(&gpus, &criteria, &locks());

                    if available < min_gpus {
                        assert!(result.is_err());
                    } else {
                        assert_eq!(result.unwrap().gpu_indices.len(), available.min(max_gpus));
                    }
                }
            }
        }
    }

    #[test]
    fn threshold_boundaries_are_inclusive() {
        let mut gpu = make_gpu(10_200, 20_000);
        gpu.utilization_percent = 70;
        gpu.hidden_usage_mb = HIDDEN_USAGE_THRESHOLD_MB;
        let criteria = SelectionCriteria {
            min_memory_mb: Some(4_000),
            max_utilization: Some(70),
            ..SelectionCriteria::default()
        };
        assert!(select_gpus(&[gpu.clone()], &criteria, &locks()).is_ok());

        let cases = [
            {
                let mut candidate = gpu.clone();
                candidate.memory_used_mb += 1;
                candidate
            },
            {
                let mut candidate = gpu.clone();
                candidate.utilization_percent += 1;
                candidate
            },
            {
                let mut candidate = gpu;
                candidate.hidden_usage_mb += 1;
                candidate
            },
        ];
        for candidate in cases {
            assert!(select_gpus(&[candidate], &criteria, &locks()).is_err());
        }
    }

    #[test]
    fn preferred_type_fills_remaining_slots_with_fallbacks() {
        let mut preferred = make_gpu(10_300, 10_000);
        preferred.name = "RTX 4090".to_string();
        let fallbacks = [make_gpu(10_301, 0), make_gpu(10_302, 1_000)];
        let criteria = SelectionCriteria {
            min_gpus: 1,
            max_gpus: 3,
            min_memory_mb: Some(0),
            gpu_type_pattern: Some("4090".to_string()),
            ..SelectionCriteria::default()
        };

        let selection = select_gpus(
            &[preferred, fallbacks[0].clone(), fallbacks[1].clone()],
            &criteria,
            &locks(),
        )
        .unwrap();

        assert_eq!(selection.gpu_indices, vec![10_300, 10_301, 10_302]);
    }
}

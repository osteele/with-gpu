#[cfg(not(target_os = "macos"))]
mod cuda;
mod lockfile;
mod nvidia;
mod selector;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use with_gpu::{GpuInfo, GpuSelection};

#[derive(Parser, Debug)]
#[command(
    name = "with-gpu",
    version,
    about = "Intelligent GPU selection wrapper for CUDA commands",
    long_about = "Automatically selects idle GPUs or allows manual GPU selection via CUDA_VISIBLE_DEVICES.\n\n\
                  Examples:\n  \
                  with-gpu just train-tc tiny\n  \
                  with-gpu --gpu 1 python train.py\n  \
                  with-gpu --min-gpus 2 --max-gpus 4 torchrun train.py\n  \
                  with-gpu --wait --timeout 300 python train.py\n  \
                  with-gpu --status"
)]
struct Cli {
    #[arg(long, help = "Manual GPU selection (e.g., '1' or '0,1,2')")]
    gpu: Option<String>,

    #[arg(long, default_value = "1", help = "Minimum number of GPUs required")]
    min_gpus: usize,

    #[arg(long, default_value = "1", help = "Maximum number of GPUs to use")]
    max_gpus: usize,

    #[arg(
        long,
        help = "Require all selected GPUs to be idle (no processes running)"
    )]
    require_idle: bool,

    #[arg(
        long,
        help = "Minimum free memory required in MB (default: 2048 MB for PyTorch)\n\
                Use --min-memory 0 to disable and allow any GPU"
    )]
    min_memory: Option<u64>,

    #[arg(
        long,
        help = "Maximum GPU utilization percentage (0-100)\n\
                Example: --max-util 70 excludes GPUs with >70% utilization"
    )]
    max_util: Option<u8>,

    #[arg(
        long,
        help = "Prefer GPU model names containing this text (case-insensitive)"
    )]
    gpu_type: Option<String>,

    #[arg(
        long,
        requires = "gpu_type",
        help = "Require --gpu-type to match instead of falling back to another model"
    )]
    strict: bool,

    #[arg(
        long,
        help = "Wait for GPUs to become available if not immediately available"
    )]
    wait: bool,

    #[arg(
        long,
        help = "Timeout in seconds when waiting for GPUs (default: no timeout)",
        requires = "wait"
    )]
    timeout: Option<u64>,

    #[arg(long, help = "Show GPU status and exit")]
    status: bool,

    #[arg(long, requires = "status", help = "Output --status data as JSON")]
    json: bool,

    #[arg(
        trailing_var_arg = true,
        allow_hyphen_values = true,
        help = "Command to execute with selected GPUs"
    )]
    command: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.min_gpus > cli.max_gpus {
        anyhow::bail!(
            "min-gpus ({}) cannot be greater than max-gpus ({})",
            cli.min_gpus,
            cli.max_gpus
        );
    }

    if let Some(util) = cli.max_util {
        if util > 100 {
            anyhow::bail!("max-util must be between 0 and 100, got {}", util);
        }
    }

    let gpus = nvidia::query_gpus()?;

    if cli.status {
        print_status(&gpus, cli.json)?;
        return Ok(());
    }

    if cli.command.is_empty() {
        anyhow::bail!("No command specified (use --help for usage)");
    }

    // On macOS, skip GPU selection entirely and just execute the command
    #[cfg(target_os = "macos")]
    {
        if gpus.is_empty() {
            // Only warn if user explicitly requested GPU features beyond defaults
            let has_non_default_flags = cli.gpu.is_some()
                || cli.min_gpus != 1
                || cli.max_gpus != 1
                || cli.require_idle
                || cli.wait;

            if has_non_default_flags {
                eprintln!(
                    "Warning: GPU selection flags ignored on macOS (no NVIDIA GPUs available)"
                );
                eprintln!();
            }
            return execute_command_without_gpus(&cli.command);
        }
    }

    let criteria = selector::SelectionCriteria {
        min_gpus: cli.min_gpus,
        max_gpus: cli.max_gpus,
        require_idle: cli.require_idle,
        min_memory_mb: cli.min_memory.or(Some(2048)),
        max_utilization: cli.max_util,
        gpu_type_pattern: cli.gpu_type.clone(),
        require_type_match: cli.strict,
    };

    // Parse manual GPU selection if provided
    let manual_gpu_indices = if let Some(ref manual_selection) = cli.gpu {
        let indices = selector::parse_manual_gpu_selection(manual_selection)?;
        selector::validate_manual_gpu_selection(&gpus, &indices)?;
        Some(indices)
    } else {
        None
    };

    let (selection, display_gpus) = if cli.wait {
        wait_for_gpus(&criteria, cli.timeout, manual_gpu_indices.as_deref())?
    } else {
        let selection = if let Some(ref indices) = manual_gpu_indices {
            selector::select_manual_gpus(&gpus, indices, &criteria)?
        } else {
            selector::select_gpus(&gpus, &criteria)?
        };
        (selection, gpus)
    };

    print_selection(&display_gpus, &selection);

    // Keep each claim alive until exec transfers its file descriptor to the command.
    let mut claims = Vec::with_capacity(selection.gpu_indices.len());
    for &gpu_index in &selection.gpu_indices {
        match lockfile::claim_gpu(gpu_index) {
            Ok(claim) => claims.push(claim),
            Err(error) => {
                anyhow::bail!(
                    "Failed to claim GPU {}: {} (try again, another process may have claimed it)",
                    gpu_index,
                    error
                );
            }
        }
    }

    let result = execute_command(&cli.command, &selection);
    drop(claims);
    result
}

fn wait_for_gpus(
    criteria: &selector::SelectionCriteria,
    timeout_secs: Option<u64>,
    manual_gpu_indices: Option<&[usize]>,
) -> Result<(GpuSelection, Vec<GpuInfo>)> {
    let start_time = Instant::now();
    let poll_interval = Duration::from_secs(5);
    let timeout = timeout_secs.map(Duration::from_secs);
    let mut attempt = 1;

    eprintln!("Waiting for GPUs to become available...");
    if let Some(timeout) = timeout_secs {
        eprintln!("  Timeout: {} seconds", timeout);
    }
    if let Some(indices) = manual_gpu_indices {
        eprintln!("  Manual selection: {:?}", indices);
    }
    eprintln!(
        "  Requirements: min={}, max={}, require_idle={}",
        criteria.min_gpus, criteria.max_gpus, criteria.require_idle
    );
    eprintln!();

    loop {
        let all_gpus = nvidia::query_gpus()?;

        let selection_result = if let Some(indices) = manual_gpu_indices {
            selector::select_manual_gpus(&all_gpus, indices, criteria)
        } else {
            selector::select_gpus(&all_gpus, criteria)
        };

        match selection_result {
            Ok(selection) => {
                eprintln!(
                    "GPUs available after {} attempts ({:.1}s)",
                    attempt,
                    start_time.elapsed().as_secs_f64()
                );
                return Ok((selection, all_gpus));
            }
            Err(e) => {
                if let Some(timeout) = timeout {
                    let elapsed = start_time.elapsed();
                    if elapsed >= timeout {
                        anyhow::bail!(
                            "Timeout after {:.1} seconds waiting for GPUs: {}",
                            elapsed.as_secs_f64(),
                            e
                        );
                    }
                }

                eprintln!(
                    "[Attempt {}] No suitable GPUs available (waited {:.0}s)",
                    attempt,
                    start_time.elapsed().as_secs_f64()
                );

                let displayed_gpus: Vec<&GpuInfo> = if let Some(indices) = manual_gpu_indices {
                    all_gpus
                        .iter()
                        .filter(|gpu| indices.contains(&gpu.index))
                        .collect()
                } else {
                    all_gpus.iter().collect()
                };
                let idle_count = displayed_gpus.iter().filter(|gpu| gpu.is_idle()).count();
                eprintln!("  Idle GPUs: {}/{}", idle_count, displayed_gpus.len());

                if idle_count > 0 {
                    eprintln!(
                        "  Idle GPU indices: {:?}",
                        displayed_gpus
                            .iter()
                            .filter(|g| g.is_idle())
                            .map(|g| g.index)
                            .collect::<Vec<_>>()
                    );
                }

                let sleep_duration =
                    polling_sleep_duration(start_time.elapsed(), timeout, poll_interval);
                thread::sleep(sleep_duration);
                attempt += 1;
            }
        }
    }
}

fn polling_sleep_duration(
    elapsed: Duration,
    timeout: Option<Duration>,
    poll_interval: Duration,
) -> Duration {
    timeout
        .map(|timeout| poll_interval.min(timeout.saturating_sub(elapsed)))
        .unwrap_or(poll_interval)
}

#[derive(Serialize)]
struct GpuStatus<'a> {
    #[serde(flatten)]
    gpu: &'a GpuInfo,
    is_idle: bool,
    memory_free_mb: u64,
    memory_usage_percent: f64,
    claimed_by_pid: Option<u32>,
}

fn print_status(gpus: &[GpuInfo], json: bool) -> Result<()> {
    if gpus.is_empty() {
        #[cfg(target_os = "macos")]
        {
            println!("No NVIDIA GPUs available (running on macOS)");
            println!("Commands will execute without GPU selection.");
            return Ok(());
        }
        #[cfg(not(target_os = "macos"))]
        {
            println!("No GPUs detected");
            return Ok(());
        }
    }

    let claimed_gpus = lockfile::get_claimed_gpus();

    if json {
        let statuses = gpus
            .iter()
            .map(|gpu| GpuStatus {
                gpu,
                is_idle: gpu.is_idle(),
                memory_free_mb: gpu.memory_free_mb(),
                memory_usage_percent: gpu.memory_usage_percent(),
                claimed_by_pid: claimed_gpus
                    .iter()
                    .find(|(index, _)| *index == gpu.index)
                    .map(|(_, pid)| *pid),
            })
            .collect::<Vec<_>>();
        println!("{}", serde_json::to_string_pretty(&statuses)?);
        return Ok(());
    }

    println!("Available GPUs:");
    for gpu in gpus {
        let claim_info = claimed_gpus
            .iter()
            .find(|(idx, _)| *idx == gpu.index)
            .map(|(_, pid)| {
                if *pid == 0 {
                    " [claimed]".to_string()
                } else {
                    format!(" [claimed by pid {}]", pid)
                }
            })
            .unwrap_or_default();
        println!("  {}{}", gpu, claim_info);
    }

    if !claimed_gpus.is_empty() {
        println!();
        println!(
            "Note: {} GPU(s) claimed by other with-gpu processes",
            claimed_gpus.len()
        );
    }
    Ok(())
}

fn print_selection(gpus: &[GpuInfo], selection: &GpuSelection) {
    eprintln!("Selected GPU(s): {}", selection.to_cuda_visible_devices());

    for &index in &selection.gpu_indices {
        if let Some(gpu) = gpus.iter().find(|g| g.index == index) {
            let free_gb = gpu.memory_free_mb() as f64 / 1024.0;

            if gpu.memory_free_mb() < 2048 {
                eprintln!(
                    "Warning: GPU {} has only {:.2} GB free (< 2 GB recommended for PyTorch)",
                    index, free_gb
                );
            }

            eprintln!("  {}", gpu);
        }
    }

    if let Some(warning) = &selection.warning {
        eprintln!("\n{}", warning);
    }

    eprintln!();
}

fn execute_command(command_parts: &[String], selection: &GpuSelection) -> Result<()> {
    if command_parts.is_empty() {
        anyhow::bail!("No command specified");
    }

    let program = &command_parts[0];
    let args = &command_parts[1..];

    let cuda_visible_devices = selection.to_cuda_visible_devices();

    #[cfg(unix)]
    {
        let error = Command::new(program)
            .args(args)
            .env("CUDA_VISIBLE_DEVICES", cuda_visible_devices)
            .exec();

        Err(error).context(format!("Failed to execute command: {}", program))
    }

    #[cfg(not(unix))]
    {
        let status = Command::new(program)
            .args(args)
            .env("CUDA_VISIBLE_DEVICES", cuda_visible_devices)
            .status()
            .context(format!("Failed to execute command: {}", program))?;

        if !status.success() {
            anyhow::bail!("Command exited with status: {}", status);
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn execute_command_without_gpus(command_parts: &[String]) -> Result<()> {
    if command_parts.is_empty() {
        anyhow::bail!("No command specified");
    }

    let program = &command_parts[0];
    let args = &command_parts[1..];

    let error = Command::new(program).args(args).exec();

    Err(error).context(format!("Failed to execute command: {}", program))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polling_sleep_does_not_overshoot_timeout() {
        let delay = polling_sleep_duration(
            Duration::from_millis(250),
            Some(Duration::from_secs(1)),
            Duration::from_secs(5),
        );

        assert_eq!(delay, Duration::from_millis(750));
    }
}

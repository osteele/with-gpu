# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Lock file coordination to prevent race conditions when multiple `with-gpu` processes start simultaneously
- `--status` now shows GPUs claimed by other `with-gpu` processes
- Hidden memory usage detection: compares total GPU memory used vs. memory attributed to visible processes
- GPUs with significant unattributed memory (>512 MB) are now marked as "suspected hidden usage" and filtered out

### Changed
- `--gpu` now respects `--min-memory`, `--max-util`, `--require-idle`, and `--wait`/`--timeout` flags
- Manual GPU selection is now composable with all filtering and wait options
- `is_idle()` now also checks for hidden memory usage

### Fixed
- Lock file validation now correctly handles EPERM (other user's process) instead of treating it as stale
- `--status` now dynamically enumerates lock files instead of hardcoding GPU indices 0-15
- GPU selection now detects "ghost" processes that use GPU memory but don't appear in NVML's process list (fixes stale nvidia-smi data bug)

## [0.3.0] - 2025-12-02

### Added
- `--version` / `-V` flag to display version information
- `--min-memory` flag to specify minimum free memory in MB (default: 2048 MB)
- `--max-util` flag to filter GPUs by maximum utilization percentage
- Warning when selected GPU has less than 2 GB free memory
- Documentation about ghost process detection threshold (500 MB)

### Changed
- Default behavior now requires 2 GB free memory (PyTorch-friendly default)
- Users can disable with `--min-memory 0` for small workloads
- Threshold filtering applied before GPU selection for better control

### Fixed
- Idle detection now checks memory usage in addition to process count to prevent selecting GPUs with allocated memory
- Prevents OOM errors when NVML process detection misses GPU-using processes (e.g., with persistence mode or MPS)

## [0.2.0] - 2025-11-20

### Added
- macOS compatibility with conditional NVML dependency compilation
- Memory-first GPU selection algorithm that prioritizes GPUs with most available VRAM
- Graceful handling for platforms without GPU support
- Enhanced error messages and timeout handling
- New `docs/limitations.md` documenting race conditions, fairness, and design trade-offs

### Changed
- GPU selection now prioritizes available memory to prevent OOM errors
- Fallback criteria now include: most free memory → fewest running processes → lowest GPU index
- `--require-idle` flag now sorts idle GPUs by available memory
- Improved development documentation with direct `cargo` commands
- Enhanced README with limitations section and clearer examples

### Fixed
- macOS build issues by conditionally excluding `nvml-wrapper` on macOS
- Command execution to properly handle cases with no GPUs available
- Status output to gracefully inform about absence of GPUs

### Removed
- Obsolete `justfile` in favor of direct `cargo` commands

## [0.1.0] - 2025-11-18

### Added
- Initial release of with-gpu
- Intelligent GPU selection based on idle status and process count
- Support for `--gpu`, `--min-gpus`, `--require-idle` flags
- Wait and timeout functionality for GPU availability
- Status command to query current GPU state
- NVML-based GPU querying for reliability
- Process replacement via `exec()` to preserve stdio

# Development Guide

This guide covers the development workflow, architecture, and guidelines for developing with-gpu.

## Architecture

### Module Organization

```
src/
├── main.rs        # CLI entry point (clap), command execution
├── lib.rs         # Shared types (GpuInfo, GpuSelection)
├── nvidia.rs      # NVML library interface for GPU queries
└── selector.rs    # GPU selection algorithm
```

### Key Design Decisions

1. **NVML library**: Uses nvidia-ml library directly (not nvidia-smi command)
   - Queries GPU memory, utilization, and process counts via NVML
   - More reliable than parsing nvidia-smi output
   - Works even when nvidia-smi is replaced/wrapped (as on cool30)

2. **Selection algorithm** (memory-first):
   - **Primary criterion**: Most available memory (free VRAM in MB, descending)
   - **Secondary criterion**: Fewest running processes (ascending)
   - **Tertiary criterion**: Lowest GPU index (ascending)
   - Exception: `--require-idle` restricts to GPUs with 0 processes (still sorted by memory)
   - Warn when using non-idle GPUs
   - Fail immediately or wait if requirements not met
   - **Rationale**: Prevents OOM errors. A GPU with 10 GB free and 1 process is more useful than an "idle" GPU with 300 MB free.

3. **Multi-GPU support**:
   - `--min-gpus`: Minimum required (default 1)
   - `--max-gpus`: Maximum to use (default 1)
   - Auto-select picks GPUs with most available memory first

4. **Wait/timeout support**:
   - `--wait`: Poll every 5 seconds until GPUs available
   - `--timeout N`: Fail after N seconds of waiting
   - Shows progress: attempt count, time waited, idle GPU count

5. **Command execution**: Use `std::process::Command::exec()` to replace current process
   - Preserves stdin/stdout/stderr
   - Sets `CUDA_VISIBLE_DEVICES` environment variable
   - Command receives full control of terminal

6. **Cross-platform support**:
   - **Linux**: Full functionality with NVML queries
   - **macOS**: No-op mode (executes command without GPU selection)
   - Uses conditional compilation (`#[cfg(target_os = "macos")]`) to handle platform differences
   - `nvml-wrapper` dependency only compiled on non-macOS platforms

## Development Workflow

### Making Changes

1. Edit source files in `src/`
2. Run quality checks:
   ```bash
   cargo fmt --check  # Check formatting
   cargo clippy -- -D warnings  # Lint
   cargo check  # Type check
   ```
3. Test locally if you have NVML/CUDA (or test on cool30 after sync)
4. Commit changes with git

### Adding New Features

When adding features:
- Keep CLI interface simple and composable
- Preserve backward compatibility (existing flags work as before)
- Test thoroughly with actual GPU workloads
- Update README.md with user-facing examples
- Update this document with design decisions

### Code Quality

Enforced by cargo tooling:
- **cargo fmt**: Code formatting
- **cargo clippy**: Linting (with `-D warnings` to fail on warnings)
- **cargo check**: Type checking and compilation

Run all checks:
```bash
cargo fmt --check && cargo clippy -- -D warnings && cargo check
```

## Testing

### Current State

This project currently has **no automated unit tests** (`cargo test` runs 0 tests).

**Why No Tests?**
- GPU selection logic requires NVML library and actual GPU hardware
- NVML cannot be easily mocked (C library with complex state)
- Would need test doubles for entire NVML interface

**Testing Approach:**

**Manual testing on real hardware** (machine with NVIDIA GPUs, e.g., 8x RTX 3090):
1. Test threshold filtering with various `--min-memory` and `--max-util` values
2. Test manual selection with `--gpu`
3. Test wait behavior with `--wait --timeout`
4. Test edge cases: no GPUs, all GPUs busy, invalid GPU IDs

**Future Improvements:**
- Consider integration tests that run on CI machines with GPUs
- Mock NVML for unit tests of selection logic (complex but possible)
- Add property-based tests for selection algorithms

### Local Testing (macOS)

On macOS, the tool runs in no-op mode (executes commands without GPU selection). You can:
- Build the binary: `cargo build --release`
- Check code quality: `cargo fmt --check && cargo clippy -- -D warnings && cargo check`
- Test basic execution: `with-gpu echo "test"` (should just execute without warnings)
- Test with flags: `with-gpu --gpu 0 echo "test"` (should show warning about macOS)
- Test status: `with-gpu --status` (should show "No NVIDIA GPUs available (running on macOS)")

### Testing on Linux with NVIDIA GPUs

If you have access to a Linux machine with NVIDIA GPUs:

```bash
# Install the tool
cd ~/code/with-gpu
cargo install --path .

# Test status display
with-gpu --status

# Test auto-selection
with-gpu echo "Auto-selected GPU"

# Test manual selection
with-gpu --gpu 1 echo "Using GPU 1"

# Test wait functionality
with-gpu --wait --timeout 10 echo "Wait test"

# Test with actual training
with-gpu python train.py
```

## Style Guidelines

### Rust Style

- Use `cargo fmt` for formatting
- Follow clippy suggestions (warnings treated as errors)
- Prefer explicit error handling with `anyhow::Result`
- Use `?` operator for error propagation
- Avoid `unwrap()` except in examples or when provably safe

### Error Messages

- Provide actionable error messages
- Suggest `--status` when GPU selection fails
- Include context (e.g., "Need 2 GPUs but only 1 available")

### CLI Design

- Short flags for common options (e.g., `--gpu`)
- Long descriptive flags for clarity
- Use `--help` to show comprehensive usage
- Support `--` to separate tool flags from command args

## Common Development Tasks

### Installing the Tool

```bash
cd ~/code/research/with-gpu
cargo install --path .
```

Installs binary to `~/.cargo/bin/with-gpu` (ensure `~/.cargo/bin` is in PATH).

### Updating After Changes

```bash
# Verify code quality
cargo fmt --check && cargo clippy -- -D warnings && cargo check

# Reinstall to ~/.cargo/bin
cargo install --path .
```

### Checking GPU Status

```bash
with-gpu --status
```

Shows all GPUs with memory usage, utilization, and process counts.

### Quick Testing

Local testing commands (if you have NVIDIA GPUs with NVML/CUDA):

```bash
# Show GPU status
cargo run -- --status

# Test with simple command
cargo run -- echo "Testing with-gpu wrapper"

# Test manual GPU selection
cargo run -- --gpu 0 echo "Using GPU 0"
```

For comprehensive testing, use a Linux machine with NVIDIA GPUs (see Testing section above).

## Troubleshooting

### NVML Library Not Found

Error: "Failed to initialize NVML (is the NVIDIA driver installed?)"

Solution:
- Ensure you're on a machine with NVIDIA GPUs
- Check that NVML library exists: `ldconfig -p | grep nvidia-ml`
- The library is typically at `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so` on Linux

### GPU Selection Fails

Error: "Need N GPUs but only M available"

Solution:
- Run `with-gpu --status` to see GPU state
- Reduce `--min-gpus` or use `--wait` to wait for GPUs to become available
- Use `--gpu` for manual selection if you know which GPU to use

### Command Not Executing

If the command doesn't run:
- Check that the command exists in PATH
- Use full path if needed: `with-gpu /full/path/to/command`
- Verify `CUDA_VISIBLE_DEVICES` is set (command should see it)

### Compilation Errors

If you get lifetime errors or type mismatches:
- Run `cargo clean` and rebuild
- Check that you're using compatible versions of dependencies
- Refer to `Cargo.toml` for correct dependency versions

## Possible Future Enhancements

Potential improvements:
- Support for AMD GPUs (ROCm)
- Configuration file for default settings
- GPU affinity pinning (CPU-GPU locality)
- Multi-node GPU selection (for distributed training)
- Integration with SLURM or other job schedulers
- Configurable polling interval for `--wait`
- Email/notification when GPUs become available

## References

- NVIDIA NVML documentation: https://docs.nvidia.com/deploy/nvml-api/
- NVIDIA CUDA documentation: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Rust clap documentation: https://docs.rs/clap/latest/clap/
- nvml-wrapper crate: https://docs.rs/nvml-wrapper/

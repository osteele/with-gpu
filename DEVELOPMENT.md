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

2. **Selection algorithm**:
   - Prefer idle GPUs (0 processes)
   - Fall back to least-used GPUs (sorted by memory used, then process count)
   - Warn when using non-idle GPUs
   - Fail immediately or wait if requirements not met

3. **Multi-GPU support**:
   - `--min-gpus`: Minimum required (default 1)
   - `--max-gpus`: Maximum to use (default 1)
   - Auto-select fills with idle first, then least-used

4. **Wait/timeout support**:
   - `--wait`: Poll every 5 seconds until GPUs available
   - `--timeout N`: Fail after N seconds of waiting
   - Shows progress: attempt count, time waited, idle GPU count

5. **Command execution**: Use `std::process::Command::exec()` to replace current process
   - Preserves stdin/stdout/stderr
   - Sets `CUDA_VISIBLE_DEVICES` environment variable
   - Command receives full control of terminal

## Development Workflow

### Making Changes

1. Edit source files in `src/`
2. Run `just all-checks` to verify code quality
3. Test locally if you have NVML/CUDA (or test on cool30 after sync)
4. Commit changes with git

### Adding New Features

When adding features:
- Keep CLI interface simple and composable
- Preserve backward compatibility (existing flags work as before)
- Add tests in justfile for new functionality
- Update README.md with user-facing examples
- Update this document with design decisions

### Code Quality

Enforced by cargo tooling:
- **cargo fmt**: Code formatting
- **cargo clippy**: Linting (with `-D warnings` to fail on warnings)
- **cargo check**: Type checking and compilation

Run all checks with: `just all-checks`

## Testing

### Local Testing (macOS without NVML)

Most functionality can't be tested on macOS since there's no NVML. You can:
- Build the binary: `just build`
- Check code quality: `just all-checks`

### Testing on cool30

After mutagen sync, test on cool30:

```bash
ssh cool30
cd ~/code/research/with-gpu
just install

# Test status display
with-gpu --status

# Test auto-selection
with-gpu echo "Auto-selected GPU"

# Test manual selection
with-gpu --gpu 1 echo "Using GPU 1"

# Test wait functionality
with-gpu --wait --timeout 10 echo "Wait test"

# Test with actual training
cd ~/code/research/linebreak-transformer
with-gpu just train-tc tiny
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
just install
```

Installs binary to `~/.cargo/bin/with-gpu` (ensure `~/.cargo/bin` is in PATH).

### Updating After Changes

```bash
just all-checks  # Verify code quality
just install     # Reinstall to ~/.cargo/bin
```

### Checking GPU Status

```bash
with-gpu --status
```

Shows all GPUs with memory usage, utilization, and process counts.

### Quick Testing

Local testing commands (if you have NVML/CUDA):

```bash
just status          # Show GPU status via with-gpu
just test-echo       # Test with simple echo command
just test-manual 0   # Test manual GPU selection
```

For comprehensive testing on machines with actual GPUs, use cool30 (see Testing section above).

## Troubleshooting

### NVML Library Not Found

Error: "Failed to initialize NVML (is the NVIDIA driver installed?)"

Solution:
- Ensure you're on a machine with NVIDIA GPUs
- Check that NVML library exists: `ldconfig -p | grep nvidia-ml`
- On cool30, the library should be at `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so`

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

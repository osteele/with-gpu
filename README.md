# with-gpu

[![Crates.io](https://img.shields.io/crates/v/with-gpu.svg)](https://crates.io/crates/with-gpu)
[![CI](https://github.com/osteele/with-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/osteele/with-gpu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

<p align="center">
  <img src="docs/mascot.png" alt="with-gpu mascot" width="300" />
</p>

Intelligent GPU selection wrapper for CUDA commands. Automatically selects GPUs with the most available memory, then sets `CUDA_VISIBLE_DEVICES` and executes your command.

## Features

- 🧠 **Memory-first selection**: Prioritizes GPUs with the most available VRAM
- 🎯 **Explicit idle filtering**: Includes non-idle GPUs unless `--require-idle` is set
- 🖥️ **Multi-GPU support**: Request minimum and maximum number of GPUs
- 🏷️ **GPU type filtering**: Prefer or require a GPU model by name
- 🎛️ **Manual selection**: Specify exact GPU IDs when needed
- ⏱️ **Wait capability**: Poll for GPU availability with configurable timeout
- 📊 **Status display**: View model names and usage as text or JSON
- ⚠️ **Warning messages**: Get notified when using non-idle GPUs
- 🔒 **Cooperative claims**: Prevents concurrent `with-gpu` commands from selecting the same GPU
- 🍎 **Cross-platform**: Works on Linux (with NVIDIA GPUs) and macOS (no-op mode)

## Installation

Install from [crates.io](https://crates.io/crates/with-gpu):

```bash
cargo install with-gpu
```

This installs `with-gpu` to `~/.cargo/bin/with-gpu` (ensure `~/.cargo/bin` is in your PATH).

### Build from Source

```bash
git clone https://github.com/osteele/with-gpu.git
cd with-gpu
cargo install --path .
```

## Usage

### Basic Usage (Auto-select)

Select the GPU with most available memory:

```bash
with-gpu python train.py
```

This prioritizes available VRAM over idle status. A used GPU with more free
memory can rank ahead of an idle GPU.

### Manual GPU Selection

Specify exact GPU ID(s):

```bash
# Single GPU
with-gpu --gpu 1 python train.py

# Multiple GPUs
with-gpu --gpu 0,1 python train.py
with-gpu --gpu 0,1,2,3 torchrun --nproc_per_node=4 train.py
```

Manual IDs are preserved exactly, including their order. Automatic `--min-gpus` and
`--max-gpus` ranking does not apply, but availability, memory, utilization, idle,
and wait filters still do.

### Multi-GPU Auto-selection

Request a range of GPUs:

```bash
# Need exactly 2 GPUs
with-gpu --min-gpus 2 python train.py

# Want 1-4 GPUs (use as many suitable GPUs as available, up to 4)
with-gpu --max-gpus 4 python train.py

# Need at least 2, prefer up to 4
with-gpu --min-gpus 2 --max-gpus 4 python train.py
```

### Prefer a GPU Model

Prefer model names containing a case-insensitive substring, with fallback to any
otherwise suitable GPU:

```bash
with-gpu --gpu-type 4090 python train.py
```

Add `--strict` to fail (or continue waiting) unless that model is available:

```bash
with-gpu --gpu-type A100 --strict --wait python train.py
```

### Require Idle GPUs

Enforce idle-only selection (no non-idle GPUs even if they have more free memory):

```bash
# Single idle GPU required
with-gpu --require-idle python train.py

# Exactly 2 idle GPUs required
with-gpu --min-gpus 2 --max-gpus 2 --require-idle python train.py
```

**Note**: Without `--require-idle`, the tool selects GPUs by available memory regardless of idle status. Use this flag when you specifically need GPUs with 0 running processes.

### Memory and Utilization Thresholds

Filter GPUs by available memory and utilization:

```bash
# Require at least 8 GB free memory (default is 2 GB)
with-gpu --min-memory 8000 python train.py

# Allow any GPU with free memory (disable 2 GB default)
with-gpu --min-memory 0 python small_inference.py

# Require GPU utilization below 70%
with-gpu --max-util 70 python train.py

# Combine thresholds: 16 GB free + max 50% utilization
with-gpu --min-memory 16000 --max-util 50 python train_llm.py
```

**Default behavior**: By default, `with-gpu` requires at least 2 GB free memory.
This avoids GPUs that are almost full, but the required memory depends on the
workload. For small jobs that need less, use `--min-memory 0`.

**Idle and hidden usage checks**: A GPU is idle when NVML reports no running
compute processes and total used memory is below 500 MB. Separately, `with-gpu`
excludes any GPU with more than 512 MB of memory that cannot be attributed to
visible NVML processes. This catches GPU usage that NVML's process list missed.

### Wait for GPUs

Wait for GPUs to become available instead of failing immediately:

```bash
# Wait indefinitely for an idle GPU
with-gpu --wait --require-idle python train.py

# Wait up to 300 seconds (5 minutes) for 2 idle GPUs
with-gpu --wait --timeout 300 --min-gpus 2 --require-idle python train.py

# Wait for 1-4 GPUs with 1 hour timeout
with-gpu --wait --timeout 3600 --max-gpus 4 python train.py
```

The tool polls every 5 seconds and shows:
- Number of attempts
- Time waited
- Current idle GPU count and indices

Selection and claiming happen in the same retry loop. If another `with-gpu`
process wins a claim race, a waiting process retries instead of failing.

### Cooperative Claim Directory

Claims use `/tmp/with-gpu` by default on Unix and the operating system's
temporary directory on Windows. Set `--lock-dir PATH` or the `WITH_GPU_LOCK_DIR`
environment variable when containers or users need a different shared
namespace. On Unix, the directory is created with mode `1777`; if an older
directory cannot be migrated to those permissions, the error recommends using a
new lock directory.

### Check GPU Status

View all GPUs and their current usage:

```bash
with-gpu --status

# JSON array suitable for scripts
with-gpu --status --json
```

Output example:
```
Available GPUs:
  GPU 0: [NVIDIA GeForce RTX 3090] USED - 15320/24268 MB (63.1%), 85 util, 3 processes
  GPU 1: [NVIDIA GeForce RTX 3090] IDLE - 0/24268 MB (0.0%), 0 util, 0 processes
  GPU 2: [NVIDIA GeForce RTX 3090] USED - 5920/24268 MB (24.4%), 12 util, 1 processes
```

In this example, auto-selection would pick GPU 1 (24 GB free), then GPU 2 (18 GB free), then GPU 0 (9 GB free).

## How It Works

1. **Queries GPUs**: Uses NVML to get model names, utilization, and running
   processes. Memory usage comes from the CUDA Driver API when available, with
   NVML as a fallback.
2. **Threshold Filtering** (before selection):
   - Default: Requires 2 GB free memory (override with `--min-memory`)
   - Optional: Maximum utilization percentage (`--max-util`)
   - Filters GPUs before applying memory-first selection
3. **Selection Algorithm**:
   - **Primary criterion**: Most available memory (free VRAM in MB, descending)
   - **Secondary criterion**: Fewest running processes (ascending)
   - **Tertiary criterion**: Lowest GPU index (ascending)
4. **Special modes**:
   - `--require-idle`: Only considers GPUs with 0 processes and <500 MB total memory used (still sorted by available memory)
   - Manual `--gpu`: Preserves the exact requested IDs and order while applying filters
5. **Warnings**: Notifies when using non-idle GPUs or GPUs with <2 GB free
6. **Execution**: Sets `CUDA_VISIBLE_DEVICES`, then replaces the current process
   on Unix or waits for the child process and preserves its exit code on Windows

Memory-first ranking favors available capacity. A GPU with 10 GB free and 1
process can rank ahead of an idle GPU with less free memory. By default, GPUs
with less than 2 GB free are filtered out.

## Examples

### Training Workflows

```bash
# Auto-select GPU with most free memory
with-gpu python train.py

# Force use of GPU 1
with-gpu --gpu 1 python train.py

# Use 2 GPUs with most free memory for distributed training
with-gpu --min-gpus 2 --max-gpus 2 torchrun --nproc_per_node=2 train.py
```

### Research Workflows

```bash
# Run multiple experiments on different GPUs
with-gpu --gpu 0 python experiment_a.py &
with-gpu --gpu 1 python experiment_b.py &
with-gpu --gpu 2 python experiment_c.py &

# Only run if a GPU is completely free
with-gpu --require-idle python long_training.py

# Use up to 8 available idle GPUs
with-gpu --max-gpus 8 --require-idle python distributed_train.py
```

## Integration with Other Tools

Works with any command that respects `CUDA_VISIBLE_DEVICES`:

- **PyTorch** / **TensorFlow** training scripts
- **torchrun** for distributed training
- Any CUDA application

## Related Tools

**[`with-limits`](https://github.com/osteele/with-limits)** - Runs a command with portable process-tree limits on host memory, sustained CPU use, and wall-clock runtime. It can be combined with `with-gpu` when a workload needs both GPU selection and host resource containment.

**[`cuda-selector`](https://github.com/SamerMakni/cuda-selector)** - Python library for in-process GPU selection. Supports memory, power, temperature, and utilization criteria with custom ranking functions. For Python-only workflows where you want device selection within your script rather than as a CLI wrapper.

**`idlegpu`** - Simple shell utility returning idle GPU ID. No multi-GPU, fallback, or wait support.

**`gpustat`** / **`nvitop`** - Monitoring tools with rich status displays. Monitoring only, no command execution.

**SLURM** / **Kubernetes** - Enterprise job schedulers. Feature-rich but heavyweight, complex setup.

### Why `with-gpu`?

Fills the gap between simple utilities and full schedulers:
- ✅ Executes commands (not just monitoring)
- ✅ Memory-first selection with a 2 GB default free-memory floor
- ✅ Non-idle GPU selection when those GPUs have the most free memory
- ✅ Wait capability with timeout
- ✅ Multi-GPU min/max support
- ✅ Lightweight (single Rust binary)
- ✅ Direct NVML queries (reliable, not parsing nvidia-smi)
- ✅ Cross-platform (Linux + macOS + Windows)

**Best for**: Individual workstations, small research groups, "just run this on the GPU with most free memory" workflows.

## Limitations

- ❌ Programs that do not use `with-gpu` do not participate in cooperative claims
- ❌ Intermittent GPU usage may appear as idle
- ❌ Claims coordinate only processes on the same host and shared lock namespace
- ❌ No queue management or FIFO ordering
- ❌ No priority system for waiting processes
- ❌ No resource reservation or advance scheduling
- ❌ Not suitable for environments requiring fairness guarantees

**Mitigation**: Launch cooperating jobs through `with-gpu`, and use `--require-idle`
or `--wait` when external GPU activity is possible. See
[docs/limitations.md](docs/limitations.md) for details.

**When you need more**: For guaranteed fair scheduling, priority queues, or resource reservations, use SLURM or Kubernetes.

Designed for **cooperative environments** (small groups, personal workstations)
where lightweight GPU selection is sufficient.

## Requirements

**On Linux:**
- NVIDIA GPU(s)
- NVIDIA driver with NVML library (libnvidia-ml.so)
- Rust toolchain for building

**On macOS:**
- Rust toolchain for building
- Commands execute normally without GPU selection. This is in order to use `with-gpu` in cross-platform scripts.

**On Windows:**
- NVIDIA GPU(s)
- NVIDIA driver with NVML library (`nvml.dll`)
- Rust toolchain for building

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for development documentation including:
- Development workflow and code quality standards
- Testing procedures
- Style guidelines
- Troubleshooting common issues

See [DESIGN.md](DESIGN.md) for design rationale and architectural decisions.

See [ROADMAP.md](ROADMAP.md) for planned features and future directions.

## Author

Oliver Steele <steele@osteele.com>

## License

Licensed under the [MIT License](LICENSE).

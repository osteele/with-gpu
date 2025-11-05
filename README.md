# with-gpu

Intelligent GPU selection wrapper for CUDA commands. Automatically finds idle GPUs or allows manual selection, then sets `CUDA_VISIBLE_DEVICES` and executes your command.

## Features

- 🎯 **Auto-select idle GPUs**: Prefers GPUs with no running processes
- 🔄 **Fallback to least-used**: If no idle GPUs, selects GPU with least memory usage
- 🖥️ **Multi-GPU support**: Request minimum and maximum number of GPUs
- 🎛️ **Manual selection**: Specify exact GPU IDs when needed
- ⏱️ **Wait capability**: Poll for GPU availability with configurable timeout
- 📊 **Status display**: View all GPUs and their current usage
- ⚠️ **Warning messages**: Get notified when using non-idle GPUs

## Installation

```bash
git clone https://github.com/osteele/with-gpu.git
cd with-gpu
cargo install --path .
```

This installs `with-gpu` to `~/.cargo/bin/with-gpu` (ensure `~/.cargo/bin` is in your PATH).

## Usage

### Basic Usage (Auto-select)

Find one idle GPU (or least-used if none idle):

```bash
with-gpu python train.py
```

### Manual GPU Selection

Specify exact GPU ID(s):

```bash
# Single GPU
with-gpu --gpu 1 python train.py

# Multiple GPUs
with-gpu --gpu 0,1 python train.py
with-gpu --gpu 0,1,2,3 torchrun --nproc_per_node=4 train.py
```

### Multi-GPU Auto-selection

Request a range of GPUs:

```bash
# Need exactly 2 GPUs
with-gpu --min-gpus 2 --max-gpus 2 python train.py

# Want 1-4 GPUs (use as many idle as available, up to 4)
with-gpu --max-gpus 4 python train.py

# Need at least 2, prefer up to 4
with-gpu --min-gpus 2 --max-gpus 4 python train.py
```

### Require Idle GPUs

Fail if no completely idle GPUs available:

```bash
# Single idle GPU required
with-gpu --require-idle python train.py

# Multiple idle GPUs required
with-gpu --min-gpus 2 --require-idle python train.py
```

### Wait for GPUs

Wait for GPUs to become available instead of failing immediately:

```bash
# Wait indefinitely for an idle GPU
with-gpu --wait python train.py

# Wait up to 300 seconds (5 minutes) for 2 idle GPUs
with-gpu --wait --timeout 300 --min-gpus 2 --require-idle python train.py

# Wait for 1-4 GPUs with 1 hour timeout
with-gpu --wait --timeout 3600 --max-gpus 4 python train.py
```

The tool polls every 5 seconds and shows:
- Number of attempts
- Time waited
- Current idle GPU count and indices

### Check GPU Status

View all GPUs and their current usage:

```bash
with-gpu --status
```

Output example:
```
Available GPUs:
  GPU 0: USED - 15320/24268 MB (63.1%), 85 util, 3 processes
  GPU 1: IDLE - 0/24268 MB (0.0%), 0 util, 0 processes
  GPU 2: USED - 5920/24268 MB (24.4%), 12 util, 1 processes
```

## How It Works

1. Queries NVIDIA GPUs via NVML library for memory, utilization, and running processes
2. Classifies GPUs as **idle** (0 processes) or **used** (1+ processes)
3. Selects GPUs based on criteria:
   - Prefer idle GPUs up to `--max-gpus`
   - If not enough idle, add least-used GPUs
   - Warn when using non-idle GPUs
   - Fail if `< --min-gpus` available or `--require-idle` not satisfied
4. Sets `CUDA_VISIBLE_DEVICES` environment variable
5. Executes your command

## Examples

### Training Workflows

```bash
# Auto-select one GPU (avoid busy GPUs)
with-gpu python train.py

# Force use of GPU 1
with-gpu --gpu 1 python train.py

# Use 2 GPUs for distributed training
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

# Use all available idle GPUs
with-gpu --max-gpus 8 python distributed_train.py
```

## Integration with Other Tools

Works with any command that respects `CUDA_VISIBLE_DEVICES`:

- **PyTorch** / **TensorFlow** training scripts
- **torchrun** for distributed training
- Any CUDA application
- **uv** / **conda** Python environments that run any of the above commands
- **just** / **make** build recipes that run any of the above commands

## Related Tools

**`idlegpu`** - Simple shell utility returning idle GPU ID. No multi-GPU, fallback, or wait support.

**`gpustat`** / **`nvitop`** - Monitoring tools with rich status displays. Monitoring only, no command execution.

**SLURM** / **Kubernetes** - Enterprise job schedulers. Feature-rich but heavyweight, complex setup.

### Why `with-gpu`?

Fills the gap between simple utilities and full schedulers:
- ✅ Executes commands (not just monitoring)
- ✅ Intelligent fallback (idle → least-used)
- ✅ Wait capability with timeout
- ✅ Multi-GPU min/max support
- ✅ Lightweight (single Rust binary)
- ✅ Direct NVML queries (reliable)

**Best for**: Individual workstations, small research groups, "just run this on an idle GPU" workflows.

## Limitations

- ❌ Multiple processes may select same GPU simultaneously
- ❌ GPU memory allocation delay creates race condition window
- ❌ Intermittent GPU usage may appear as idle
- ❌ No queue management or FIFO ordering
- ❌ No priority system for waiting processes
- ❌ No resource reservation or advance scheduling
- ❌ Not suitable for environments requiring fairness guarantees

**Mitigation**: Use `--require-idle`, `--wait`, or stagger launches. See [docs/limitations.md](docs/limitations.md) for detailed discussion.

**When you need more**: For guaranteed fair scheduling, priority queues, or resource reservations, use SLURM or Kubernetes.

Designed for **cooperative environments** (small groups, personal workstations) where "find me an idle GPU" is sufficient.

## Requirements

- NVIDIA GPU(s)
- NVIDIA driver with NVML library (libnvidia-ml.so)
- Linux or macOS (uses Unix process exec)
- Rust toolchain for building

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for comprehensive development documentation including:
- Architecture and design decisions
- Development workflow and code quality standards
- Testing procedures
- Style guidelines
- Troubleshooting common issues

## License

MIT

## Author

Oliver Steele <steele@osteele.com>

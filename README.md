# with-gpu

Intelligent GPU selection wrapper for CUDA commands. Automatically selects GPUs with the most available memory, then sets `CUDA_VISIBLE_DEVICES` and executes your command.

## Features

- 🧠 **Memory-first selection**: Prioritizes GPUs with most available VRAM (prevents OOM errors)
- 🎯 **Smart fallback**: Uses non-idle GPUs with free memory when no idle GPUs available
- 🖥️ **Multi-GPU support**: Request minimum and maximum number of GPUs
- 🎛️ **Manual selection**: Specify exact GPU IDs when needed
- ⏱️ **Wait capability**: Poll for GPU availability with configurable timeout
- 📊 **Status display**: View all GPUs and their current usage
- ⚠️ **Warning messages**: Get notified when using non-idle GPUs
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

This prioritizes available VRAM over idle status, preventing OOM errors.

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

Enforce idle-only selection (no non-idle GPUs even if they have more free memory):

```bash
# Single idle GPU required
with-gpu --require-idle python train.py

# Multiple idle GPUs required
with-gpu --min-gpus 2 --require-idle python train.py
```

**Note**: Without `--require-idle`, the tool selects GPUs by available memory regardless of idle status. Use this flag when you specifically need GPUs with 0 running processes.

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

In this example, auto-selection would pick GPU 1 (24 GB free), then GPU 2 (18 GB free), then GPU 0 (9 GB free).

## How It Works

1. **Queries GPUs**: Uses NVML library to get memory usage, utilization, and running processes for each GPU
2. **Selection Algorithm**:
   - **Primary criterion**: Most available memory (free VRAM in MB, descending)
   - **Secondary criterion**: Fewest running processes (ascending)
   - **Tertiary criterion**: Lowest GPU index (ascending)
3. **Special modes**:
   - `--require-idle`: Only considers GPUs with 0 processes (still sorted by available memory)
   - Manual `--gpu`: Bypasses auto-selection entirely
4. **Warnings**: Notifies when using non-idle GPUs (unless `--require-idle` prevents it)
5. **Execution**: Sets `CUDA_VISIBLE_DEVICES` and replaces current process with your command

**Why memory-first?** A GPU with 10 GB free and 1 process is more useful than an "idle" GPU with 300 MB free. This prevents OOM errors that occurred with the old idle-first algorithm.

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
- ✅ Memory-first selection (prevents OOM errors)
- ✅ Intelligent fallback (uses non-idle GPUs with free memory)
- ✅ Wait capability with timeout
- ✅ Multi-GPU min/max support
- ✅ Lightweight (single Rust binary)
- ✅ Direct NVML queries (reliable, not parsing nvidia-smi)
- ✅ Cross-platform (Linux + macOS)

**Best for**: Individual workstations, small research groups, "just run this on the GPU with most free memory" workflows.

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

**On Linux:**
- NVIDIA GPU(s)
- NVIDIA driver with NVML library (libnvidia-ml.so)
- Rust toolchain for building

**On macOS:**
- Rust toolchain for building
- Commands execute normally without GPU selection (cross-platform script compatibility)

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

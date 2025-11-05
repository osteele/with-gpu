# with-gpu

Intelligent GPU selection wrapper for CUDA commands. Automatically finds idle GPUs or allows manual selection, then sets `CUDA_VISIBLE_DEVICES` and executes your command.

## Features

- **Auto-select idle GPUs**: Prefers GPUs with no running processes
- **Fallback to least-used**: If no idle GPUs, selects GPU with least memory usage
- **Multi-GPU support**: Request minimum and maximum number of GPUs
- **Manual selection**: Specify exact GPU IDs when needed
- **Status display**: View all GPUs and their current usage
- **Warning messages**: Get notified when using non-idle GPUs

## Installation

```bash
cd ~/code/research/with-gpu
just install
```

This installs `with-gpu` to `~/.cargo/bin/with-gpu`.

## Usage

### Basic Usage (Auto-select)

Find one idle GPU (or least-used if none idle):

```bash
with-gpu python train.py
with-gpu just train-tc tiny
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

### Training on a GPU Server

```bash
# Auto-select one GPU (avoid the busy GPU 0)
ssh gpu-server
cd ~/code/research/linebreak-transformer
with-gpu just train-tc tiny

# Force use of GPU 1
with-gpu --gpu 1 just train-tc tiny

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

- PyTorch training scripts
- `just` recipes
- `torchrun` for distributed training
- TensorFlow scripts
- Any CUDA application

## Related Tools

### Simple Shell Utilities

**`idlegpu`** (used at some HPC clusters) - Returns the device number of an idle GPU
- **Pros**: Extremely simple, minimal overhead
- **Cons**: No multi-GPU support, no fallback logic, no waiting, requires manual scripting
- **Usage**: `` CUDA_VISIBLE_DEVICES=`idlegpu` python train.py ``

### Python Monitoring Tools

**`gpustat`** - Popular CLI tool for monitoring GPU status
- **Pros**: Beautiful output, widely used, integrates with watch/tmux
- **Cons**: Monitoring only, doesn't execute commands, requires Python
- **Usage**: `gpustat` (view status), `gpustat --watch` (continuous monitoring)

**`nvitop`** - Enhanced monitoring tool with process information
- **Pros**: Rich process details, interactive UI
- **Cons**: Monitoring only, no automatic selection or execution

### Job Schedulers

**SLURM**, **Kubernetes** - Full cluster job schedulers
- **Pros**: Enterprise-grade, multi-user, sophisticated policies, queue management
- **Cons**: Heavyweight setup, overkill for individual workstations, complex configuration
- **Usage**: `sbatch --gres=gpu:2 job.sh` (SLURM)

### What Makes `with-gpu` Different

`with-gpu` fills the gap between simple utilities and heavyweight schedulers:

1. **Automatic execution wrapper** - Monitors GPUs AND runs your command (not just monitoring)
2. **Intelligent fallback** - Uses idle GPUs first, falls back to least-used if needed
3. **Wait capability** - Polls for GPU availability with configurable timeout (new feature)
4. **Multi-GPU aware** - Handles min/max GPU requirements intelligently
5. **Lightweight** - Single Rust binary, no Python runtime, no cluster infrastructure
6. **Process replacement** - Uses `exec()` to replace wrapper (preserves stdio, signal handling)
7. **NVML direct** - Queries GPU driver library directly (more reliable than parsing nvidia-smi)

**Best for**: Individual GPU workstations, small research groups, interactive development where you want "just run this on an idle GPU" without setup overhead.

## Limitations

### Race Conditions

**Multiple processes selecting simultaneously**: If multiple `with-gpu` processes run at the same time, they may select the same GPU before either has started using it. The OS scheduler determines which waiting process runs next (no FIFO guarantees).

**GPU acquisition delay**: Programs may take time to allocate GPU memory after starting. During this window, another `with-gpu` process might see the GPU as idle and select it.

**Intermittent GPU usage**: Programs that release GPU memory between execution phases may appear idle when they're not. The tool can't distinguish between "done" and "between phases."

**Mitigation strategies**:
- Use `--require-idle` to be more conservative
- Use `--wait` to reduce simultaneous selection attempts
- Stagger experiment launches by a few seconds
- **Future enhancement**: Optional lockfile (`--lockfile`) to serialize GPU selection

### Fairness and Priority

`with-gpu` provides **no fairness guarantees**:
- No queue management or FIFO ordering
- No priority system
- Waiting processes compete via OS scheduler (effectively random)
- No resource reservation or advance scheduling

**When you need fairness**: If you need guaranteed fair scheduling, priority queues, or resource reservations, you've outgrown this tool and should use a proper workload manager like SLURM, which provides:
- Job queues with priority policies
- Resource reservations and backfill scheduling
- Fair-share scheduling across users
- Quality-of-service guarantees

`with-gpu` is designed for **cooperative** environments (small research groups, personal workstations) where lightweight "find me an idle GPU" is sufficient.

## Requirements

- NVIDIA GPU(s)
- NVIDIA driver with NVML library (libnvidia-ml.so)
- Linux or macOS (uses Unix process exec)
- Rust toolchain for building

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for comprehensive development documentation including:
- Architecture and design decisions
- Development workflow and code quality standards
- Testing procedures (local and on cool30)
- Style guidelines
- Troubleshooting common issues

Quick start:
```bash
cd ~/code/research/with-gpu
just all-checks  # Run all quality checks
just install     # Install to ~/.cargo/bin
```

## License

MIT

## Author

Oliver Steele <steele@osteele.com>

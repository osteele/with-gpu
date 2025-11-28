# Design Decisions

This document explains the architectural and design decisions behind `with-gpu`, particularly around threshold filtering and memory management.

## Design Philosophy

Key principles:
- **Safe defaults**: CLI tools should prevent common errors
- **Explicit opt-out**: Users can override (`--min-memory 0`)
- **Clear warnings**: Inform users when approaching danger zones
- **Fail fast**: Error early rather than let jobs OOM after hours of computation

## Threshold Filtering Design

### Why 2 GB Default Minimum Memory?

**Decision**: Default to requiring 2048 MB (2 GB) free memory when auto-selecting GPUs.

**Rationale**:
- **PyTorch initialization**: ~500 MB baseline memory usage before loading any models
- **Typical models**: Most research models need 1-10 GB for parameters and activations
- **Safety margin**: 2 GB catches 99% of OOM cases while being conservative
- **User override**: Can use `--min-memory 0` for small jobs that need less

### Why 500 MB Idle Detection Threshold?

**Decision**: A GPU is "idle" only if it has 0 processes AND <500 MB memory used.

**Rationale**:
- **Ghost process detection**: NVML's `running_compute_processes()` can miss processes
  - Happens with CUDA persistence mode, MPS, or certain driver states
  - Example: User's cool30 OOM error - GPU had 17 GB used but NVML reported 0 processes
- **Driver overhead**: CUDA driver + persistence daemon typically use 200-400 MB
- **Safety margin**: 500 MB allows driver overhead while catching ghost processes (>1 GB used)

**Why not 1 GB or 2 GB?**
- 1 GB would be safer but might incorrectly mark idle GPUs as "used" on some systems
- 2 GB would match the default minimum but conflates two different concepts:
  - Idle detection: "Is this GPU truly unused?"
  - Minimum memory: "Does this GPU have enough space for my job?"
- 500 MB strikes a balance: catches ghost processes while allowing normal driver overhead

### Threshold vs. Idle Detection

These are **separate concepts** serving different purposes:

| Aspect | Idle Detection (500 MB) | Minimum Memory (2 GB default) |
|--------|------------------------|-------------------------------|
| **Purpose** | Detect ghost processes | Prevent OOM errors |
| **When applied** | Only with `--require-idle` | Every auto-selection (unless overridden) |
| **Can disable?** | No (always active) | Yes (`--min-memory 0`) |
| **Rationale** | Reliability (NVML isn't perfect) | Usability (prevent common errors) |

## Warning Strategy

### Always Warn on Low Memory

**Decision**: Warn when selected GPU has <2 GB free, even if user explicitly allowed it.

**Rationale**:
- Users might forget the implications of `--min-memory 0`
- Warning doesn't block execution, just informs
- Better to over-communicate than let users hit OOM errors

**Example**:
```bash
$ with-gpu --min-memory 0 --gpu 3 python train.py
Warning: GPU 3 has only 0.43 GB free (< 2 GB recommended for PyTorch)
Selected GPU(s): 3
  GPU 3: USED - 17000/24268 MB (70.0%), 45 util, 1 processes
```

### Warning vs. Error

| Situation | Behavior | Rationale |
|-----------|----------|-----------|
| GPU has <2 GB free (default) | **Error** (filtered out) | Prevent OOM by default |
| GPU has <2 GB free (explicit `--min-memory 0`) | **Warning** | User chose this, just inform |
| Using non-idle GPUs | **Warning** | Common case, not necessarily bad |
| No GPUs meet criteria | **Error** | Can't proceed without a GPU |

## Future Extensions

### Potential Threshold Types

Considered for roadmap:
- `--max-temp`: Maximum temperature in Celsius
- `--max-power`: Maximum power draw in Watts

**Why not implemented now?**:
- Less common use case than memory/utilization
- Temperature monitoring often handled by data center infrastructure
- Power limits usually set at system level, not per-job

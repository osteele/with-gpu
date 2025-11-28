# Roadmap

This document outlines planned features and potential future directions for `with-gpu`.

## Planned Features

### Temperature Filtering (`--max-temp`)

Filter GPUs by maximum temperature for thermal management.

```bash
# Only use GPUs below 80°C
with-gpu --max-temp 80 python train.py

# Combine with other thresholds
with-gpu --min-memory 8000 --max-temp 75 --max-util 70 python train.py
```

**Benefits**:
- Avoid thermal throttling in long-running jobs
- Prevent using GPUs that might shut down due to overheating

**Trade-offs**:
- Temperature fluctuates rapidly - may cause flapping
- Most systems have thermal management at hardware level
- Less critical than memory/utilization for most users

---

### Power Filtering (`--max-power`)

Filter GPUs by maximum power draw for power-constrained environments.

```bash
# Only use GPUs drawing less than 200W
with-gpu --max-power 200 python train.py

# Power-efficient training
with-gpu --max-power 150 --max-util 70 python train.py
```

**Benefits**:
- Useful in power-limited environments (laptops, edge devices)
- Can help avoid power budget issues in multi-GPU systems
- Enables power-aware scheduling

**Trade-offs**:
- Power draw varies significantly during computation
- Most systems manage power budgets at infrastructure level
- Less predictable than static thresholds like memory

---

## Potential Future Enhancements

### 1. GPU Affinity/NUMA Awareness

**Idea**: Prefer GPUs on the same NUMA node as the CPU cores.

**Use case**: Optimize memory bandwidth in multi-socket systems.

**Implementation**: Query PCIe topology via NVML, match with CPU affinity.

**Verdict**: Low priority - most research workstations have simple topologies. Could add if there's user demand.

---

### 2. Configuration File Support

**Idea**: Allow users to set defaults in `~/.config/with-gpu/config.toml`.

**Example**:
```toml
[defaults]
min_memory = 8000  # 8 GB default
max_util = 80      # Max 80% utilization

[profiles]
[profiles.llm]
min_memory = 40000
max_util = 50

[profiles.inference]
min_memory = 4000
max_util = 90
```

**Usage**:
```bash
with-gpu --profile llm python train_llm.py
```

**Verdict**: Maybe - adds value for power users, but increases complexity. Consider if there's strong user demand.

---

### 3. JSON/Machine-Readable Output

**Idea**: Add `--json` flag for machine-parseable output.

**Use case**: Integration with other tools, scripting.

**Example**:
```bash
$ with-gpu --status --json
{
  "gpus": [
    {"index": 0, "memory_free_mb": 15000, "memory_total_mb": 24268, ...},
    {"index": 1, "memory_free_mb": 24000, "memory_total_mb": 24268, ...}
  ]
}
```

**Verdict**: Low priority but straightforward to add. Could be useful for integration.

---

## Out of Scope

These features are explicitly **not** planned:

### GPU Memory Reservation
- **Why**: Would require persistence between tool invocation and job start (lock files, shared memory, or coordinator service)
- **Alternative**: Users with race condition concerns should use SLURM or similar

### Multi-Node Support
- **Why**: Would require network communication and coordination
- **Alternative**: Use SLURM/Kubernetes for distributed training across multiple nodes

### Historical Usage Tracking
- **Why**: Would require persistent state storage and background daemon
- **Alternative**: Users needing fairness should use SLURM

### Remote GPU Selection
- **Why**: Adds network protocol, authentication, complexity
- **Alternative**: SSH into remote machine and run `with-gpu` there

### Queue Management
- **Why**: Requires persistent state, fairness algorithms, priority handling
- **Alternative**: Use SLURM, Kubernetes, or other job schedulers

### GPU Partitioning (MIG support)
- **Why**: NVIDIA MIG is enterprise feature, limited hardware support
- **Alternative**: Use nvidia-smi or data center tools for MIG management

### Windows Support
- **Why**: Different NVML behavior, less common for research workloads
- **Alternative**: Use WSL2 with Linux `with-gpu`

---

## Contribution Guidelines

If you'd like to implement any roadmap feature:

1. **Open an issue** to discuss the approach before coding
2. **Check DESIGN.md** for architectural principles
3. **Follow existing patterns** (CLI parsing, threshold filtering, etc.)
4. **Add tests** where possible (at minimum, manual test plan)
5. **Update documentation** (README, CHANGELOG, help text)

See DEVELOPMENT.md for code style and development workflow.

---

## Versioning Strategy

- **Patch (0.x.Y)**: Bug fixes, documentation updates
- **Minor (0.X.0)**: New features, non-breaking changes
- **Major (X.0.0)**: Breaking changes (when 1.0 is reached)

Currently in 0.x phase - breaking changes are acceptable with proper documentation and migration path.

---

## Decision Log

| Feature | Decision | Rationale | Date |
|---------|----------|-----------|------|
| Temperature filtering | Roadmap | Lower priority than memory/util | 2025-11 |
| Power filtering | Roadmap | Lower priority than memory/util | 2025-11 |
| Memory reservation | Won't implement | Too complex for single-node tool | 2025-11 |
| Multi-node support | Won't implement | Use SLURM instead | 2025-11 |
| Config file support | Maybe | Useful but adds complexity | 2025-11 |
| JSON output | Low priority | Easy to add if demanded | 2025-11 |

---

## Community Input

Have ideas for `with-gpu`? Open an issue at:
https://github.com/osteele/with-gpu/issues

Before requesting a feature:
1. Check if it's in "Non-Goals" (won't be implemented)
2. Check if it's already on the roadmap
3. Explain your use case and why existing tools don't solve it

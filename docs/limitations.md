# Limitations

This document describes the limitations of `with-gpu` in detail. For a quick overview, see the [Limitations section in README.md](../README.md#limitations).

## Coordination Boundaries

### Cooperative Claims

Concurrent `with-gpu` processes coordinate through advisory locks in
`/tmp/with-gpu`. A claim is acquired before the command starts, inherited across
process replacement, and released by the operating system when the command exits.
This prevents two cooperating commands from claiming the same GPU, including
during application startup before CUDA memory is allocated.

Claims are local to one host and its shared `/tmp` namespace. They do not provide
distributed coordination across machines or containers with isolated temporary
filesystems.

### Programs Outside `with-gpu`

Programs launched directly do not acquire a cooperative claim. They can begin
using a GPU after `with-gpu` checks its status or while another command holds a
claim. The lock cannot reserve CUDA memory or prevent direct device access.

For coordination to work, launch all cooperating workloads through `with-gpu`.

### Intermittent External GPU Usage

Programs outside `with-gpu` that release GPU memory between execution phases may
appear idle when they are not. The tool cannot distinguish between "done" and
"between phases" without a cooperative claim.

**Examples:**
- Data preprocessing on CPU, then GPU training in batches
- Evaluation phases that temporarily release memory
- Checkpointing or logging that clears GPU cache
- Interactive notebooks with cell-by-cell execution

**What happens:** `with-gpu` sees 0 processes or low memory usage and assumes the GPU is available, but the program may resume GPU usage at any moment.

## Mitigation Strategies

### Use `--require-idle`

More conservative selection that only uses GPUs with 0 running processes:

```bash
with-gpu --require-idle python train.py
```

**Pros:**
- Avoids GPUs that might be in "between phases"
- Safer for long-running jobs

**Cons:**
- May wait longer or fail when GPUs have any activity
- Won't utilize GPUs with minimal background processes

### Use `--wait`

Wait for a cooperative claim or suitable GPU state instead of failing immediately:

```bash
with-gpu --wait python train.py
```

If a suitable GPU is busy or claimed, `--wait` retries until one becomes
available. This does not establish FIFO ordering between waiters.

## Fairness and Priority

### No Fairness Guarantees

`with-gpu` provides **no fairness guarantees**:

- **No queue management**: Processes don't wait in an ordered queue
- **No FIFO ordering**: First process to request doesn't necessarily get GPU first
- **No priority system**: All processes are treated equally, regardless of importance or wait time
- **Random selection**: When multiple processes wait, OS scheduler determines who runs next (effectively random from user perspective)
- **No resource reservation**: Cannot reserve GPUs for future use or specific users

### Why No Fairness?

**Design philosophy:** `with-gpu` is designed for **cooperative environments** where:
- Small research groups (2-10 people)
- Personal workstations with multiple GPUs
- Trust-based sharing among collaborators
- Lightweight "find me an idle GPU" workflows

**Keeps it simple:**
- No daemon or background service
- No queue or scheduling database
- No authentication or user tracking
- Single binary, instant startup

### When Fairness Matters

If you need guaranteed fair scheduling, you've outgrown this tool and should use a proper workload manager.

**Use SLURM when you need:**
- Job queues with priority policies
- Resource reservations and advance scheduling
- Backfill scheduling (running small jobs while large jobs wait)
- Fair-share scheduling across users
- Quality-of-service guarantees
- Multi-node GPU clusters
- Historical usage tracking
- Preemption and job checkpointing

**Use Kubernetes when you need:**
- Container orchestration with GPU resources
- Auto-scaling based on demand
- Resource quotas per namespace/user
- Pod priority and preemption
- Complex scheduling policies
- Integration with cloud providers

### Example Unfairness Scenario

**Scenario:** User A and User B both want GPU 0, which is busy.

User A runs: `with-gpu --wait python long_train.py` (will run for 24 hours)

User B runs (1 second later): `with-gpu --wait python quick_test.py` (will run for 5 minutes)

**What happens:**
- Both processes poll every 5 seconds
- GPU 0 becomes available
- OS scheduler randomly picks either A or B to run next
- If B wins, A waits another 5 minutes (fine)
- If A wins, B waits 24 hours (unfair, but that's the design)

**Workarounds:**
- Communication: "Hey, I'm running a quick test, can you wait?"
- Staggered launches: Start long jobs at night, short jobs during day
- Manual coordination: Slack channel with "GPU 0 reserved 3-5pm"

## Design Constraints

### Why Not Add Fairness?

Adding fairness would require:

1. **Persistent daemon** - Background process to manage queue
2. **Shared state** - Database or file to track waiting processes
3. **User authentication** - Know who requested what when
4. **Complex logic** - Priority calculation, aging, backfill
5. **Configuration** - Policy files, user quotas, etc.

This transforms a simple wrapper into a full workload manager. At that point, just use SLURM.

### Design Goal

`with-gpu` optimizes for:
- Zero configuration
- Instant startup (no daemon)
- No admin privileges needed
- No persistent scheduler state; lock files are only local coordination points
- Works on any machine with NVML

**Trade-off:** Simplicity over fairness. If you need fairness, use proper tools.

## When to Use `with-gpu`

**Good fit:**
- Personal workstation with 2-8 GPUs
- Small lab with 1-2 shared GPU servers
- Cooperative environment (friends, trusted colleagues)
- Interactive development and experimentation
- "Just run this on an idle GPU" workflows
- Temporary or informal setups

**Bad fit:**
- Large lab with 10+ users competing for GPUs
- Users who don't trust each other
- Need to enforce quotas or priorities
- Long-running batch jobs that must run eventually
- Production ML workflows
- GPU clusters with dozens of machines

## Summary

`with-gpu` makes **trade-offs for simplicity**:

- ❌ No enforcement for programs outside `with-gpu` → ✅ No daemon or privileges
- ❌ No fairness guarantees → ✅ No user tracking, zero config
- ❌ No CUDA resource enforcement → ✅ Lightweight advisory coordination

These limitations are **by design** to keep the tool lightweight and simple. If you need more sophisticated resource management, use SLURM or Kubernetes.

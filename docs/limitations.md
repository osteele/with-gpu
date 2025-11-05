# Limitations

This document describes the limitations of `with-gpu` in detail. For a quick overview, see the [Limitations section in README.md](../README.md#limitations).

## Race Conditions

### Multiple Processes Selecting Simultaneously

If multiple `with-gpu` processes run at the same time, they may select the same GPU before either has started using it. The OS scheduler determines which waiting process runs next, with no FIFO guarantees.

**What happens:**
1. Process A queries GPUs and finds GPU 1 idle
2. Process B queries GPUs (before A has started) and also finds GPU 1 idle
3. Both processes select GPU 1
4. Both commands start, potentially causing out-of-memory errors

**Why it happens:** There's a time gap between querying GPU status and actually allocating memory. During this window, the GPU appears idle to other processes.

### GPU Acquisition Delay

Programs may take time to allocate GPU memory after starting. During this window, another `with-gpu` process might see the GPU as idle and select it.

**What happens:**
1. Process A starts training script on GPU 1
2. Training script imports libraries, initializes (1-5 seconds)
3. Process B checks GPUs, sees GPU 1 has 0 processes, selects it
4. Process A finally allocates 20GB of GPU memory
5. Process B tries to allocate memory, gets out-of-memory error

**Typical delay times:**
- PyTorch import + CUDA initialization: 2-3 seconds
- Loading large models: 5-10 seconds
- Multi-GPU initialization: 10-15 seconds

### Intermittent GPU Usage

Programs that release GPU memory between execution phases may appear idle when they're not. The tool can't distinguish between "done" and "between phases."

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

Reduces the chance of simultaneous selection by spreading out attempts:

```bash
with-gpu --wait python train.py
```

**How it helps:**
- If no GPU available, waits and retries
- Other processes start using their selected GPUs during the wait
- Next check sees those GPUs as occupied

**Limitation:** Doesn't eliminate race conditions, just reduces frequency.

### Stagger Experiment Launches

Manually delay between launching multiple experiments:

```bash
with-gpu python experiment1.py &
sleep 10  # Give experiment1 time to allocate GPU
with-gpu python experiment2.py &
sleep 10
with-gpu python experiment3.py &
```

**Best practice:** 5-10 second gaps are usually sufficient for GPU allocation to complete.

### Future Enhancement: Lockfile

**Not yet implemented:** Optional `--lockfile` flag to serialize GPU selection:

```bash
with-gpu --lockfile /tmp/gpu-lock python train.py
```

Would use file-based locking to ensure only one process selects GPUs at a time. See [Possible Future Enhancements](../DEVELOPMENT.md#possible-future-enhancements).

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
- No shared state between processes
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
- No persistent state
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

- ❌ No race condition prevention → ✅ No daemon, instant startup
- ❌ No fairness guarantees → ✅ No user tracking, zero config
- ❌ No resource reservation → ✅ No persistent state, stateless

These limitations are **by design** to keep the tool lightweight and simple. If you need more sophisticated resource management, use SLURM or Kubernetes.

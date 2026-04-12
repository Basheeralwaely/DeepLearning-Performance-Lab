# Phase 2: Profiling & Diagnostics - Research

**Researched:** 2026-04-12
**Domain:** PyTorch profiling, GPU memory analysis, DataLoader performance, torch.compile diagnostics
**Confidence:** HIGH

## Summary

Phase 2 creates four standalone Python tutorials that teach ML engineers how to identify and measure performance bottlenecks using PyTorch's built-in profiling and diagnostic tools. All four tutorials use PyTorch 2.10.0 with CUDA 12.8, which is already installed on the target system (RTX 2070, 7.6 GB VRAM, 12 CPU cores). No additional libraries are needed -- every API required (torch.profiler, torch.cuda memory APIs, torch.utils.checkpoint, torch.compile, torch._dynamo.explain) is part of the PyTorch standard library and has been verified as available.

The tutorials follow established Phase 1 patterns: standalone .py files in `profiling/`, using SimpleCNN/SimpleMLP from `utils/models.py`, the `@benchmark` decorator for timing, and `print_benchmark_table` for comparison output. Each tutorial focuses purely on DIAGNOSTICS (measuring and identifying bottlenecks), not on applying optimizations.

**Primary recommendation:** Implement four tutorials using only PyTorch built-in APIs. No external dependencies needed. Focus on producing clear, measurable output (console tables + trace files) that teaches users to READ profiling data, not just generate it.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Use existing SimpleCNN/SimpleMLP from `utils/models.py` across all profiling tutorials
- **D-02:** For memory profiling (PROF-02), scale up batch size and input resolution with SimpleCNN to trigger real memory pressure and OOM conditions. Do not use artificial memory fraction limits.
- **D-03:** DataLoader tuning tutorial (PROF-03) uses synthetic data with fake I/O delay in `__getitem__` -- no dataset downloads needed
- **D-04:** Sweep 4 num_workers configurations
- **D-05:** Gradient checkpointing in PROF-02 is a profiling demo only -- show memory before/after, do NOT deep-dive into the checkpointing API
- **D-06:** torch.compile in PROF-04 focuses on diagnostics: mode comparisons, graph breaks, compilation overhead. Production deployment belongs in Phase 5.
- **D-07:** PyTorch Profiler tutorial (PROF-01) produces both console table output (key_averages) AND a Chrome trace JSON file
- **D-08:** Trace files saved to `./profiler_output/` directory in repo root. Add to `.gitignore`.

### Claude's Discretion
- Specific num_workers values for the DataLoader sweep (4 values that show clear throughput differences)
- How to structure the fake I/O delay in the synthetic dataset (sleep duration, transform complexity)
- Profiler configuration details (schedule, activities, record_shapes, etc.)
- Tutorial file naming within `profiling/` folder (following D-02 from Phase 1: topic-based, no numbering)

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PROF-01 | PyTorch Profiler with trace export and bottleneck analysis | `torch.profiler.profile` with `ProfilerActivity.CPU/CUDA`, `key_averages()` for console table, `export_chrome_trace()` for JSON trace. All verified available. |
| PROF-02 | GPU memory profiling, OOM debugging, gradient checkpointing | `torch.cuda.memory_summary()`, `max_memory_allocated()`, `memory_reserved()`, `reset_peak_memory_stats()`. `torch.utils.checkpoint.checkpoint()` for checkpointing demo. All verified. |
| PROF-03 | DataLoader tuning (prefetch, num_workers, pinned memory) | `torch.utils.data.DataLoader` with `num_workers`, `pin_memory`, `prefetch_factor` params. Synthetic dataset with `time.sleep()` in `__getitem__`. |
| PROF-04 | torch.compile modes, graph breaks, speedup measurement | `torch.compile(mode=...)` with "default", "reduce-overhead", "max-autotune". `torch._dynamo.explain()` for graph break detection. All verified. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.10.0+cu128 | All profiling APIs | Built-in profiler, memory APIs, torch.compile, dynamo [VERIFIED: local install] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.profiler | (built-in) | CPU/GPU trace profiling | PROF-01: trace export + key_averages |
| torch.cuda | (built-in) | Memory profiling APIs | PROF-02: memory_summary, max_memory_allocated |
| torch.utils.checkpoint | (built-in) | Gradient checkpointing | PROF-02: memory impact demo |
| torch.utils.data | (built-in) | DataLoader + Dataset | PROF-03: num_workers, pin_memory, prefetch |
| torch._dynamo | (built-in) | Graph break detection | PROF-04: explain() for graph breaks |
| time | (stdlib) | Fake I/O delay | PROF-03: sleep() in synthetic dataset |

### Alternatives Considered
None -- all required functionality is in PyTorch standard library. No external packages needed.

**Installation:**
```bash
# No additional packages needed. PyTorch 2.10.0+cu128 already installed.
```

## Architecture Patterns

### Recommended Project Structure
```
profiling/
    reference_tutorial.py          # existing Phase 1 output
    pytorch_profiler.py            # PROF-01
    memory_profiling.py            # PROF-02
    dataloader_tuning.py           # PROF-03
    torch_compile_diagnostics.py   # PROF-04
profiler_output/                   # D-08: trace artifacts (gitignored)
.gitignore                         # NEW: must be created
```

### Pattern 1: Tutorial File Structure (from reference_tutorial.py)
**What:** Every tutorial follows the same skeleton established in Phase 1.
**When to use:** All four tutorials.
**Example:**
```python
# Source: profiling/reference_tutorial.py [VERIFIED: codebase]
"""
Tutorial Title
==============
[Multi-line explanation of the technique]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import setup_logging, benchmark, print_benchmark_table, SimpleCNN, get_sample_batch, get_device, print_device_info

logger = setup_logging("tutorial_name")

# Constants at module level
BATCH_SIZE = 64
NUM_ITERATIONS = 100

def main():
    print("\n" + "=" * 60)
    print("  Tutorial Title")
    print("=" * 60 + "\n")

    device = get_device()
    print_device_info()

    # Section headers with print(), details with logger.info()
    print("\n--- Section 1: Title ---\n")
    logger.info("Explanation of what's happening...")

    # Benchmark table at end
    print_benchmark_table(results)

if __name__ == "__main__":
    main()
```

### Pattern 2: Profiler with Schedule and Chrome Trace Export (PROF-01)
**What:** Use torch.profiler.profile with a schedule for warm-up/active recording, export both console table and trace file.
**When to use:** PROF-01 tutorial.
**Example:**
```python
# Source: torch.profiler API [VERIFIED: local Python introspection]
from torch.profiler import profile, ProfilerActivity, schedule, record_function

os.makedirs("profiler_output", exist_ok=True)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=lambda p: p.export_chrome_trace("profiler_output/trace.json"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(5):  # wait(1) + warmup(1) + active(3) = 5
        with record_function("training_step"):
            output = model(x)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        prof.step()

# Console output
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
```

### Pattern 3: Memory Profiling with OOM Demo (PROF-02)
**What:** Show GPU memory breakdown at each stage, demonstrate OOM recovery, compare with/without gradient checkpointing.
**When to use:** PROF-02 tutorial.
**Example:**
```python
# Source: torch.cuda memory APIs [VERIFIED: local Python introspection]
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

# Log memory at each stage
def log_memory(stage_name):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    logger.info(f"{stage_name}: Allocated={allocated:.1f}MB, Reserved={reserved:.1f}MB, Peak={peak:.1f}MB")

# OOM recovery pattern
try:
    # Attempt with large batch
    x = torch.randn(large_batch, 3, 224, 224, device=device)
    output = model(x)
except torch.cuda.OutOfMemoryError:
    logger.warning("OOM caught! Clearing cache and reducing batch size...")
    torch.cuda.empty_cache()

# Gradient checkpointing comparison
from torch.utils.checkpoint import checkpoint
# use_reentrant=False is the modern recommended approach
output = checkpoint(model.features, x, use_reentrant=False)
```

### Pattern 4: Synthetic Dataset with Controllable I/O Delay (PROF-03)
**What:** Custom Dataset class with configurable sleep in __getitem__ to simulate I/O-bound workloads.
**When to use:** PROF-03 tutorial.
**Example:**
```python
# Source: standard PyTorch pattern [ASSUMED]
import time
from torch.utils.data import Dataset, DataLoader

class SyntheticIODataset(Dataset):
    def __init__(self, size=1000, io_delay=0.01):
        self.size = size
        self.io_delay = io_delay

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        time.sleep(self.io_delay)  # Simulate disk I/O
        image = torch.randn(3, 32, 32)
        label = torch.randint(0, 10, (1,)).item()
        return image, label

# Sweep configurations
for num_workers in [0, 2, 4, 8]:
    loader = DataLoader(dataset, batch_size=64, num_workers=num_workers,
                       pin_memory=True, prefetch_factor=2 if num_workers > 0 else None)
```

### Pattern 5: torch.compile Mode Comparison + Graph Break Detection (PROF-04)
**What:** Compare compile modes side-by-side, use dynamo.explain for graph break analysis.
**When to use:** PROF-04 tutorial.
**Example:**
```python
# Source: torch.compile + torch._dynamo [VERIFIED: local Python introspection]
import torch._dynamo as dynamo

# Mode comparison
modes = ["default", "reduce-overhead", "max-autotune"]
results = []
for mode in modes:
    dynamo.reset()  # Clear compilation cache between modes
    compiled_model = torch.compile(model, mode=mode)
    # Warm-up compilation
    _ = compiled_model(x)
    # Timed run
    # ... benchmark ...

# Graph break detection
explanation = dynamo.explain(model)(x)
logger.info(f"Graph count: {explanation.graph_count}")
logger.info(f"Graph break count: {explanation.graph_break_count}")
for reason in explanation.break_reasons:
    logger.info(f"Break reason: {reason.reason[:200]}")
```

### Anti-Patterns to Avoid
- **Using torch.no_grad() instead of inference_mode() for pure inference benchmarks:** inference_mode() is strictly faster and the reference tutorial already establishes this pattern.
- **Forgetting torch.cuda.synchronize() before timing:** GPU ops are async; without sync, timing is meaningless. The `@benchmark` decorator handles this, but any manual timing in profiling tutorials must also synchronize.
- **Forgetting dynamo.reset() between compile mode comparisons:** Previous compilation state leaks. Must reset between modes.
- **Setting prefetch_factor with num_workers=0:** Raises ValueError in PyTorch. Must be None or omitted when num_workers=0.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Execution timing | Custom timer loops | `@benchmark` decorator from utils | Already handles CUDA sync, peak memory, returns structured dict |
| Comparison tables | Custom string formatting | `print_benchmark_table` from utils | Consistent format, handles memory column |
| Device detection | Manual torch.cuda checks | `get_device` / `print_device_info` from utils | Consistent with other tutorials |
| Profiler trace export | Manual JSON writing | `prof.export_chrome_trace()` | PyTorch built-in, compatible with chrome://tracing |
| Memory breakdown | Manual allocation tracking | `torch.cuda.memory_summary()` | Comprehensive built-in table |

**Key insight:** Phase 1 utilities already handle timing, memory tracking, and comparison tables. Phase 2 tutorials should use these for their own benchmarks, and ADDITIONALLY demonstrate PyTorch's dedicated profiling APIs as the tutorial subject matter.

## Common Pitfalls

### Pitfall 1: OOM in Memory Profiling Tutorial Crashes the Whole Script
**What goes wrong:** Intentionally triggering OOM to demonstrate debugging kills the process before showing the recovery strategy.
**Why it happens:** Unhandled OOM exceptions, or allocating so much that CUDA context itself is destroyed.
**How to avoid:** Wrap OOM-triggering code in try/except `torch.cuda.OutOfMemoryError`. Start with a moderately large batch and incrementally increase. Call `torch.cuda.empty_cache()` in the except block. On a 7.6 GB RTX 2070, a batch of ~256 at 224x224 with SimpleCNN should be enough to trigger OOM during backward pass.
**Warning signs:** Script exits with CUDA OOM error instead of catching it gracefully.

### Pitfall 2: torch.compile First-Run Compilation Overhead Dominates Timing
**What goes wrong:** The first call to a compiled model includes compilation time (seconds), making it look slower than eager mode.
**Why it happens:** torch.compile is lazy -- compilation happens on first forward pass.
**How to avoid:** Always run a warm-up pass after compilation before timing. Measure compilation time separately and report it explicitly (it's educational -- users need to understand this cost).
**Warning signs:** Compiled model appears slower than eager mode.

### Pitfall 3: DataLoader Worker Count Exceeding CPU Cores
**What goes wrong:** Setting num_workers > CPU cores causes worker contention, reducing throughput.
**Why it happens:** Each worker is a separate process. More workers than cores means context switching overhead.
**How to avoid:** Target system has 12 CPU cores. Recommended sweep: [0, 2, 4, 8]. All values are within core count. Log the CPU count in the tutorial output.
**Warning signs:** Throughput decreases at higher worker counts.

### Pitfall 4: Profiler Schedule Step Count Mismatch
**What goes wrong:** The training loop iteration count doesn't match what the profiler schedule expects (wait + warmup + active).
**Why it happens:** Off-by-one errors or not understanding the schedule phases.
**How to avoid:** Compute total steps explicitly: `total_steps = (wait + warmup + active) * repeat` (or `wait + warmup + active` if repeat=0 means no repeat). Ensure the loop runs at least this many iterations, and call `prof.step()` at the end of each iteration.
**Warning signs:** Empty or partial trace files.

### Pitfall 5: Gradient Checkpointing use_reentrant Deprecation Warning
**What goes wrong:** Calling `checkpoint()` without `use_reentrant` parameter produces a deprecation warning that clutters output.
**Why it happens:** PyTorch is migrating to `use_reentrant=False` as default. Current default is True but deprecated.
**How to avoid:** Always pass `use_reentrant=False` explicitly. This is the modern recommended approach.
**Warning signs:** FutureWarning in console output.

## Code Examples

### PROF-01: Profiler Key Averages Table Output
```python
# Source: torch.profiler API [VERIFIED: local introspection]
# Sort options for key_averages().table():
#   "cpu_time_total", "cuda_time_total", "self_cpu_time_total",
#   "self_cuda_time_total", "cpu_memory_usage", "self_cpu_memory_usage",
#   "cuda_memory_usage", "self_cuda_memory_usage"

# Show top operations by GPU time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

# Show memory-heavy operations
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

# Group by input shapes (useful for identifying which layer configs are expensive)
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))
```

### PROF-02: Memory Tracking Helper
```python
# Source: torch.cuda memory APIs [VERIFIED: local introspection]
def log_memory_state(label):
    """Log current GPU memory state with a descriptive label."""
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    logger.info(f"[{label}] Allocated: {alloc:.1f} MB | Reserved: {reserved:.1f} MB | Peak: {peak:.1f} MB")

# Full summary table
print(torch.cuda.memory_summary(abbreviated=True))
```

### PROF-03: DataLoader Throughput Measurement
```python
# Source: standard PyTorch DataLoader pattern [ASSUMED]
import time

def measure_throughput(loader, num_batches=50):
    """Measure batches/sec for a DataLoader configuration."""
    start = time.perf_counter()
    for i, (data, target) in enumerate(loader):
        if i >= num_batches:
            break
        data = data.to(device, non_blocking=True)  # non_blocking with pin_memory
        target = target.to(device, non_blocking=True)
    elapsed = time.perf_counter() - start
    return num_batches / elapsed  # batches per second
```

### PROF-04: Graph Break Detection with Explanation
```python
# Source: torch._dynamo.explain [VERIFIED: local introspection]
import torch._dynamo as dynamo

dynamo.reset()
explanation = dynamo.explain(model)(sample_input)

logger.info(f"Total graph count: {explanation.graph_count}")
logger.info(f"Graph break count: {explanation.graph_break_count}")
logger.info(f"Ops per graph: {explanation.ops_per_graph}")

if explanation.break_reasons:
    for i, reason in enumerate(explanation.break_reasons):
        logger.warning(f"Graph break #{i+1}: {reason.reason.split(chr(10))[0]}")
else:
    logger.info("No graph breaks detected -- model is fully traceable")
```

## Discretion Recommendations

### num_workers Values (D-04)
**Recommendation:** `[0, 2, 4, 8]`
**Rationale:** 0 shows the baseline (main process only, maximum I/O bottleneck). 2 shows initial parallelism benefit. 4 is a common sweet spot. 8 shows diminishing returns. All are within the 12-core CPU limit. [ASSUMED -- based on general best practice]

### Fake I/O Delay Structure (D-03)
**Recommendation:** `time.sleep(0.01)` (10ms) per item in `__getitem__`
**Rationale:** 10ms simulates realistic disk I/O latency (HDD random read). With batch_size=64 and num_workers=0, this creates a clear ~640ms bottleneck per batch that workers dramatically reduce. Too short (1ms) and the difference is hard to see; too long (100ms) and the tutorial takes forever. [ASSUMED]

### Profiler Configuration (PROF-01)
**Recommendation:** `schedule(wait=1, warmup=1, active=3, repeat=1)` with `record_shapes=True, profile_memory=True, with_stack=True`
**Rationale:** wait=1 lets CUDA warm up, warmup=1 discards the first active step, active=3 captures enough data for statistics. record_shapes and profile_memory give users the richest possible trace. [ASSUMED]

### Tutorial File Naming
**Recommendation:** `pytorch_profiler.py`, `memory_profiling.py`, `dataloader_tuning.py`, `torch_compile_diagnostics.py`
**Rationale:** Follows D-02 from Phase 1 (topic-based, no numbering). Names are descriptive and match the technique taught. [ASSUMED]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.autograd.profiler` | `torch.profiler` (new API) | PyTorch 1.8+ | New API supports schedule, Chrome trace, TensorBoard integration |
| `use_reentrant=True` for checkpointing | `use_reentrant=False` | PyTorch 2.x | Old default is deprecated, new default is safer and more compatible |
| `torch.jit.trace` / `torch.jit.script` | `torch.compile` | PyTorch 2.0+ | torch.compile is the modern compilation path; TorchScript is maintenance mode |
| Manual CUDA memory tracking | `torch.cuda.memory_summary()` | PyTorch 1.4+ | Built-in comprehensive memory reporting |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | num_workers=[0,2,4,8] produces clear throughput differences | Discretion Recommendations | Low -- easy to adjust values |
| A2 | 10ms sleep simulates realistic I/O delay for clear demo | Discretion Recommendations | Low -- can tune if output is unclear |
| A3 | Batch 256 at 224x224 triggers OOM on 7.6GB RTX 2070 during backward | Pitfall 1 | Medium -- need to test and calibrate exact batch size |
| A4 | profiler schedule(wait=1, warmup=1, active=3) is optimal for tutorial | Discretion Recommendations | Low -- standard recommendation |
| A5 | DataLoader throughput measurement pattern | Code Examples | Low -- standard pattern |

## Open Questions

1. **Exact OOM batch size for PROF-02**
   - What we know: RTX 2070 has 7.6 GB VRAM. SimpleCNN with input_size=224 will use more memory than default 32.
   - What's unclear: The exact batch size that triggers OOM during backward (activations + gradients). Depends on CUDA context overhead.
   - Recommendation: Start at batch_size=256 with 224x224 input; iterate up if no OOM. Include a binary search or incremental approach in the tutorial so it adapts to different GPUs.

2. **profiler_output directory and .gitignore creation**
   - What we know: D-08 requires `profiler_output/` with .gitignore. Currently no .gitignore exists in repo.
   - What's unclear: Should .gitignore cover only profiler_output or also __pycache__ and other standard ignores?
   - Recommendation: Create a comprehensive .gitignore (Python + profiler artifacts) as a Wave 0 / infrastructure task.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PyTorch | All tutorials | Yes | 2.10.0+cu128 | -- |
| CUDA | GPU profiling | Yes | 12.8 | CPU-only mode (reduced tutorial value) |
| NVIDIA GPU | All tutorials | Yes | RTX 2070 7.6GB | -- |
| torch.profiler | PROF-01 | Yes | built-in | -- |
| torch.compile | PROF-04 | Yes | built-in | -- |
| torch._dynamo | PROF-04 | Yes | built-in | -- |

**Missing dependencies with no fallback:** None.
**Missing dependencies with fallback:** None.

## Sources

### Primary (HIGH confidence)
- PyTorch 2.10.0 local installation -- verified all APIs via Python introspection: torch.profiler (profile, ProfilerActivity, schedule, record_function), torch.cuda memory APIs (memory_summary, memory_allocated, memory_reserved, max_memory_allocated, reset_peak_memory_stats, empty_cache), torch.utils.checkpoint, torch.compile modes, torch._dynamo.explain
- Existing codebase (utils/models.py, utils/benchmark.py, profiling/reference_tutorial.py) -- verified patterns and available utilities

### Secondary (MEDIUM confidence)
- torch.compile mode documentation extracted from docstring [VERIFIED: local introspection]
- torch._dynamo.explain ExplainOutput fields [VERIFIED: local test run]

### Tertiary (LOW confidence)
- None -- all critical claims verified locally

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all APIs verified locally on exact PyTorch version
- Architecture: HIGH -- follows established Phase 1 patterns, all APIs tested
- Pitfalls: MEDIUM -- based on verified API behavior + general PyTorch experience
- Discretion items: MEDIUM -- reasonable defaults but may need tuning on target hardware

**Research date:** 2026-04-12
**Valid until:** 2026-05-12 (stable -- PyTorch profiling APIs are mature)

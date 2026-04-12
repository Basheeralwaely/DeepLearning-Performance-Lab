# Phase 2: Profiling & Diagnostics - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-12
**Phase:** 02-profiling-diagnostics
**Areas discussed:** Model complexity for demos, DataLoader workload, Tutorial scope boundaries, Profiler output & artifacts

---

## Model Complexity for Demos

| Option | Description | Selected |
|--------|-------------|----------|
| Existing utils models (Recommended) | Use SimpleCNN/SimpleMLP from utils/ consistently. Keep profiling focus on the TOOL, not the model. Increase batch size or input size to create measurable bottlenecks. | ✓ |
| Torchvision pretrained | Use ResNet-50 or similar from torchvision. More realistic bottleneck profiles, but adds a dependency and shifts focus toward the model. | |
| Add a heavier model to utils | Create a new TransformerModel or deeper CNN in utils/models.py. Avoids external deps while showing real bottlenecks. | |

**User's choice:** Existing utils models
**Notes:** None

### OOM Demo Approach

| Option | Description | Selected |
|--------|-------------|----------|
| Scale up inputs (Recommended) | Use large batch sizes or high-res inputs with SimpleCNN to trigger real memory pressure. | ✓ |
| Simulate constraints | Use torch.cuda.set_per_process_memory_fraction() to artificially limit GPU memory. | |
| You decide | Claude picks the best approach based on what works reliably across different GPU sizes. | |

**User's choice:** Scale up inputs
**Notes:** None

---

## DataLoader Workload

| Option | Description | Selected |
|--------|-------------|----------|
| Synthetic with fake I/O (Recommended) | Generate tensors on-the-fly with an artificial sleep/disk-read delay in __getitem__. No downloads needed, I/O bottleneck is controllable and reproducible. | ✓ |
| CIFAR-10 download | Use torchvision.datasets.CIFAR10 with download=True. Real dataset, but small (163MB). | |
| Fake disk-based dataset | Generate image files to /tmp at startup, then load from disk. Most realistic I/O pattern but adds setup complexity. | |

**User's choice:** Synthetic with fake I/O
**Notes:** None

### Num Workers Sweep

| Option | Description | Selected |
|--------|-------------|----------|
| Auto range (Recommended) | Sweep 0, 1, 2, 4, and os.cpu_count() automatically. Adapts to the machine. | |
| Fixed set of 3 | Test just 0, 2, and 4 workers. Faster to run, simpler output. | |
| You decide | Claude picks the right sweep range. | |

**User's choice:** Other — "4" (sweep 4 configurations, Claude determines specific values)
**Notes:** User specified 4 as the number of configurations to sweep.

---

## Tutorial Scope Boundaries

### Gradient Checkpointing Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Profiling demo only (Recommended) | Show gradient checkpointing as one tool for reducing memory. Profile before/after. Don't deep-dive into API or custom policies. | ✓ |
| Full technique coverage | Cover checkpointing comprehensively: API, custom checkpoint functions, segment-level control. | |
| You decide | Claude scopes it based on what fits the profiling narrative. | |

**User's choice:** Profiling demo only
**Notes:** None

### torch.compile Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Diagnostic focus (Recommended) | Focus on MEASURING torch.compile: mode comparisons, detecting graph breaks, understanding compilation overhead. Leave production patterns to Phase 5. | ✓ |
| Comprehensive coverage | Cover torch.compile fully here. Phase 5 would skip it entirely. | |
| You decide | Claude determines the right split between Phase 2 and Phase 5. | |

**User's choice:** Diagnostic focus
**Notes:** None

---

## Profiler Output & Artifacts

### Output Format

| Option | Description | Selected |
|--------|-------------|----------|
| Console + Chrome trace (Recommended) | Print key_averages table to console AND export Chrome trace JSON. Immediate console insight plus visual trace for chrome://tracing. | ✓ |
| Console only | Only print profiler tables to console. Simple and standalone but misses visual power. | |
| Console + TensorBoard | Print tables AND export TensorBoard logs. Richer visualization but adds tensorboard dependency. | |

**User's choice:** Console + Chrome trace
**Notes:** None

### Trace File Location

| Option | Description | Selected |
|--------|-------------|----------|
| ./profiler_output/ (Recommended) | Create profiler_output/ in repo root, add to .gitignore. Clean and predictable. | ✓ |
| /tmp/profiler_output/ | System temp directory. No repo clutter but users need to find their traces. | |
| You decide | Claude picks the most practical location. | |

**User's choice:** ./profiler_output/
**Notes:** None

---

## Claude's Discretion

- Specific num_workers values for the DataLoader sweep
- Fake I/O delay structure in synthetic dataset
- Profiler configuration details (schedule, activities, record_shapes)
- Tutorial file naming within profiling/ folder

## Deferred Ideas

None — discussion stayed within phase scope

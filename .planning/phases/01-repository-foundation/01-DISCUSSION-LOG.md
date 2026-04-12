# Phase 1: Repository Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-12
**Phase:** 01-repository-foundation
**Areas discussed:** Folder structure, Tutorial format

---

## Folder structure

### Folder naming convention

| Option | Description | Selected |
|--------|-------------|----------|
| Short descriptive | distributed_training/, mixed_precision/, pruning/, quantization/, profiling/, inference/ | ✓ |
| Numbered + name | 01_profiling/, 02_mixed_precision/, etc. — suggests a learning order | |
| Flat kebab-case | distributed-training/, mixed-precision/, tensorrt/, pruning/, profiling/ | |

**User's choice:** Short descriptive (snake_case)
**Notes:** No implied learning order — users navigate by technique

### File naming within folders

| Option | Description | Selected |
|--------|-------------|----------|
| Numbered + topic | 01_ddp_basics.py, 02_fsdp.py, 03_model_parallel.py | |
| Just topic | ddp_basics.py, fsdp.py, model_parallel.py — no implied order | ✓ |
| Prefixed by type | train_ddp.py, train_fsdp.py, bench_throughput.py | |

**User's choice:** Just topic — no numbering
**Notes:** None

### Shared utilities folder

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, utils/ folder | Shared helpers importable by all tutorials | ✓ |
| No, self-contained | Each tutorial includes everything it needs | |
| You decide | Claude picks the best approach | |

**User's choice:** Yes, utils/ folder
**Notes:** Keeps tutorials focused on the technique, common code extracted

---

## Tutorial format

### Logging style

| Option | Description | Selected |
|--------|-------------|----------|
| Python logging | logging.getLogger() with configured formatters | |
| Rich print | print() with clear section headers and separators | |
| Both | logging module for technique output + print for section headers | ✓ |

**User's choice:** Both — logging for technique output, print for section headers/explanations
**Notes:** None

### Benchmark presentation

| Option | Description | Selected |
|--------|-------------|----------|
| Inline comparison | Print a formatted table at the end: Baseline vs Optimized with speedup/memory diff | ✓ |
| Running commentary | Log each measurement as it happens, then summarize at the end | |
| You decide | Claude picks the best benchmark presentation | |

**User's choice:** Inline comparison table at the end
**Notes:** None

### Code explanations

| Option | Description | Selected |
|--------|-------------|----------|
| Heavy comments | Inline comments explaining each block + module docstring | |
| Docstring + minimal | Rich module docstring, minimal inline comments for non-obvious parts | ✓ |
| Teaching style | Step-by-step print statements narrating + inline comments | |

**User's choice:** Docstring + minimal inline comments
**Notes:** None

---

## Claude's Discretion

- Shared utilities design (what goes in utils/)
- README structure and navigation
- Reference tutorial topic choice

## Deferred Ideas

None — discussion stayed within phase scope

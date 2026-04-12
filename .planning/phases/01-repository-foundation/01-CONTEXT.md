# Phase 1: Repository Foundation - Context

**Gathered:** 2026-04-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Scaffold the repository folder structure, create shared utility modules, and establish the tutorial conventions (file format, logging style, benchmark presentation) that all subsequent phases will follow. This phase produces the skeleton — no technique tutorials yet, but a reference example demonstrating the conventions.

</domain>

<decisions>
## Implementation Decisions

### Folder Structure
- **D-01:** Technique folders use short descriptive snake_case names: `distributed_training/`, `mixed_precision/`, `pruning/`, `profiling/`, `inference/`, `compression/`
- **D-02:** Tutorial .py files within each folder are named by topic only (no numbering): `ddp_basics.py`, `fsdp.py`, `model_parallel.py`
- **D-03:** A top-level `utils/` folder holds shared helpers importable by all tutorials (logging setup, benchmark utilities, simple model definitions, device detection)

### Tutorial Format
- **D-04:** Logging uses both `logging` module (for technique output with proper log levels) and `print()` (for section headers and visual explanations)
- **D-05:** Before/after benchmarks presented as a formatted comparison table printed at the end of each tutorial: Baseline vs Optimized with speedup and memory diff
- **D-06:** Code explanations use rich module-level docstrings explaining the technique, with minimal inline comments only for non-obvious code sections

### Claude's Discretion
- Shared utilities design: What specific helpers go in `utils/` (logging config, timer/benchmark decorator, model factories, device helpers) — Claude determines the right abstractions
- README structure and navigation approach — Claude determines how users discover tutorials
- Reference tutorial topic choice — Claude picks a simple example that demonstrates conventions without overlapping with Phase 2-6 content

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

No external specs — requirements fully captured in decisions above. The repo is greenfield with only LICENSE and README.md present.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- None — greenfield repository with only CLAUDE.md, LICENSE, README.md

### Established Patterns
- None — this phase establishes all patterns

### Integration Points
- Every subsequent phase (2-6) will import from `utils/` and follow the conventions set here
- Tutorial .py files must be standalone-runnable: `python tutorial_name.py`

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. Key constraint: tutorials target ML engineers who already know PyTorch, so explanations should be technique-focused, not framework-introductory.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-repository-foundation*
*Context gathered: 2026-04-12*

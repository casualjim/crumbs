# Scribe-like Covering Context for Crumbs

Status: Draft
Last updated: 2026-01-12
Owners: TBD

## Summary

Crumbs already builds a chunk+embedding index, a lightweight symbol/reference graph, and topology views, and it can assemble prompt-ready context for tasks/issues/topology. What we’re missing is a **fast, single-shot “covering context” workflow** that coding agents can call directly from the CLI to get *only* what’s needed to understand (or change) a specific target **function/class/file**.

This document proposes adding a **Scribe-like covering set subsystem** to Crumbs:

- A **Covering Set** engine that returns the minimal dependency closure for a target at either:
  - **File granularity** (fast, good defaults)
  - **Entity granularity** (functions/classes; surgical precision)
- **Budget-aware degradation** beyond line trimming: **FULL → CHUNK → SIGNATURE**
- Optional **transformer-aware positioning** (HEAD/MIDDLE/TAIL) for better LLM recall
- New CLI helper commands so agents can obtain covering context with **one tool call**
- Integration into the existing context assembly pipeline as an additional **candidate source**

This complements (not replaces) the existing context assembly design in:
- `docs/crumbs-engineering-design.md`
- `docs/unified-context-pipeline.md`

## Problem Statement

Today, agents interacting with Crumbs can:
- index repos
- run semantic search
- assemble context for a task/issue/topology

But when an agent needs to understand a *specific* function/class and its dependencies, it still has to either:
- pull broad “issue/task” context (often too wide), or
- repeatedly search/read files and stitch dependencies manually (many tool calls)

Scribe demonstrates that a dedicated **covering set** command is a high-leverage primitive for agents: it yields **minimal, dependency-complete context** with a single call.

## Goals

1. **One-call covering context**: CLI returns the minimal set of code needed to understand/modify a target.
2. **Entity-level precision** (optional): return only relevant function/class bodies and signatures.
3. **Budget-aware content control**: deterministic selection under a token budget with graceful degradation.
4. **Pipeline integration**: covering set output becomes a first-class *candidate source* in unified assembly.
5. **Agent-friendly output**: Markdown-first prompt payload (existing format) plus JSON for tool integration.

## Non-goals

- Replacing semantic retrieval (vector/FTS) or reranking.
- Building an IDE or replacing LSP.
- Perfect cross-language semantic resolution from day one.
- Remote repository support (local repos only in the first iteration).

## Current State (Crumbs)

From `docs/crumbs-engineering-design.md` and current implementation:

- Indexer stores:
  - chunked text + embeddings
  - symbol “definitions” and “references” (currently name-based)
  - file dependency edges derived from name matches
  - git co-change edges
- Context assembly supports:
  - explicit includes/pins
  - hybrid retrieval
  - topology/cochange expansion
  - token budgeting (line-span trimming)

Gaps relative to a Scribe-like workflow:

- No dedicated “covering set” CLI entrypoint for agents.
- No entity-level index (function/class boundaries + per-entity references).
- Budgeting is line-based; no semantic demotion (chunks/signatures).
- No explicit context positioning strategy (HEAD/MIDDLE/TAIL).

## Proposed Architecture

### High-level

```
Targets (file/entity/diff)
        │
        ▼
Covering Set Engine
  - resolve target(s)
  - traverse dependency graph(s)
  - (optional) entity-level expansion
  - apply limits + token budget
  - degrade content if needed
        │
        ├────────► CLI output (Markdown+XML, JSON)
        │
        └────────► Assembly pipeline seed (candidate pool boost)
```

### Key Concept: “Covering Set”

A covering set is the **smallest set of code blocks** that:
- includes the target, and
- includes enough dependency context to make the target understandable,
under configurable constraints (`max_depth`, `include_dependents`, token budget, etc.).

This is a distinct primitive from “search”: it is **graph-guided dependency closure**, not relevance ranking.

## Data Model Extensions

Crumbs currently stores symbol names and per-file dependency edges via name match. To support entity-level covering sets and demotion, we add an entity index.

### New Tables (proposed)

1. `entities`
   - `id` (stable hash)
   - `file_path`
   - `language`
   - `kind` (`function|method|class|struct|trait|interface|type|enum|module|…`)
   - `name`
   - `start_byte`, `end_byte`
   - `start_line`, `end_line`
   - `signature` (string; best-effort)
   - `doc` (string; optional)
   - `content_hash` (file hash at indexing time)

2. `entity_references`
   - `entity_id`
   - `reference` (string)
   - `reference_kind` (`call|type|import|field|…` best-effort)

3. `entity_dependency_edges`
   - `src_entity_id`
   - `dst_entity_id`
   - `weight` (optional; count/strength)
   - `reason` (`call|type|import|…`)

4. (Optional) `entity_resolutions`
   - `entity_id`
   - `reference`
   - `resolved_entity_id`

Notes:
- These tables allow fast entity extraction without re-parsing at query time.
- We can still keep the current file-level `file_dependency_edges` for fast file closure.

## Indexing Changes

### Entity extraction (tree-sitter)

During `crumbs index`, for supported languages:
- parse the full file content
- extract entities (functions/classes/etc.) with start/end byte/line
- extract per-entity references
- (optional) compute entity-level edges using resolution heuristics

Initial language set should match what we can support reliably:
- Rust, TypeScript/JavaScript, Python, Go

If parsing fails or language unsupported:
- skip entity indexing for that file, fall back to file-level covering sets.

### Resolution strategy (incremental)

Entity dependency edges require mapping a reference string to a definition:

Phase 1 (best-effort):
- resolve by exact name match within same workspace/package scope
- prefer same file; then same directory/module; then repository
- record ambiguity as warnings and choose the “best” candidate deterministically

Phase 2 (language-aware):
- incorporate import graph/module resolution per language
- handle aliases and qualified names

The covering set engine must remain functional even with imperfect resolution:
- if a reference can’t be resolved, keep it as an *unresolved dependency warning*

## Covering Set Engine

### Inputs

- `targets`: one or more target selectors:
  - file: `path/to/file.rs`
  - line anchor: `path/to/file.rs#L120`
  - entity: `path/to/file.rs:MyType` or `path/to/file.rs:my_function`
  - (optional) global entity search: `:MyType` with `--search-scope` (future)
- `granularity`: `file` | `entity`
- `direction`: dependencies / dependents / both
- `max_depth`
- `max_files` / `max_entities`
- `token_budget` and `reserved_output_tokens`
- `degradation_mode`: `trim_lines` | `demote` (FULL→CHUNK→SIGNATURE)
- `output_format`: `markdown_xml` (default) | `json`

### Outputs

- `CoveringSetResult`:
  - selected blocks/files/entities (with provenance + distance + reason)
  - warnings (unresolved references, missing targets, budget drops)
  - stats (examined, selected, max depth reached, token utilization)

## Budgeting and Degradation

Crumbs’ current approach trims line spans. Covering context benefits from a progressive strategy:

1. **FULL**: include full entity/file blocks
2. **CHUNK**: include semantic chunks (imports, interfaces, key functions)
3. **SIGNATURE**: include signatures + docs only

This preserves “shape” and API contracts even under tight budgets and avoids mid-function truncation.

Implementation approach:
- Reuse Crumbs token budgeting scaffolding, but allow a block to be “rendered” at different fidelity levels.
- Prefer degrading *supporting dependencies* first, never the explicit target unless budget is extremely constrained.

## Context Positioning (Optional but High Value)

Add an optional post-selection ordering step:
- **HEAD (20%)**: target + most query-relevant/high-centrality blocks
- **MIDDLE (60%)**: supporting context
- **TAIL (20%)**: core/high-centrality building blocks and key interfaces

This can be a simple deterministic ordering initially (based on:
distance from target, file PageRank, query relevance), and later refined.

## CLI: Agent-Facing Helper Commands

### 1) `crumbs context cover`

Purpose: one-shot covering context for a target file/entity.

Proposed interface:

```
crumbs context cover <TARGET...>
  --granularity (file|entity)         # default: file
  --include-dependents                # default: false
  --max-depth <N>                     # default: 2 (file), 2 (entity-focused)
  --max-files <N>                     # default: 40
  --max-entities <N>                  # default: 60
  --max-tokens <N>                    # default: embedder context length
  --reserved-output-tokens <N>        # default: 0
  --degrade (trim-lines|demote)       # default: demote when entity mode
  --output-format (markdown_xml|json) # default: markdown_xml
  --stdout                            # default: true (agent-friendly)
```

Targets:
- `path/to/file.rs`
- `path/to/file.rs#L120`
- `path/to/file.rs:my_function`
- `path/to/file.rs:MyType`

### 2) `crumbs context cover-diff`

Purpose: “what code do my current changes affect?” (code review / impact analysis).

Inputs:
- working tree vs HEAD (default)
- `--base <rev>` for CI/review workflows

Output:
- covering set for changed files (dependencies and optionally dependents), with line ranges.

### 3) `crumbs context signatures` (optional)

Purpose: fast signature-only export for a set of files/entities (great for tight budgets).

This can reuse the same entity index; output includes docs + signatures.

## Integration into Unified Assembly

The unified pipeline (`docs/unified-context-pipeline.md`) should treat covering set results as a **candidate source** with strong evidence:

- Add a `CandidateSource::CoveringSet { target, distance, direction }`
- When assembling `context issue`:
  - if `affected_symbols` exist, compute covering set seeds and merge them into the unified candidate pool
  - if no affected symbols, optionally attempt to infer targets from:
    - files referenced in issue text
    - top semantic hits (then compute covering set around them)

This yields:
- better precision (dependency complete)
- better diversity control (covering set gives core + supports)
- fewer “random” expansions compared to pure topology-only or semantic-only paths

## Phased Implementation Plan

### Phase 0: File-level covering set (fast path)

- Implement file-level covering sets using existing `file_dependency_edges` + cochange edges.
- Add `crumbs context cover` with `--granularity file`.
- Add `crumbs context cover-diff` (changed files → closure).
- Output uses existing Markdown+XML block format.

### Phase 1: Entity index + entity covering sets

- Add entity tables and indexing step during `crumbs index`.
- Implement `--granularity entity` for `crumbs context cover`.
- Entity extraction via tree-sitter, store start/end ranges and signature/doc.

### Phase 2: Demotion (FULL→CHUNK→SIGNATURE)

- Implement render-time degradation modes and integrate with budgeting.
- Prefer degrading non-target dependencies first; keep targets at highest fidelity possible.

### Phase 3: Context positioning

- Implement deterministic HEAD/MIDDLE/TAIL ordering for `cover` output.
- Optionally reuse for `context task` and `context issue`.

## Observability and Debuggability

Add debug output (in JSON and optionally in Markdown warnings):
- target resolution info (what entity/file matched)
- coverage stats (selected counts, max depth reached)
- unresolved references list
- budget decisions (drops/demotions)
- provenance per block (why included)

## Risks and Open Questions

1. **Resolution accuracy**: name collisions and imports make symbol resolution tricky.
2. **Indexing cost**: entity parsing adds time; must be incremental and cache-aware.
3. **Multi-language parity**: start with a small set of languages, with fallbacks.
4. **UX surface area**: keep CLI minimal; avoid a proliferation of subcommands.

## Success Metrics

- Median tool calls reduced for “understand this function” workflows (goal: 1).
- Higher relevance density (tokens used vs. tokens actually referenced by the model).
- Fewer assembly misses where key dependency types/functions are absent.
- Faster time-to-first-useful-context for agents (especially on medium+ repos).


# Unified Context Pipeline Design

Status: Proposal
Last updated: 2026-01-10
Author: AI Assistant

## Summary

Replace the current either/or context assembly logic with a **unified candidate graph pipeline** that always gathers from all available signals (semantic, topology, cochange, issue metadata), scores them with a feature-based function, and applies diversity-aware selection.

## Problem Statement

The current `crumbs issue context` command uses mutually exclusive paths:

```rust
// Current logic in main.rs:1048-1162
if issue.affected_symbols.is_empty() {
    // Semantic retrieval only (assembly pipeline)
} else {
    // Topology only (star neighborhood + depth)
}
```

This leaves value on the table:
- **Semantic-only path** misses structural relationships even when symbols could be inferred
- **Topology-only path** misses semantically relevant chunks that aren't graph neighbors
- Neither path leverages issue metadata (labels, dependencies, relates_to, duplicates)
- No way to combine evidence from multiple sources

## Goals

1. **Always use all available signals** — semantic, topology, cochange, and issue metadata work together
2. **Evidence accumulation** — chunks appearing in multiple sources get boosted, not arbitrarily chosen
3. **Transparent scoring** — per-candidate evidence vectors for debugging and tuning
4. **Single code path** — eliminate branching logic, simplify maintenance
5. **Extensible** — easy to add new signals without restructuring

## Non-Goals

- Training or fine-tuning models
- Changing the output format (Markdown + XML tags)
- Modifying the indexer or embedding pipeline
- Real-time/streaming retrieval

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CANDIDATE SOURCES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │    Semantic    │  │    Topology    │  │    History     │                 │
│  │  (vector+FTS   │  │  (star neigh   │  │   (cochange    │                 │
│  │   +rerank)     │  │   + depth)     │  │    edges)      │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
│         │                   │                    │                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │  Issue Meta    │  │  Dependency    │  │   Duplicate    │                 │
│  │ (labels, deps, │  │    Issues      │  │    Issues      │                 │
│  │  relates_to)   │  │  (seed boost)  │  │  (top hits)    │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        UNIFIED CANDIDATE POOL                                │
│  - Each candidate has: file_path, byte_range, text, chunk_hash              │
│  - Evidence vector: [semantic, topology, cochange, recency, centrality]     │
│  - Source provenance list for debugging                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FEATURE-BASED SCORING                                  │
│  - Weighted combination of evidence signals                                 │
│  - Additive boosts for issue metadata matches                               │
│  - Configurable weights (defaults provided)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SECOND-STAGE RERANK (optional)                          │
│  - Cross-encoder on top-N for precision boost                               │
│  - Blend rerank score with feature score                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DIVERSITY-AWARE SELECTION                                 │
│  - MMR or coverage quotas to avoid over-concentration                       │
│  - Max blocks per file cap                                                  │
│  - Explicit/pinned blocks always survive                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BUDGET & MERGE                                       │
│  - Token budgeting with line-span trimming (existing logic)                 │
│  - Deduplication by (file, byte_range)                                      │
│  - Same output format as today                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Design

### 1. New Candidate Model

Replace the simple `CandidateChunk` with an evidence-rich struct:

```rust
/// A candidate chunk with evidence from multiple sources.
pub struct UnifiedCandidate {
    // Identity
    pub id: String,
    pub file_path: String,
    pub start_byte: i64,
    pub end_byte: i64,
    pub chunk_hash: [u8; 32],
    pub start_line: i64,
    pub end_line: i64,
    pub text: String,

    // Evidence scores (None = not retrieved from this source)
    pub semantic_score: Option<f64>,
    pub topology_distance: Option<usize>,
    pub topology_weight: Option<f64>,
    pub cochange_weight: Option<f64>,
    pub recency_score: Option<f64>,
    pub centrality_score: Option<f64>,

    // Issue metadata boosts
    pub label_match_count: usize,
    pub dependency_match: bool,
    pub relates_to_match: bool,
    pub duplicate_origin: bool,

    // Computed scores
    pub feature_score: f64,
    pub rerank_score: Option<f64>,
    pub final_score: f64,

    // Provenance
    pub sources: Vec<CandidateSource>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CandidateSource {
    Explicit,
    Pinned,
    SemanticRetrieval,
    TopologyNeighbor { depth: usize },
    CochangeExpansion,
    DependencyIssue { issue_id: String },
    RelatedIssue { issue_id: String },
    DuplicateIssue { issue_id: String },
}
```

### 2. Unified Retrieval Stage

A single retrieval stage that gathers from all sources:

```rust
pub struct UnifiedRetrieveConfig {
    pub semantic_limit: usize,
    pub topology_depth: usize,
    pub topology_limit: usize,
    pub cochange_limit: usize,
    pub per_file_limit: usize,
    pub dependency_issue_limit: usize,
    pub related_issue_limit: usize,
    pub duplicate_limit: usize,
    pub duplicate_threshold: f64,
}

impl Default for UnifiedRetrieveConfig {
    fn default() -> Self {
        Self {
            semantic_limit: 50,
            topology_depth: 2,
            topology_limit: 30,
            cochange_limit: 20,
            per_file_limit: 5,
            dependency_issue_limit: 10,
            related_issue_limit: 10,
            duplicate_limit: 3,
            duplicate_threshold: 0.6,
        }
    }
}
```

Retrieval logic:

1. **Always run semantic retrieval** using `issue.summary_query()` (title + description + labels)
2. **If `affected_symbols` exist**, run topology star neighborhood expansion
3. **Expand via cochange** from all seed files collected so far
4. **Pull context from dependency issues** (blocking dependencies)
5. **Pull context from relates_to issues**
6. **Optionally pull from duplicate/similar issues**

All results merge into a single `HashMap<[u8; 32], UnifiedCandidate>` keyed by chunk_hash, accumulating evidence.

### 3. Scoring Function

```rust
pub struct ScoringConfig {
    // Weight for each evidence source (must sum to ~1.0)
    pub semantic_weight: f64,      // 0.40
    pub topology_weight: f64,      // 0.25
    pub cochange_weight: f64,      // 0.15
    pub recency_weight: f64,       // 0.10
    pub centrality_weight: f64,    // 0.05
    pub metadata_weight: f64,      // 0.05

    // Topology scoring
    pub topology_distance_decay: f64,  // score = 1.0 / (depth + 1)

    // Metadata boosts (additive)
    pub label_match_boost: f64,    // +0.05 per matching label
    pub dependency_boost: f64,     // +0.10 for dependency match
    pub relates_to_boost: f64,     // +0.08 for relates_to match
    pub duplicate_boost: f64,      // +0.05 for duplicate origin

    // Multi-source bonus
    pub multi_source_bonus: f64,   // +0.10 if appears in 2+ sources
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            semantic_weight: 0.40,
            topology_weight: 0.25,
            cochange_weight: 0.15,
            recency_weight: 0.10,
            centrality_weight: 0.05,
            metadata_weight: 0.05,
            topology_distance_decay: 1.0,
            label_match_boost: 0.05,
            dependency_boost: 0.10,
            relates_to_boost: 0.08,
            duplicate_boost: 0.05,
            multi_source_bonus: 0.10,
        }
    }
}
```

Scoring algorithm:
1. For each evidence type present, compute normalized score and multiply by weight
2. Divide by total weight of present evidence (handles missing sources gracefully)
3. Add metadata boosts (clamped to 1.0)
4. Add multi-source bonus if candidate appears in 2+ sources

### 4. Second-Stage Reranking

Apply the existing reranker to top-N candidates:

```rust
async fn rerank_candidates(
    reranker: &dyn RerankingProvider,
    query: &str,
    candidates: &mut [UnifiedCandidate],
    top_n: usize,
    rerank_blend: f64,  // 0.7 = 70% rerank, 30% feature
) -> Result<()>;
```

### 5. Diversity-Aware Selection

Prevent over-concentration using MMR (Maximal Marginal Relevance):

```rust
pub struct SelectionConfig {
    pub max_blocks: usize,
    pub max_per_file: usize,
    pub diversity_lambda: f64,  // 0.0 = pure diversity, 1.0 = pure relevance
}

fn select_with_mmr(
    candidates: &[UnifiedCandidate],
    config: &SelectionConfig,
) -> Vec<UnifiedCandidate>;
```

### 6. Semantic Query Construction

Build a richer query from issue metadata:

```rust
fn build_semantic_query(issue: &Issue) -> String {
    let mut parts = Vec::new();
    parts.push(issue.title.clone());
    if !issue.description.is_empty() {
        parts.push(issue.description.clone());
    }
    if !issue.acceptance_criteria.is_empty() {
        parts.push(issue.acceptance_criteria.clone());
    }
    if !issue.labels.is_empty() {
        parts.push(format!("labels: {}", issue.labels.join(", ")));
    }
    parts.join("\n")
}
```

---

## Implementation Plan

### Phase 1: Core Infrastructure

- [ ] Add `UnifiedCandidate` struct to `assembly/pipeline.rs`
- [ ] Add new `CandidateSource` variants (TopologyNeighbor, CochangeExpansion, DependencyIssue, RelatedIssue, DuplicateIssue)
- [ ] Add `ScoringConfig` and `UnifiedRetrieveConfig` structs
- [ ] Add `SelectionConfig` struct

### Phase 2: Unified Retrieval

- [ ] Implement `UnifiedRetrieve` stage that gathers from all sources
- [ ] Implement candidate merging logic (accumulate evidence by chunk_hash)
- [ ] Implement `build_semantic_query()` function
- [ ] Add helper to pull context from related/dependency issues

### Phase 3: Scoring and Selection

- [ ] Implement `compute_unified_score()` function
- [ ] Implement `select_with_mmr()` for diversity-aware selection
- [ ] Wire reranker into second-stage (optional, controlled by config)

### Phase 4: Integration

- [ ] Replace `IssueCommand::Context` either/or logic with unified path
- [ ] Update `assembly/output.rs` to show multi-source provenance
- [ ] Add scoring weights to config file (optional, with defaults)

### Phase 5: Cleanup

- [ ] Remove dead code (see "Code to Remove" section)
- [ ] Update tests
- [ ] Update documentation

---

## Code to Remove

The following code becomes obsolete and should be removed:

### `src/main.rs`

**Lines 1048-1162: The either/or branch in `IssueCommand::Context`**

```rust
// REMOVE: This entire if/else block
if issue.affected_symbols.is_empty() {
    // semantic-only path (~40 lines)
} else {
    // topology-only path (~60 lines)
}
```

Replace with a single call to the unified pipeline.

### `src/assembly/pipeline.rs`

**`CandidateSource` enum** — Replace with the new multi-source variant enum

```rust
// REMOVE (or replace):
pub enum CandidateSource {
    Explicit,
    Pinned,
    Primary,
    Expanded,
}
```

**`CandidateChunk` struct** — Replace with `UnifiedCandidate`

```rust
// REMOVE (or keep for backward compat, but migrate callers):
pub struct CandidateChunk {
    pub id: String,
    pub file_path: String,
    // ... simple fields without evidence vector
}
```

**`DefaultExpandGraph` stage** — Absorbed into unified retrieval

```rust
// REMOVE:
pub struct DefaultExpandGraph {
    pub max_expanded_files: usize,
}

impl ExpandGraph for DefaultExpandGraph { ... }
```

**`DefaultRefineAst` stage** — Absorbed into unified retrieval

```rust
// REMOVE:
pub struct DefaultRefineAst {
    pub per_file_limit: usize,
}

impl RefineAst for DefaultRefineAst { ... }
```

### `src/main.rs` helper functions

**`collect_seed_sources()`** — Logic moves into unified retrieval

```rust
// REMOVE or refactor:
fn collect_seed_sources(...) -> Result<Vec<SeedSource>> { ... }
```

**`select_topology_files()`** — Logic moves into unified retrieval

```rust
// REMOVE or refactor:
fn select_topology_files(...) -> Result<TopologySelection> { ... }
```

**`build_topology_blocks()`** — Logic moves into unified retrieval

```rust
// REMOVE or refactor:
async fn build_topology_blocks(...) -> Result<Vec<ContextBlock>> { ... }
```

### Traits that become unnecessary

**`ExpandGraph` trait** — No longer needed as a separate stage

```rust
// REMOVE:
#[async_trait(?Send)]
pub trait ExpandGraph: Send + Sync {
    async fn expand(...) -> Result<Handle<ExpandedCandidates>>;
}
```

**`RefineAst` trait** — No longer needed as a separate stage

```rust
// REMOVE:
#[async_trait(?Send)]
pub trait RefineAst: Send + Sync {
    async fn refine(...) -> Result<Handle<AstBlocks>>;
}
```

### Intermediate structs

**`ExpandedCandidates`** — No longer needed

```rust
// REMOVE:
pub struct ExpandedCandidates {
    pub chunks: Vec<CandidateChunk>,
    pub expanded_files: Vec<String>,
    pub warnings: Vec<String>,
}
```

**`AstBlocks`** — No longer needed (merge directly into budgeting)

```rust
// REMOVE:
pub struct AstBlocks {
    pub blocks: Vec<ContextBlock>,
    pub warnings: Vec<String>,
}
```

---

## New Code Structure

After cleanup, the pipeline simplifies to:

```
QueryInput
    │
    ▼
┌──────────────────────┐
│   UnifiedRetrieve    │  ← gathers from all sources, merges evidence
└──────────────────────┘
    │
    ▼
UnifiedCandidateSet (Vec<UnifiedCandidate> + warnings)
    │
    ▼
┌──────────────────────┐
│   ScoreAndSelect     │  ← feature scoring + optional rerank + MMR
└──────────────────────┘
    │
    ▼
ScoredCandidates (Vec<UnifiedCandidate> sorted by final_score)
    │
    ▼
┌──────────────────────┐
│   BudgetAndMerge     │  ← token budgeting, dedup (existing logic)
└──────────────────────┘
    │
    ▼
BudgetedBlocks
    │
    ▼
┌──────────────────────┐
│   AssembleContext    │  ← finalize output (existing logic)
└──────────────────────┘
    │
    ▼
AssembledContext
```

The 5-stage pipeline becomes 4 stages with clearer responsibilities.

---

## Configuration

Add to `config.toml` (all optional with defaults):

```toml
[context]
# Retrieval limits
semantic_limit = 50
topology_depth = 2
topology_limit = 30
cochange_limit = 20
per_file_limit = 5

# Scoring weights (should sum to ~1.0)
semantic_weight = 0.40
topology_weight = 0.25
cochange_weight = 0.15
recency_weight = 0.10
centrality_weight = 0.05

# Metadata boosts
label_match_boost = 0.05
dependency_boost = 0.10
relates_to_boost = 0.08

# Selection
max_blocks = 40
max_per_file = 5
diversity_lambda = 0.7

# Reranking
rerank_top_n = 30
rerank_blend = 0.7
```

---

## Migration Path

1. **Implement new pipeline alongside existing** — don't break current behavior
2. **Add `--unified` flag to `issue context`** — opt-in to new pipeline
3. **Compare results** — run both paths, compare output quality
4. **Remove old path** — once validated, delete the code listed above

---

## Success Metrics

- **Coverage**: % of relevant files appearing in context (measure via user feedback)
- **Precision**: % of context blocks actually used by the model (measure via telemetry)
- **Diversity**: # of unique files in context vs total blocks
- **Latency**: Time to assemble context (should be comparable or faster due to fewer DB round-trips)

---

## Open Questions

1. **Recency signal**: Pre-compute during indexing/topology build (via gix ingestion), not git blame at query time.
2. **Centrality caching**: Cache PageRank/centrality in the topology snapshot, not on-demand.
3. **Duplicate detection**: Use existing `find_duplicates()` (embeddings) and honor explicit `duplicate_of`/`superseded_by` when set (non-empty).
4. **Rerank scope**: Reuse the hybridsearch reranker as a second-stage rerank on the unified pool (top-N), then apply diversity selection on the blended score.

---

## References

- [Crumbs Engineering Design](./crumbs-engineering-design.md)
- [MMR: Maximal Marginal Relevance](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf)
- [Learning to Rank for Information Retrieval](https://www.microsoft.com/en-us/research/publication/learning-to-rank-for-information-retrieval/)

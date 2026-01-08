# Crumbs Context Engineering Design for GPT-5.2 + GPT-5.2-Codex

Status: Draft
Last updated: 2026-01-08
Owners: TBD

## Summary
This design doc specifies how assembled **code context** is produced and serialized for downstream callers. It covers retrieval, expansion, deduplication, token budgeting, and **Markdown-first output with lightweight XML tags** for GPT-5.2-Codex (primary), Claude Sonnet 4.5, and Gemini 3 Pro.

**Scope:** This tool is a **stateless context generator**. It does not store conversation state and does not call LLMs. It accepts inputs (query, scope, explicit includes/excludes, model limits), performs retrieval/expansion/budgeting, and outputs a prompt-ready **context payload**. Instruction authoring, agent UX, compaction, and evaluation tooling live elsewhere.

## Goals
- Deliver high-signal, model-ready context payloads for GPT-5.2 and GPT-5.2-Codex.
- Use retrieval + expansion + token budgeting to fit within model limits.
- Support long-horizon coding workflows by accepting pre-compacted inputs.
- Produce a **Markdown-first** output that is readable by humans and reliably parsable via lightweight XML tags.

## Non-goals
- Building a UI/UX for human review (CLI-first is fine).
- Training or fine-tuning models.
- Replacing upstream embedding or vector DB providers.
- Long-term memory, conversation state storage, or compaction.
- Initiating LLM calls (this tool only prepares context).
- Authoring or injecting system/developer instructions (caller-supplied).

## Design principles
- **Markdown-first, flat structure:** headings for humans; XML tags wrap the grouped payloads that must stay together.
- **Explicit scope and intent:** retrieval respects scope, includes, and excludes.
- **Deterministic trimming:** budget-driven, repeatable selection and truncation.
- **Transparent provenance:** include source and reason in block headers when available.

## Inputs (contract)
The context generator receives:
- **query**: the user request or task statement.
- **scope**: one or more scope selectors (current file, workspace, repo, remote repo).
- **explicit_includes**: files, symbols, or paths that must be included if possible.
- **explicit_excludes**: paths or patterns that must not be included.
- **pinned_items**: blocks that should survive trimming if possible.
- **model_limits**: max input tokens, max output tokens, and tokenizer ID.
- **repo_map** (optional): tree summary or module map provided by caller.

## Retrieval, expansion, and selection
Retrieval and expansion are **integral** to context assembly. Selection is a staged process:

1. **Apply scope**
   - Reject candidates outside the allowed scope set.

2. **Seed with explicit includes**
   - Add explicitly requested files/symbols first.
   - If an explicit include is missing, emit a warning in the output.

3. **Primary retrieval**
   - Hybrid search (vector + keyword), filtered by scope and excludes.

4. **Expansion**
   - Expand from primary hits via dependencies, graph neighbors, or co-change.
   - Expansion never crosses explicit excludes or scope boundaries.

5. **Rank and order**
   - Base ordering: explicit includes > pinned > primary retrieval > expansion.
   - Within each tier: relevance score, then file locality, then graph proximity.

6. **Deduplicate**
   - Identity = (file_path, start_line, end_line).
   - Keep highest scoring occurrence; prefer explicit > pinned > primary > expanded.

## Metadata schema for context blocks
Each retrieved block must include:

**Required fields:**
- `file_path`: absolute or repo-relative path
- `start_line`: starting line number (1-indexed)
- `end_line`: ending line number (inclusive)
- `relevance`: 0.0-1.0 score for retrieval relevance
- `source`: `explicit`, `pinned`, `primary`, `expanded`
- `scope`: scope selector that admitted this block
- `language`: programming language

**Optional fields:**
- `symbols`: list of function/class names in the block
- `token_count`: token count of the block content
- `reason`: short label explaining inclusion (e.g., explicit_include, dependency_neighbor)
- `last_modified`: git timestamp of last change
- `commit_sha`: most recent commit affecting these lines
- `author`: primary author from git blame

**Serialization notes:**
- `file_path` is rendered as `path:` in Markdown output.
- `start_line` and `end_line` are rendered as `Lines: start-end`.

## Canonical output format (Markdown + XML tags)
This is the default format. It is human-readable Markdown with lightweight XML tags wrapping only the grouped payloads that must stay together (structure map, summary map, each context block, user query). Tags are **not** used as section titles and there is no full XML document.

````text
## Repository: crumbs
Tech stack: Node.js, Rust

### Summary Map
<SUMMARY_MAP>
```
...tree...
```
</SUMMARY_MAP>

## Warnings
<WARNINGS>
- explicit include not found: src/missing.rs
</WARNINGS>

## Retrieved Context (4 blocks)

### Primary Results

<BLOCK>
path: src/config.rs
Lines: 29-115
relevance: 0.9737
source: primary
reason: retrieval
language: Rust
symbols: Embedding, api_key, context_length, ...
```rust
// ...
```
</BLOCK>

<BLOCK>
path: src/embedding.rs
Lines: 111-184
relevance: 0.8918
source: primary
reason: retrieval
language: Rust
symbols: EmbedApiResponse, EmbeddingProvider, ...
```rust
// ...
```
</BLOCK>

## User Query
<USER_QUERY>
We're adding a new embedding model with fastembed for local embedding
</USER_QUERY>
````

## Token budgeting and trimming
Budgeting is **parameterized by model limits** provided by the caller.

Definitions:
- `input_budget = model_max_input_tokens - reserved_output_tokens`
- `reserved_output_tokens` is task-dependent and caller-supplied.

Trimming rules (in order):
1. Never drop explicit includes or pinned blocks if any budget remains.
2. Drop lowest-relevance expanded blocks first.
3. If still over budget, trim oversized blocks by reducing line span.
4. If still over budget, drop lowest-relevance primary blocks.
5. Emit a warning in the output if any explicit include could not fit.

## File reference format (context payload)
Prefer **repo-relative** file paths, but absolute paths are allowed when needed.

Accepted formats:
- `src/app.py`
- `/home/user/repo/src/app.py`
- `src/app.py:42` or `/home/user/repo/src/app.py:42` (single line)
- `src/app.py#L42` or `/home/user/repo/src/app.py#L42` (GitHub-style)
- `Lines: 42-115` (line span, paired with the file path)
- `a/src/app.py:42` or `b/src/app.py:42` (diff prefix)

Not accepted:
- `file://` or `vscode://` URIs

## Deduplication strategy
When multiple chunks overlap:
1. Identity = `(file_path, start_line, end_line)`
2. Keep highest-scoring occurrence of each identity
3. If scores equal, prefer `explicit` > `pinned` > `primary` > `expanded`
4. If still tied, keep first occurrence

## Current capabilities in this repo
- Chunked indexing with configurable size/overlap/tokenizer (tokenizer configured under embedding); large file handling and parallel chunking (`src/indexer.rs`, `src/config.rs`).
- Embedding client with context-length and batch-size enforcement, plus rate limiting (`src/embedding.rs`).
- Hybrid search (vector + FTS) with configurable weighting (`src/search.rs`).
- Graph/AST extraction and git co-change expansion (`src/graph.rs`).
- Assembly pipeline stages (retrieve, expand, refine, budget, assemble), but not wired to a CLI or API (`src/assembly/*`).

## Gap analysis (best practice -> current state -> gap)
- **Scope semantics:** Scope is path-prefix based only; no named scopes (current file/workspace/multi-repo).
- **Explicit includes/pins:** File-path includes/pins supported; symbol-level includes not yet supported.
- **Budget trimming:** Line-span trimming is prefix-only; no middle-window or symbol-aware trimming.

## Proposed design
1. **Context prompt output**
   - Add a `crumbs prompt` subcommand that emits the canonical Markdown + XML tags payload.

2. **Scope-aware retrieval and expansion**
   - Accept scope, includes, excludes, and pins as inputs.
   - Enforce precedence: explicit > pinned > primary > expanded.

3. **Budgeting and trimming**
   - Tokenize using the caller-provided tokenizer.
   - Enforce `input_budget` with deterministic trimming rules.

4. **Block provenance**
   - Emit per-block provenance fields (source, reason) in the block header.

## Roadmap
- **M1 (Output + block headers):** Implemented (Markdown + XML tags with per-block provenance).
- **M2 (Selection inputs):** Implemented (path-prefix scope, includes, excludes, pins).
- **M3 (Budgeting):** Implemented (token budget + line-span trimming).

## Open questions
- How should scope be expressed for multi-repo workspaces (labeling, precedence)?
- What is the best trimming strategy for large files (top-N symbols vs. contiguous spans)?
- Should relevance be normalized per retrieval method or globally?

## References
- OpenAI GPT-5.2 prompting guide (cookbook.openai.com/examples/gpt-5/gpt-5-2_prompting_guide)
- GPT-5.2-Codex system prompt (github.com/openai/codex)
- Context engineering best practices (github.com/Meirtz/Awesome-Context-Engineering)
- RAG context window management patterns
- Repository-level code generation research (arXiv, Springer)

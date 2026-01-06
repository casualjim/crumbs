# Context Engineering Design for GPT-5.2 + GPT-5.2-Codex

Status: Draft
Last updated: 2026-01-05
Owners: TBD

## Summary
This design doc specifies the output format for assembled code context that will be included in agent prompts. It defines the structure, metadata, and formatting requirements for GPT-5.2-Codex (primary), Gemini 3 Pro, and Claude Sonnet 4.5. The doc also maps these requirements to current capabilities and identifies gaps in the assembly pipeline.

## Goals
- Deliver high-signal, model-ready context for GPT-5.2 and GPT-5.2-Codex.
- Use retrieval, graph expansion, and token budgeting to fit within model context limits.
- Support long-horizon coding workflows with compaction and conversation state.
- Provide repeatable quality via evaluation, model snapshot pinning, and telemetry.

## Non-goals
- Building a UI/UX for human review (CLI-first is fine).
- Training or fine-tuning models.
- Replacing upstream embedding or vector DB providers.

## External best practices (OpenAI)

### Model capabilities and controls
- GPT-5.2 supports a very large context window (400,000 tokens) and long outputs (128,000 tokens); budgets must be explicit to avoid overrun.
- GPT-5.2 exposes controls for reasoning effort and verbosity; set these per task (e.g., low reasoning + low verbosity for simple tasks, higher for complex).
- GPT-5.2-Codex improves long-horizon coding (larger changes, multi-file edits) and includes improved context compaction.

### Prompt structure and instruction hierarchy
- Use clear instruction hierarchy (system/developer/user) and keep prompts structured with explicit formatting.
- Put instructions at the beginning of prompts and separate instruction vs. context with clear delimiters.
- Ask for specific output formats and provide examples when needed.
- GPT-family models respond best to precise, explicit instructions; keep ambiguity low.

### Retrieval (RAG) and chunking
- Use embeddings to retrieve relevant snippets and augment prompts with only the most relevant chunks.
- Start with practical chunking defaults (around 800 tokens with 400 overlap) and tune per repo and language.
- Prefer hybrid retrieval (vector + keyword) and metadata filters to reduce noise.

### Context window management and compaction
- Use conversation state to avoid resending full history; compact long conversations as they grow.
- Compaction should preserve user intent while summarizing assistant/tool messages to reclaim space; use the compaction endpoint when possible.

### Prompt caching
- Use prompt caching by placing static instruction prefixes at the start of requests (caching uses the longest matching prefix).
- Keep dynamic, per-request content after the stable prefix for maximum cache hits.

### Evaluation and model stability
- Pin model snapshots and maintain evals to detect regressions across model updates.

## Context output format specification

This section defines how assembled context should be formatted for consumption by AI agents.

### Target models and format preferences

**Primary: GPT-5.2-Codex**
- Prefers XML tags for major sections: `<context>`, `<file>`, `<metadata>`
- File references: absolute or workspace-relative paths with single line numbers (`src/app.py:42`, not ranges)
- Supports `#Lline` syntax for GitHub-style references
- Minimal verbosity; flat structure preferred over deep nesting
- Code blocks in fenced markdown with language info strings

**Secondary: Claude Sonnet 4.5**
- Requires XML tags for reliable parsing
- Custom tag names encouraged (semantic naming)
- Supports nested structures for complex hierarchies
- Explicit references to tags in instructions improve reliability

**Tertiary: Gemini 3 Pro**
- Supports XML-like tags with PTCF pattern
- Standard tags: `<CONTEXT>`, `<CODE>`, `<METADATA>`
- Instructions should follow context in long prompts

### Metadata schema for code blocks

Each retrieved code block must include:

**Required fields:**
- `file_path`: Absolute or repo-relative path
- `start_line`: Starting line number (1-indexed)
- `end_line`: Ending line number (inclusive)
- `relevance`: Score between 0.0-1.0 indicating retrieval relevance
- `source`: One of `primary`, `expanded`, `cochange`, `dependency`
- `language`: Programming language for syntax context

**Optional fields:**
- `symbols`: List of function/class names defined in the block
- `token_count`: Number of tokens in the code content
- `last_modified`: Git timestamp of last change
- `commit_sha`: Most recent commit affecting these lines
- `author`: Primary author from git blame

### Context assembly structure

The assembled prompt should have these sections in order:

```xml
<system>
  <!-- Static instructions, forms cacheable prefix -->
  <role>You are an expert coding assistant...</role>
  <constraints>
    - Only use code from provided context
    - Cite file:line for all references
    - Do not invent or hallucinate code
  </constraints>
</system>

<repository_overview>
  <!-- Semi-static repo information, part of cacheable prefix -->
  <name>project-name</name>
  <structure>
    src/
      api/
      models/
      utils/
    tests/
  </structure>
  <tech_stack>Python 3.11, FastAPI, PostgreSQL</tech_stack>
</repository_overview>

<code_context>
  <!-- Dynamic per-query retrieved context -->
  <retrieved_files count="5" total_tokens="2400">
    <file>
      <path>src/utils/auth.py</path>
      <lines start="42" end="67"/>
      <relevance>0.92</relevance>
      <source>primary</source>
      <language>python</language>
      <symbols>authenticate, validate_token</symbols>
      <content><![CDATA[
def authenticate(username: str, password: str) -> bool:
    """Authenticate user credentials."""
    # implementation
]]></content>
    </file>
  </retrieved_files>

  <expanded_files count="2" total_tokens="800">
    <file>
      <path>src/models/user.py</path>
      <lines start="10" end="45"/>
      <relevance>0.76</relevance>
      <source>cochange</source>
      <symbols>User, get_by_username</symbols>
      <content><![CDATA[
class User:
    # implementation
]]></content>
    </file>
  </expanded_files>
</code_context>

<user_query>
  <!-- The actual user request -->
  How does authentication work in this codebase?
</user_query>
```

### Alternative flat format (Codex-optimized)

For GPT-5.2-Codex, a flatter Markdown structure may perform equally well:

    ## Repository: project-name
    Tech stack: Python 3.11, FastAPI, PostgreSQL

    ## Retrieved Context (5 files, 2400 tokens)

    ### Primary Results

    **src/utils/auth.py:42-67** (relevance: 0.92)
    Symbols: `authenticate`, `validate_token`

    ```python
    def authenticate(username: str, password: str) -> bool:
        """Authenticate user credentials."""
        # implementation
    ```

    ### Expanded Context (2 files, 800 tokens)

    **src/models/user.py:10-45** (relevance: 0.76, source: cochange)
    Symbols: `User`, `get_by_username`

    ```python
    class User:
        # implementation
    ```

### Token budget allocation

For GPT-5.2 with 400K context window:

```
Total context:    400,000 tokens
Reserved output:    4,000 tokens
Available input:  396,000 tokens

Allocation:
  System prompt:       2,000 tokens (static, cached)
  Repo overview:       1,000 tokens (semi-static, cached)
  User query:            500 tokens (dynamic)
  Retrieved context: 392,500 tokens (dynamic, trimmed to fit)
```

Budget enforcement:
1. Sort candidates by relevance score (descending)
2. For ties, prefer same-file locality
3. For ties, prefer direct dependencies over co-change neighbors
4. Accumulate tokens until budget exhausted
5. Truncate remaining candidates

### File reference format

When agents reference files in their responses:

**Accepted formats:**
- `src/app.py:42` - workspace-relative with line
- `/home/user/repo/src/app.py:42` - absolute with line
- `src/app.py#L42` - GitHub-style reference
- `a/src/app.py:42` or `b/src/app.py:42` - diff prefix

**Not accepted:**
- `file://` URIs
- `vscode://` URIs
- Line ranges like `:42-50` (Codex limitation)

### Deduplication strategy

When multiple chunks overlap:
1. Compute chunk identity as `(file_path, start_line, end_line)`
2. Keep highest-scoring occurrence of each identity
3. If scores equal, prefer `primary` > `expanded` > `cochange`
4. If still tied, keep first occurrence

## Current capabilities in this repo
- Chunked indexing with configurable size/overlap/tokenizer (tokenizer configured under embedding); large file handling and parallel chunking (`src/indexer.rs`, `src/config.rs`).
- Embedding client with context-length and batch-size enforcement, plus rate limiting (`src/embedding.rs`).
- Hybrid search (vector + FTS) with configurable weighting (`src/search.rs`).
- Graph/AST extraction and git co-change expansion (`src/graph.rs`).
- Assembly pipeline stages (retrieve, expand, refine, budget, assemble), but not wired to a CLI or API (`src/assembly/*`).

## Gap analysis (best practice -> current state -> gap)
- Model-aware prompt assembly: no CLI/API that assembles prompts or model-ready payloads.
- Token budgeting: no tokenizer-based budgeting for GPT-5.2 context size or output limits.
- Context window management: no compaction, summarization, or conversation state support.
- Retrieval quality: no query rewriting, reranking, or metadata filtering; no retrieval evals.
- Prompt caching: no concept of stable prefix or cache key handling.
- Tool integration: no native support for OpenAI tools like file search, nor model-specific controls (verbosity/reasoning effort).
- Eval/observability: no regression suite for retrieval quality, prompt correctness, or tool latency.
- Code-specific context layers: no repo map, dependency map, or change-focused context packaging.

## Detailed gap map

### Retrieval
Already:
- Vector + FTS hybrid retrieval with weighting.
- Git co-change expansion to pull related files.

Gaps:
- Query rewriting and decomposition (multi-part requests).
- Metadata filters (language, path prefix, recency, module tag, ownership).
- Reranking with a cross-encoder or lightweight heuristic.
- De-duplication by semantic similarity (not just byte ranges).
- Retrieval eval harness and regression tracking.

### Assembly and prompt output
Already:
- Pipeline stages for retrieval, expansion, refinement, budget, and assembly exist in code.
- Basic `ContextBlock` and `AssembledContext` types defined.

Gaps:
- No user-facing `context prompt` command or API endpoint.
- No XML or Markdown serialization of assembled context.
- No metadata enrichment (symbols, tokens, git info) in output.
- No stable prompt prefix generation for caching.
- No repo overview/map generation for context header.
- No output format selection (XML vs Markdown vs JSON).

### Token budgeting
Already:
- Embedding client enforces input context length.

Gaps:
- No model-aware token counting for prompt assembly.
- No explicit budget slices (instructions, user task, retrieved context, output headroom).
- No dynamic trimming or ordering by token cost.

### Context window management
Already:
- None.

Gaps:
- No conversation state storage or session model.
- No compaction, summarization, or state refresh strategy.
- No periodic refresh of stale context blocks.

### Model controls and tooling
Already:
- Provider dialect support for embeddings (OpenAI-compatible).

Gaps:
- No support for GPT-5.2 reasoning effort / verbosity in configuration.
- No first-class integration for OpenAI file search or tool calls.
- No policy for model snapshot pinning and rotation.

### Prompt caching
Already:
- None.

Gaps:
- No stable prefix generation to maximize cache hits.
- No caching diagnostics or cache hit metrics.

### Code context layers
Already:
- Graph extraction for symbols and references.
- Git history co-change graph.
- File path and byte range tracking in chunks.

Gaps:
- No line number extraction (only byte offsets currently).
- No symbol extraction from chunks (function/class names).
- No repo map or directory tree generation for overview section.
- No dependency graph serialization for prompt inclusion.
- No change-focused context packaging (diff-aware context).
- No file importance or ownership weighting from git history.
- No last-modified timestamps or commit SHA tracking per chunk.

### Evaluation and observability
Already:
- None.

Gaps:
- No eval suite for retrieval correctness or prompt quality.
- No quality gates for context packing and token utilization.
- No telemetry for retrieval precision, context hit rate, or model response outcomes.

## Proposed design
1. **Prompt assembly API**
   - New `context prompt` subcommand that outputs a model-ready payload.
   - Standard prompt template sections: system/developer instructions, task statement, constraints, retrieved context blocks, and file metadata.

2. **Token budgeting and ordering**
   - Tokenize using the target model’s tokenizer.
   - Allocate budgets for: instructions, task, retrieval blocks, and output headroom.
   - Prefer ordering by relevance score, then by file locality and graph proximity.

3. **Retrieval quality upgrades**
   - Add query rewriting and query decomposition for multi-part tasks.
   - Add reranking (cross-encoder or lightweight scoring) for top-N candidates.
   - Support metadata filters: language, path, recency, repo area, or issue tag.

4. **Context window management**
   - Introduce conversation state storage.
   - Add compaction using summaries or OpenAI’s compaction endpoint.
   - Maintain stable “core context” and refresh variable context blocks.

5. **Prompt caching support**
   - Treat the instruction prefix and repo summary as the cacheable prefix.
   - Keep per-request dynamic blocks after the prefix to maximize cache hits.

6. **Model controls & tool integration**
   - Expose GPT-5.2 reasoning effort and verbosity controls in config/CLI.
   - Provide optional use of OpenAI file search for large repos.

7. **Evaluation and observability**
   - Add a retrieval eval harness (golden queries with expected files).
   - Add prompt regression tests with model snapshots pinned.
   - Log retrieval precision, token budgets, and context hit rate.

## Roadmap
- **M1 (Output format + metadata):** Implement XML/Markdown serialization, metadata enrichment (line numbers, symbols, git info), repo overview generation.
- **M2 (Prompt assembly + budgets):** Implement `context prompt` CLI command, token budgeting with tiktoken, deduplication, ordering strategy.
- **M3 (Retrieval upgrades):** Query rewriting, metadata filters, reranking.
- **M4 (Compaction + state):** Conversation state + compaction routines.
- **M5 (Evals + caching):** Eval harness, prompt caching guidance, telemetry.

## Open questions
- How should prompt-budget tokenization relate to `embedding.tokenizer` (same tokenizer vs model-specific budgeting tokenizer)?
- Should we support all three output formats (XML, Markdown, JSON) or prioritize one?
- How to handle line number extraction from byte offsets for non-UTF8 files?
- Should symbol extraction use tree-sitter for all languages or heuristics?
- What's the optimal repo overview detail level (full tree vs summary)?
- How to detect tech stack automatically (parse package files vs heuristics)?
- Should deduplication be exact (line numbers) or fuzzy (overlapping ranges)?
- Do we standardize on a single embedding model, or allow per-project overrides?
- What storage format should hold conversation state and compaction artifacts?

## References
- OpenAI GPT-5.2 prompting guide (cookbook.openai.com/examples/gpt-5/gpt-5-2_prompting_guide)
- GPT-5.2-Codex system prompt (github.com/openai/codex)
- Claude Sonnet 4.5 XML tag documentation (docs.anthropic.com)
- Gemini 3 Pro structured prompting guide (ai.google.dev/gemini-api)
- Context engineering best practices (github.com/Meirtz/Awesome-Context-Engineering)
- RAG context window management patterns
- Repository-level code generation research (arXiv, Springer)

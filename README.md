# crumbs

`crumbs` is a Git-repo indexer and semantic search tool. It builds a local index
of your codebase (chunks + embeddings + symbol/reference graph + git history
co-change edges) so queries can be answered with high-signal code context. The
design target is model-ready prompt context assembly (see
`docs/crumbs-engineering-design.md`), and the current code provides the
indexing + retrieval foundation for that pipeline.

## What it does

- Chunks files with configurable size/overlap and embedding tokenizer.
- Stores embeddings for semantic retrieval.
- Extracts symbol/reference graphs from Tree-sitter queries.
- Adds git co-change history edges via `gix`.
- Supports hybrid retrieval (vector + FTS) for search.
- Tracks local issues in JSONL + SQLite and can assemble topology-aware context per issue.

## Key concepts

- Co-change: a lightweight graph derived from git history that links files
  which frequently change together in the same commits. This is used to
  expand context around a file or query to nearby, behaviorally-coupled files.
- Symbol/reference graph: a per-file graph of definitions and references
  extracted from Tree-sitter queries to connect identifiers across code.

## Quickstart

1) Create config and secrets files:
```
crumbs init
```

2) Set your embedder API key (or put it in `secrets.toml`):
```
export EMBEDDER_API_KEY="..."
```

3) Build the index:
```
crumbs index
```

4) Run a search:
```
crumbs search "add numbers"
```

Optional: create a repo-local config in the current repo:
```
crumbs init --local
```

Optional: assemble prompt-ready context:
```
crumbs context task "refactor the search pipeline"
```

Output is Markdown with lightweight XML tags by default.

Optional: set prompt token budgets:
```
crumbs context task --max-tokens 400000 --reserved-output-tokens 4000 "refactor the search pipeline"
```

Optional: use a separate tokenizer for prompt budgeting:
```
crumbs context task --prompt-tokenizer tiktoken:o200k_base "refactor the search pipeline"
```

Optional: retrieval tweaks (scope + pinning):
```
crumbs context task --scope src/ --pin src/search.rs "refactor the search pipeline"
```

For more examples and workflows, see `docs/user-guide.md`.

## Issue management (inspired by Grits)

`crumbs` includes a lightweight issue tracker modeled after the Grits workflow, with a
twin storage layer: a human-readable `issues.jsonl` file plus a SQLite table for fast queries.
The JSONL file lives alongside `crumbs.db` in your data directory (typically
`.config/crumbs/` under the repo when using local config).

Example usage:
```
crumbs issue create "Add reranking fallback" -d "Use FTS when reranker is unavailable" --label search
crumbs issue update cr-abc123 --status in-progress --add-symbol "src/search.rs"
crumbs issue list --status open
crumbs issue ready --assignee ivan
crumbs issue search "rerank"
crumbs issue edit cr-abc123
crumbs context issue cr-abc123 --depth 2 --limit 30
```

Credit: Issue management workflow and JSONL + SQLite dual storage are inspired by
[Grits](https://github.com/babybirdprd/grits).

## Configuration

Config is loaded in this order (later files override earlier):
- `--config-file <path>` (if provided)
- Per-repo overrides (optional):
  - `.config/crumbs.toml`
  - `.config/crumbs.secrets.toml`
  - `.config/crumbs/config.toml`
  - `.config/crumbs/secrets.toml`
- OS config dir (recommended default):
  - macOS: `~/Library/Application Support/crumbs/{config,secrets}.toml`
  - Windows: `%APPDATA%\\crumbs\\{config,secrets}.toml`
  - Linux: `${XDG_CONFIG_HOME}/crumbs/{config,secrets}.toml` or `~/.config/crumbs/{config,secrets}.toml`
- macOS also checks `~/.config/crumbs/{config,secrets}.toml`
- System config:
  - `/etc/crumbs/{config,secrets}.toml`

Minimal config example (projects are optional):
```
[embedding]
url = "https://api.deepinfra.com/v1/openai"
model = "Qwen/Qwen3-Embedding-0.6B"
tokenizer = "hf:Qwen/Qwen3-Embedding-0.6B"
dialect = "deepinfra"
timeout_seconds = 10
embedding_dim = 1024
context_length = 32768
max_batch_size = 15
tokens_per_minute = 1000000

[reranker]
url = "https://api.deepinfra.com/v1"
model = "Qwen/Qwen3-Reranker-0.6B"
dialect = "deepinfra"
timeout_seconds = 10

[chunking]
max_chunk_size = 1500
overlap = 0.2
max_parallel = 4
max_file_size = 5242880
large_file_threads = 4

[history]
depth = 10240
commit_size_limit_ratio = 1.0
multi_parents = false
issue_regex = "(#\\d+)"
# commit_exclude_regex = ""
# author_exclude_regex = ""
# path_specs = ""

[projects.example]
repo = "/path/to/repo"
# data_dir = "/path/to/data"
# database = "crumbs.db"

[search]
limit = 10
hybrid_weight = 0.6
```

## Build & test

```
mise test
mise format
```

## Install from source

When installing with Cargo, use the lockfile to avoid pulling incompatible
transitive dependencies:
```
cargo install --path . --locked --force
```

Note: tests that hit the embedder require a real API key in config or secrets.

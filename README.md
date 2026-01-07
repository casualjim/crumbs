# context

`context` is a Git-repo indexer and semantic search tool. It builds a local index
of your codebase (chunks + embeddings + symbol/reference graph + git history
co-change edges) so queries can be answered with high-signal code context. The
design target is model-ready prompt context assembly (see
`docs/context-engineering-design.md`), and the current code provides the
indexing + retrieval foundation for that pipeline.

## What it does

- Chunks files with configurable size/overlap and embedding tokenizer.
- Stores embeddings for semantic retrieval.
- Extracts symbol/reference graphs from Tree-sitter queries.
- Adds git co-change history edges via `cupido`.
- Supports hybrid retrieval (vector + FTS) for search.

## Key concepts

- Co-change: a lightweight graph derived from git history that links files
  which frequently change together in the same commits. This is used to
  expand context around a file or query to nearby, behaviorally-coupled files.
- Symbol/reference graph: a per-file graph of definitions and references
  extracted from Tree-sitter queries to connect identifiers across code.

## Quickstart

1) Create config and secrets files:
```
context init
```

2) Set your embedder API key (or put it in `secrets.toml`):
```
export EMBEDDER_API_KEY="..."
```

3) Build the index:
```
context index
```

4) Run a search:
```
context search "add numbers"
```

Optional: create a repo-local config in the current repo:
```
context init --local
```

Optional: assemble prompt-ready context:
```
context prompt "refactor the search pipeline"
```

Optional: choose Markdown output for readability:
```
context prompt --format markdown "refactor the search pipeline"
```

Optional: set prompt token budgets:
```
context prompt --max-tokens 400000 --reserved-output-tokens 4000 "refactor the search pipeline"
```

Optional: use a separate tokenizer for prompt budgeting:
```
context prompt --prompt-tokenizer tiktoken:o200k_base "refactor the search pipeline"
```

Optional: retrieval tweaks (filters, decomposition, rerank):
```
context prompt --path-prefix src/ --file-ext rs --decompose --rerank "refactor the search pipeline"
```

## Configuration

Config is loaded in this order (later files override earlier):
- `--config-file <path>` (if provided)
- Per-repo overrides (optional):
  - `.config/context.toml`
  - `.config/context.secrets.toml`
  - `.config/context/config.toml`
  - `.config/context/secrets.toml`
- OS config dir (recommended default):
  - macOS: `~/Library/Application Support/context/{config,secrets}.toml`
  - Windows: `%APPDATA%\\context\\{config,secrets}.toml`
  - Linux: `${XDG_CONFIG_HOME}/context/{config,secrets}.toml` or `~/.config/context/{config,secrets}.toml`
- macOS also checks `~/.config/context/{config,secrets}.toml`
- System config:
  - `/etc/context/{config,secrets}.toml`

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
# database = "context.db"

[search]
limit = 10
hybrid_weight = 0.6
```

## Build & test

```
cargo build
cargo test --all
```

Note: tests that hit the embedder require a real API key in config or secrets.

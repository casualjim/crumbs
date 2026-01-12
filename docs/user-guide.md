# crumbs user guide

This guide focuses on day-to-day usage (indexing, search, context assembly, and issue workflows).

## Install

From source (recommended for this repo):
```
cargo install --path . --locked --force
```

## First run (index + search)

1) Create config + secrets:
```
crumbs init
```

2) Provide API keys:
- Environment variables: `EMBEDDER_API_KEY`, `RERANKER_API_KEY`
- Or `secrets.toml` (written by `crumbs init`)

3) Build the index:
```
crumbs index
```

4) Search:
```
crumbs search "how does indexing work?"
```

## Repo-local config and data

To keep everything in the repo (useful for multiple projects):
```
crumbs init --local
```

This writes config under `.config/crumbs/` in the repo and stores `crumbs.db` and `issues.jsonl` alongside it.

## Ignoring files

Create a `.crumbsignore` file (gitignore-like patterns) to exclude files/directories from indexing.

## Context assembly

Generate prompt-ready context for an arbitrary task:
```
crumbs context task "refactor the search pipeline"
```

Common knobs:
- Limit retrieved scope: `--scope src/`
- Force-include specific files: `--pin path/to/file.rs`
- Token budget: `--max-tokens N` and `--reserved-output-tokens N`
- Prompt budgeting tokenizer: `--prompt-tokenizer tiktoken:o200k_base` (defaults to embedding tokenizer if unset)

Generate context for an issue (uses issue metadata + topology + retrieval):
```
crumbs context issue cr-abc123 --depth 2 --limit 30
```

## Issues (local-first)

Issues are stored in two places:
- `issues.jsonl` for human-friendly review and git workflows
- SQLite tables for fast queries and joins (search, topology, etc.)

Typical flow:
```
crumbs issue create "Fix flaky topology test" -d "Track down nondeterminism in refactor plan" --label topology
crumbs issue ready
crumbs issue update cr-abc123 --status in-progress --add-symbol "src/topology/refactor.rs"
crumbs issue edit cr-abc123
crumbs issue close cr-abc123
```

Helpers:
- Search: `crumbs issue search "rerank"`
- Stale: `crumbs issue stale --days 30`
- Triage in bulk: `crumbs issue triage cr-a cr-b --status open --add-label backlog`
- Infer from context: `crumbs issue infer error|diff|todo`

## Using with Codex (GPT-5.2)

A practical loop for agent-assisted work:
1) Pick work: `crumbs issue ready`
2) Get full context: `crumbs context issue <id> --depth 2 --limit 30`
3) Work and iterate, then keep the issue updated (`crumbs issue update` / `crumbs issue edit`)

If you use Codex skills, this repo includes a starter skill with prompt templates at
`skills/public/crumbs-codex/` (see `skills/public/crumbs-codex/references/prompts.md`).

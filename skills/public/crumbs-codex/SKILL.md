---
name: crumbs-codex
description: >-
  Use `crumbs` (the CLI in this repo) for local-first issue tracking and repo context retrieval:
  indexing (chunks+embeddings+graphs+git co-change), semantic search, and assembling prompt-ready
  context for an issue or task. Use when a user asks to set up `crumbs`, run
  `crumbs index/search/context`, manage issues (`crumbs issue create/update/edit/ready/close/...`),
  or generate optimized Codex/GPT-5.2 prompts/workflows based on `crumbs` output.
---

# Crumbs Codex

## Overview

Use `crumbs` to keep issues and context local to a repo, then feed the assembled context directly into Codex or other LLM tooling.

## Workflow

### 1) Set up config and secrets

Initialize config:
```
crumbs init
```

Use repo-local config/data:
```
crumbs init --local
```

Provide API keys (env or `secrets.toml`):
- `EMBEDDER_API_KEY`
- `RERANKER_API_KEY`

Optional: add `.crumbsignore` to exclude paths from indexing.

### 2) Build/refresh the index

```
crumbs index
```

### 3) Use the index

Search:
```
crumbs search "explain how issue ready works"
```

Assemble prompt-ready context for a task:
```
crumbs context task "refactor the search pipeline"
```

Assemble prompt-ready context for an issue:
```
crumbs context issue cr-abc123 --depth 2 --limit 30
```

## Issue workflow (local-first)

Create an issue:
```
crumbs issue create "Add ready command to CLI" -d "Rename next -> ready (breaking) and update docs"
```

Pick actionable work:
```
crumbs issue ready
```

Update/track work:
```
crumbs issue update cr-abc123 --status in-progress --add-symbol "src/main.rs"
crumbs issue edit cr-abc123
crumbs issue close cr-abc123
```

See `references/prompts.md` for GPT-5.2/Codex prompt templates.

## Developing crumbs (this repo)

Prefer project tasks:
- Format: `mise format`
- Test: `mise test`

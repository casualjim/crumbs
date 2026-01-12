---
name: crumbs-memory
description: >-
  Use `crumbs` (a local-first repo indexer + issue store) as durable, queryable long-term memory
  for a codebase: capture decisions, constraints, investigations, and task state in `crumbs issue`
  fields and comments; retrieve high-signal context with `crumbs search` and `crumbs context`
  (task/issue). Use when you need to (1) persist context across sessions/agents, (2) do fast
  historical recall ("why is this like this?"), (3) hand off work cleanly, or (4) assemble prompt-ready
  code context without re-reading the whole repo.
---

# Crumbs Memory

## Overview

Treat `crumbs` issues as structured “memory nodes” and the index as a retrieval layer. Capture what
matters once (decisions, invariants, affected symbols/files), then recall it quickly with a single
`crumbs context issue|task` call.

## Golden rules (borrow what works in Grits)

1) Normalize anchors: prefer repo-relative paths and forward slashes (Windows-safe).
2) Anchor early: always add at least one `--add-symbol` (file path or identifier string).
3) Prefer one-shot recall: `crumbs context issue <id> --depth 2 --limit 30` (includes issue header by default).
4) Keep memory versioned only on purpose: `crumbs issue sync` can commit/push (verify before running).

## Quick start (per repo)

1) Initialize config (prefer repo-local so memory can live with the repo):
```
crumbs init --local
```

2) Build/refresh the index:
```
crumbs index
```

Optional: start each session with a quick “what’s in flight?” check:
```
crumbs issue dashboard --limit 5
```

3) Create a memory issue:
```
crumbs issue create "Memory: <topic>" --type chore --label memory -d "<1-2 sentence summary>"
```

4) Attach the first “anchors” so retrieval stays precise:
```
crumbs issue update <id> --add-symbol "path/to/file.rs" --add-symbol "path/to/file.rs:MyType"
```

If you don’t know anchors yet, start from search:
```
crumbs search "the concept / identifier / error message"
```

## Memory workflow

### 1) Capture (turn work into memory)

Use issues as a structured, long-lived “state object”:

- Put the durable truth in fields:
  - `description`: what/why (stable problem statement)
  - `design`: decisions + rationale + constraints
  - `acceptance`: what “done” means
  - `notes`: investigation log + gotchas + next steps
- Attach anchors early: `--add-symbol` with file paths or symbol-ish strings.
- Use labels as retrieval hints: `--add-label memory,decision,rfc,investigation,hand-off`.

Practical “end of session” capture:
```
crumbs issue update <id> --status in-progress --add-symbol "path/to/touched/file.rs"
crumbs issue edit <id>
```

If you want a scriptable append-to-notes flow (no editor), use `scripts/append_notes.py`.
Example:
```
python3 scripts/append_notes.py <id> --title "Handoff" --author "<name>" <<'EOF'
What changed:
What is still true:
Next steps:
EOF
```

If you’re starting from a raw artifact, draft an issue first:
```
crumbs issue infer error "<paste error>"
crumbs issue infer diff "<paste diff>"
crumbs issue infer todo "<paste TODO block>"
```

### 2) Recall (turn memory into context)

Find prior memory:
```
crumbs issue search "key phrase"
crumbs issue list --status in-progress
crumbs issue ready --limit 5
crumbs issue list --status open
```

Assemble prompt-ready context from a memory issue (best for hand-offs):
```
crumbs context issue <id> --depth 2 --limit 30
```

Assemble context for a fresh task (best for “what do I need to touch?”):
```
crumbs context task "Implement: <goal>" --scope <likely/dir/>
```

Topology-guided context (good when you have a seed file but not the full surface area yet):
```
crumbs topology star --file "path/to/file.rs" --depth 2 --limit 50
crumbs topology path --start "path/to/a.rs" --end "path/to/b.rs"
crumbs topology hotspots --limit 10
```

### 3) Refresh (keep memory trustworthy)

When code changes materially, reindex so retrieval stays aligned with reality:
```
crumbs index
```

## Templates and prompts

- Issue field templates: `references/memory_issue_template.md`
- Agent prompt templates (session start/end + handoff): `references/prompts.md`

## Integrations (external trackers)

Use `external_ref` as the stable bridge to GitHub/Jira/Linear/etc. (and put full URLs in `design` if you want).

Examples:
```
crumbs issue update <id> --external-ref "github:owner/repo#123"
crumbs issue update <id> --external-ref "jira:PROJ-456"
crumbs issue update <id> --external-ref "linear:ENG-789"
```

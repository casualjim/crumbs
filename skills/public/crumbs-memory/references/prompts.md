# Prompt templates for using `crumbs` as long-term memory

Paste one of these into Codex when you want a repeatable memory workflow.

## Start a session from memory

```
You are working in a git repo that uses the `crumbs` CLI as a local-first memory store
(issues + index + context assembly).

1) Hydrate quickly:
   - `crumbs issue dashboard --limit 5`
2) Find the most relevant memory issue(s) with `crumbs issue search "<topic>"`.
3) For the top candidate, run `crumbs context issue <id> --depth 2 --limit 30`.
4) Summarize: current state, constraints, and next steps before making code changes.
5) If you need broader context, run `crumbs context task "<goal>" --scope <likely/dir/>`.
```

## End a session with a clean handoff

```
Use `crumbs` to write a durable handoff note:

1) Identify the issue representing this thread (or create one labeled `hand-off`).
2) Update it:
   - status (open/in-progress/blocked/closed)
   - affected symbols/files
   - notes: what changed, what is still true, what’s next
3) Prefer concrete anchors over prose (file paths, symbol names, commands run).
4) Reindex with `crumbs index` if code changed materially.
5) If your workflow versions issues in git, consider `crumbs issue sync --no-push` (it can commit/push).
```

## Turn “why is this like this?” into memory

```
Goal: explain a confusing behavior and store the answer for future sessions.

1) Create: `crumbs issue create "Memory: <why question>" --type chore --label "memory,investigation" -d "<one sentence>"`.
2) Use `crumbs search` to find relevant code and history; add anchors via `crumbs issue update --add-symbol ...`.
3) Write the distilled explanation into the issue `design` field (decision + rationale).
4) Assemble a compact context payload with `crumbs context issue <id> --depth 2 --limit 30`.
```

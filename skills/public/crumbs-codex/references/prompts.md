# Codex / GPT-5.2 prompt templates for crumbs

Paste one of these into Codex when you want a repeatable workflow.

## Work an issue (recommended default)

Goal: pick an actionable issue, assemble high-signal context, implement, then update the issue.

```
You are working in a git repo that uses the `crumbs` CLI for local-first issue tracking and context retrieval.

1) Run `crumbs issue ready` and pick the best issue to work on next.
2) Run `crumbs issue get <id>` and `crumbs context issue <id> --depth 2 --limit 30`.
3) Use the assembled context to make a concrete code change that resolves the issue.
4) Update the issue with affected symbols/files and status using `crumbs issue update ...`.
5) If the change introduces follow-ups, create new issues with clear titles and affected symbols.

Do not do backward-compat migrations unless explicitly requested.
Prefer `mise format` and `mise test` when validating changes in this repo.
```

## Triage and cleanup

```
Use `crumbs` to find and clean up issue hygiene:

1) Run `crumbs issue stale --days 30`.
2) For each stale issue, decide: close, downgrade priority, re-assign, or add labels.
3) Use `crumbs issue triage ...` for bulk updates and `crumbs issue edit <id>` for detailed edits.
4) Run `crumbs issue duplicates` and mark duplicates using `crumbs issue update --duplicate-of ...`.
```

## Turn an error into an issue

```
Given the error message below, use crumbs to turn it into an actionable issue:

1) Run `crumbs issue infer error "<paste error>"` to draft a candidate.
2) Search for existing matches with `crumbs issue search "<key phrases>"`.
3) If it already exists, update it; otherwise create it and add likely affected symbols/files.
4) Assemble context for investigation with `crumbs context task "debug: <short problem statement>" --scope <likely dir>/`.

Error:
<PASTE_ERROR_HERE>
```

## Refactor from topology

```
Use crumbs topology tools to guide a refactor:

1) Run `crumbs topology cycles` and identify the worst cycle.
2) Run `crumbs topology refactor` to get candidate cuts.
3) Create an issue describing the plan and add affected symbols/files.
4) Assemble context with `crumbs context task "apply topology refactor plan for <cycle>" --scope <dirs>/`.
5) Implement and validate; then close the issue.
```


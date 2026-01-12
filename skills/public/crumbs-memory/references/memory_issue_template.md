# Memory issue template (fields + sections)

Use this as a checklist for what to put into a long-lived “memory” issue.

## Recommended issue metadata

- Title: `Memory: <topic>`
- Type: `chore` (usually) or `task` (if it’s actionable work)
- Labels: `memory` plus one of `decision`, `investigation`, `hand-off`, `rfc`
- Anchors: `--add-symbol` with repo-relative paths (forward slashes) and important identifiers
- External link (optional): `external_ref` for GitHub/Jira/Linear IDs

## Suggested field contents

### description

- What this is about (1–3 sentences)
- Why it exists / what problem it prevents
- Scope boundaries (what is explicitly out of scope)

### design

- Decision log (what we chose + why)
- Constraints/invariants (must-hold truths)
- External references (links, tickets) as plain text (and/or `external_ref`)

### acceptance

- “Done means…” (observable outcomes)
- Safety checks (tests, commands, rollout notes)

### notes

Prefer a dated running log, optimized for handoff:

```
### 2026-01-12
- Current state:
- Key files/symbols:
- Risks:
- Next steps:
```

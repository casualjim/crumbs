# Repository Guidelines

## Project Overview

`remark` is a Rust terminal UI (TUI) for writing code review notes in Git repos. Reviews are stored as git notes (default `refs/notes/remark`) and can be rendered into a collated prompt via `remark prompt`.

## Project Structure

- `src/`: Rust source code (single binary crate; entrypoint in `src/main.rs`).
- `.github/workflows/`: CI and release automation (Rust checks, release-plz, cargo-dist).
- `wix/`: Windows installer metadata used by `cargo-dist`/WiX.
- `dist-workspace.toml`: release configuration (artifacts).

## Build, Test, and Development Commands

- `cargo build`: build a debug binary.
- `cargo build --release`: build `target/release/remark`.
- `cargo run -- <args>`: run locally (example: `cargo run -- prompt --filter base`).
- `cargo fmt --all`: format code (CI enforces `-- --check`).
- `cargo clippy --all-targets --all-features -- -D warnings`: lint (CI treats warnings as errors).
- `cargo test --all`: run unit tests.

## Coding Style & Naming Conventions

- Rust edition: 2024 (see `Cargo.toml`).
- Formatting: `rustfmt` with default settings; keep diffs small and idiomatic.
- Linting: address `clippy` warnings instead of suppressing them.
- Naming: follow standard Rust conventions (`UpperCamelCase` types, `snake_case` fns/modules).

## Testing Guidelines

- Tests are primarily unit tests colocated in modules (`mod tests { ... }` in `src/*.rs`).
- Prefer small, behavior-focused tests; add new tests near the code they cover.
- Run `cargo test --all` before opening a PR.

## Commit & Pull Request Guidelines

- Commit messages follow Conventional Commits (examples: `feat: ...`, `fix: ...`, `chore: ...`, `refactor: ...`, `test: ...`).
- PRs should include: a clear description, rationale, and any user-facing behavior changes.
- Ensure CI passes (`fmt`, `clippy`, `test`) before merging.
- **CRITICAL**: Do not create commits unless the user explicitly asks. Always confirm before staging or committing changes.
- **CRITICAL**: No fakes. Call the embedding API directly (no mocks or simulated responses).

## Release Notes (Maintainers)

Releases are cut automatically after CI passes on pushes to `main`. The `Cut Release` workflow runs `git-cliff` to update `CHANGELOG.md` and then uses `cargo-release` to bump the patch version and tag `vX.Y.Z`. Pushing the tag triggers `cargo-dist` to publish GitHub Release artifacts. A repo secret named `RELEASE_TOKEN` is required so the tag push can trigger the `Release` workflow. There is no release PR flow.


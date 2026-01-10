# Agent Coding Guidelines

> **Important:** Prefer the `mise` tasks for installs, builds, tests, and formatting. Only use raw toolchain commands when no `mise` wrapper exists, and call that out explicitly.
>
> **CRITICAL: Do NOT run git mutations without explicit approval from the user** Do NOT ever run git checkout/revert/restore/reset without EXPLICIT APPROVAL from the USER

## Tool Preference Order

When working with code, **strongly prefer** higher-priority tools over lower-priority ones. Only fall back to lower-priority tools if the higher ones aren't sufficient:

### 1. Language Server Protocol (LSP) Tools - HIGHEST PRIORITY
- **Rust**: `rust-lsp` (e.g., `mcp__rust-lsp__search`, `mcp__rust-lsp__outline`, `mcp__rust-lsp__references`)
- **TypeScript**: `typescript-lsp`
- **Go**: `golang-lsp`

**Why LSP first?** Type-aware, accurate, context-sensitive, IDE-quality navigation and refactoring.

### 2. Code Index Tools - SECOND PRIORITY
- Faster than raw text search tools
- Indexed file and symbol search
- Use when LSP doesn't provide what you need or for cross-language searches

### 3. Default CLI Tools - FALLBACK ONLY
- `rg` (ripgrep), `grep`, `fd`, `find`, `cat`, `ls`, `sed`, etc.
- Use only when LSP and Code Index can't accomplish the task
- Still useful for non-code files or when you need regex patterns across arbitrary text

### GitHub Operations
- **Always use `gh` CLI** for GitHub operations (PRs, issues, repos, etc.)
- Examples: `gh pr list`, `gh issue create`, `gh repo view`

## Build, Test, and Development Commands
Always default to the `mise` tasks below; only run direct toolchain commands if no `mise` wrapper exists and note the deviation.

**For code navigation and understanding, follow the Tool Preference Order above!** See the "Code Navigation with LSP" section for detailed LSP commands.

- `mise install`: Install pinned Rust, Bun, Wrangler, etc.
- `mise format`: Quick Checks for this codebase
- `mise test`: All tests (Rust nextest).

**IMPORTANT** after changes ALWAYS run `mise format` and if you modified rust code also `mise test`

## Code Navigation with LSP

**IMPORTANT: Follow the Tool Preference Order!** LSP tools are your PRIMARY navigation method for supported languages (Rust, TypeScript, Go).

LSP tools should be used for:
- Finding symbols and definitions
- Navigating to references
- Getting function signatures and documentation
- Understanding code structure
- Finding implementations and usages

**Only fall back to Code Index or CLI tools** when LSP doesn't provide what you need (e.g., cross-language searches, regex patterns, non-code files).

### Rust LSP Commands Available

Use these `mcp__rust-lsp__*` tools for navigation:

```bash
# Get file structure and symbols
mcp__rust-lsp__outline <file_path>

# Search for symbols across the codebase
mcp__rust-lsp__search <query>

# Find all references to a symbol
mcp__rust-lsp__references <file_path> <line> <character>

# Get detailed info about a symbol at cursor position
mcp__rust-lsp__inspect <file_path> <line> <character>

# Get code completions at a position
mcp__rust-lsp__completion <file_path> <line> <character>

# Rename a symbol across the codebase
mcp__rust-lsp__rename <file_path> <line> <character> <new_name>

# Get diagnostics (errors/warnings) for a file
mcp__rust-lsp__diagnostics <file_path>
```

### Navigation Examples

```bash
# Find all search-related services
mcp__rust-lsp__search "SearchService"

# Explore the main application structure
mcp__rust-lsp__outline "crates/slipstreamd/src/lib.rs"

# Find all references to AppState
mcp__rust-lsp__references "crates/slipstreamd/src/app.rs" 16 1

# Inspect a function to get its documentation
mcp__rust-lsp__inspect "crates/embedding/src/lib.rs" 127 1

# Get completions for method calls
mcp__rust-lsp__completion "crates/slipstreamd/src/routes.rs" 42 20
```

### Why Use LSP First?

- **Accurate**: Understands the language's type system and module resolution
- **Fast**: Instant navigation without scanning files
- **Context-aware**: Knows about imports, traits, generics, interfaces
- **Complete**: Shows parameters, return types, documentation
- **IDE-quality**: Same experience as modern IDEs

**Remember: Follow the Tool Preference Order - LSP first, Code Index second, CLI tools as fallback!**


> REMINDER:
> ALWAYS get approval from the user for git checkout/reset/restore/revert/...
> NEVER run destructive git commands without explicit approval


## Code Style & Formatting
- Rust:
  - Use `eyre::Result` for error handling, `thiserror` for domain errors
  - No `unwrap()` or `expect()` in public APIs
  - Async streaming first - avoid `collect()` patterns
  - Imports: Group std/core, external crates, and internal modules separately
  - Formatting: run `mise format`; never invoke `cargo fmt` directly
  - Strict error handling - fail spectacularly, don't swallow errors
- TypeScript:
  - Strict mode with no `any` or `unknown`
  - Bun package manager
  - Double quotes for strings
- General:
  - 2-space indentation (except Python which uses 4)
  - LF line endings with final newline
  - Trim trailing whitespace
  - UTF-8 encoding

## Naming Conventions
- Rust: snake_case for variables/functions, PascalCase for types
- TypeScript: camelCase for variables/functions, PascalCase for types
- Files: snake_case for Rust, camelCase for TypeScript

## Error Handling
- Rust: Use `eyre::Result` for function returns, `thiserror` for domain-specific errors
- TypeScript: Proper error catching and handling without swallowing
- Never ignore errors - propagate or handle explicitly

## Commit Messages
- Write clear, descriptive commit messages in plain English
- Do NOT use conventional commits, semantic commits, or any commit prefixes (no "feat:", "fix:", "refactor:", etc.)
- Focus on WHAT changed and WHY, not the type of change
- First line should be a clear summary (50-72 chars recommended)
- Use the body for detailed explanation if needed
- Reference issue IDs when relevant (e.g., "Closes: slipstream-24")

Good examples:
- "Split search into dedicated Searcher service"
- "Add reranking provider for DeepInfra Qwen3-Reranker"
- "Fix flaky test by increasing tolerance for timing variance"

Bad examples:
- "refactor(embedding): Split search into dedicated Searcher service"
- "feat: add reranking provider"
- "fix: flaky test"

## SUPER IMPORTANT
- Do NOT run git commands that can result in loss of work unilaterally. ALWAYS get approval from the user for git checkout/reset/restore/revert/...

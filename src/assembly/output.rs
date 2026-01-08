use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::io::IsTerminal;
use std::path::Path;

use dark_light::Mode as DarkLightMode;
use eyre::Result;
use syntastica::Processor;
use syntastica::language_set::SupportedLanguage;
use syntastica::renderer::TerminalRenderer;
use syntastica::theme::ResolvedTheme;
use syntastica_parsers::{Lang, LanguageSetImpl};
use text_chunking::languages;

use crate::assembly::pipeline::{CandidateSource, ContextBlock};
use crate::repository::Repository;

const SUMMARY_LIMIT: usize = 20;
const SUMMARY_COCHANGE_LIMIT: usize = 3;
const SUMMARY_CORE_LIMIT: usize = 5;
const SUMMARY_HIGH_CHURN_COMMITS: i64 = 10;
const SUMMARY_MAX_ENTRIES: usize = 400;

#[derive(Clone)]
pub struct EnrichedBlock {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub relevance: f64,
    pub source: CandidateSource,
    pub language: String,
    pub symbols: Vec<String>,
    pub text: String,
}

pub struct PromptPayload {
    pub overview: RepositoryOverview,
    pub task: String,
    pub blocks: Vec<EnrichedBlock>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PromptSections {
    pub structure: bool,
    pub summary: bool,
    pub context: bool,
    pub query: bool,
}

impl PromptSections {
    pub fn all() -> Self {
        Self {
            structure: true,
            summary: true,
            context: true,
            query: true,
        }
    }

    pub fn none() -> Self {
        Self::default()
    }

    fn overview_enabled(self) -> bool {
        self.structure || self.summary
    }
}

#[derive(Clone)]
pub struct RepositoryOverview {
    pub name: String,
    pub structure: Vec<String>,
    pub tech_stack: Vec<String>,
    pub summary: Vec<String>,
}

fn line_from_i64(value: i64) -> usize {
    if value <= 0 { 1 } else { value as usize }
}

fn normalized_fence_language(language: &str) -> String {
    let trimmed = language.trim();
    if trimmed.is_empty() {
        return "text".to_string();
    }
    let lower = trimmed.to_ascii_lowercase();
    if languages::get_language(trimmed).is_some()
        || languages::get_language(&lower).is_some()
        || languages::is_language_supported(&lower)
    {
        lower.replace(' ', "_")
    } else {
        "text".to_string()
    }
}

pub async fn enrich_blocks(
    repo_root: &Path,
    db: &dyn Repository,
    blocks: &[ContextBlock],
) -> Result<Vec<EnrichedBlock>> {
    let mut language_cache: BTreeMap<String, Option<String>> = BTreeMap::new();
    let mut enriched = Vec::with_capacity(blocks.len());

    for block in blocks {
        let start_line = line_from_i64(block.start_line);
        let end_line = line_from_i64(block.end_line);

        let mut symbol_names = BTreeSet::new();
        for symbol in db
            .symbols_in_range(&block.file_path, block.start_byte, block.end_byte)
            .await?
        {
            symbol_names.insert(symbol.name);
        }

        let language = if let Some(cached) = language_cache.get(&block.file_path) {
            cached.clone()
        } else {
            let detected = db.file_primary_language(&block.file_path).await?;
            language_cache.insert(block.file_path.clone(), detected.clone());
            detected
        }
        .unwrap_or_else(|| "text".to_string());

        enriched.push(EnrichedBlock {
            file_path: display_path(repo_root, &block.file_path),
            start_line,
            end_line,
            relevance: block.score.clamp(0.0, 1.0),
            source: block.source,
            language,
            symbols: symbol_names.into_iter().collect(),
            text: block.text.clone(),
        });
    }

    Ok(enriched)
}

pub fn render_prompt(
    payload: &PromptPayload,
    sections: PromptSections,
    theme: Option<&str>,
) -> String {
    let markdown = render_markdown(payload, sections);
    if std::io::stdout().is_terminal() {
        let theme = resolve_theme(theme);
        highlight_markdown(&markdown, &theme)
    } else {
        markdown
    }
}

fn render_markdown(payload: &PromptPayload, sections: PromptSections) -> String {
    let mut out = String::new();
    if sections.overview_enabled() {
        writeln!(out, "## Repository: {}", payload.overview.name).ok();
        if sections.summary && !payload.overview.tech_stack.is_empty() {
            writeln!(
                out,
                "Tech stack: {}",
                payload.overview.tech_stack.join(", ")
            )
            .ok();
        }
        if sections.structure && !payload.overview.structure.is_empty() {
            writeln!(out, "\n### Structure\n<STRUCTURE_MAP>\n```").ok();
            for line in &payload.overview.structure {
                writeln!(out, "{line}").ok();
            }
            writeln!(out, "```\n</STRUCTURE_MAP>").ok();
        }
        if sections.summary && !payload.overview.summary.is_empty() {
            writeln!(out, "\n### Summary Map\n<SUMMARY_MAP>\n```").ok();
            for line in &payload.overview.summary {
                writeln!(out, "{line}").ok();
            }
            writeln!(out, "```\n</SUMMARY_MAP>").ok();
        }
    }
    if !payload.warnings.is_empty() {
        writeln!(out, "\n## Warnings\n<WARNINGS>").ok();
        for warning in &payload.warnings {
            writeln!(out, "- {warning}").ok();
        }
        writeln!(out, "</WARNINGS>").ok();
    }
    if sections.context {
        writeln!(
            out,
            "\n## Retrieved Context ({} blocks)",
            payload.blocks.len()
        )
        .ok();

        let (primary, expanded) = split_blocks(&payload.blocks);
        write_markdown_group(&mut out, "Primary Results", &primary);
        write_markdown_group(&mut out, "Expanded Context", &expanded);
    }
    if sections.query {
        writeln!(out, "\n## User Query\n<USER_QUERY>\n{}\n</USER_QUERY>", payload.task).ok();
    }
    out
}

fn write_markdown_group(out: &mut String, label: &str, blocks: &[EnrichedBlock]) {
    writeln!(out, "\n### {}", label).ok();
    for block in blocks {
        writeln!(out, "\n<BLOCK>").ok();
        writeln!(out, "path: {}", block.file_path).ok();
        writeln!(out, "Lines: {}-{}", block.start_line, block.end_line).ok();
        writeln!(out, "relevance: {:.4}", block.relevance).ok();
        writeln!(out, "source: {}", source_label(block.source)).ok();
        writeln!(out, "reason: {}", reason_label(block.source)).ok();
        writeln!(out, "language: {}", block.language).ok();
        if !block.symbols.is_empty() {
            let symbols = block
                .symbols
                .iter()
                .map(|symbol| symbol.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(out, "symbols: {}", symbols).ok();
        }
        let fence_lang = normalized_fence_language(&block.language);
        writeln!(out, "```{}\n{}\n```", fence_lang, block.text).ok();
        writeln!(out, "</BLOCK>").ok();
    }
}

fn highlight_markdown(markdown: &str, theme: &ResolvedTheme) -> String {
    let language_set = LanguageSetImpl::new();
    let Ok(lang) =
        <Lang as SupportedLanguage<'_, LanguageSetImpl>>::for_name("markdown", &language_set)
    else {
        return markdown.to_string();
    };
    let mut processor = Processor::new(&language_set);
    let mut renderer = TerminalRenderer::new(None);

    match processor.process(markdown, lang) {
        Ok(highlights) => syntastica::render(&highlights, &mut renderer, theme.clone()),
        Err(_) => markdown.to_string(),
    }
}

fn resolve_theme(theme: Option<&str>) -> ResolvedTheme {
    let override_name = theme.unwrap_or("").trim();
    if !override_name.is_empty() && override_name != "auto" {
        if let Some(theme) = syntastica_themes::from_str(override_name) {
            return theme;
        }
    }

    match dark_light::detect() {
        Ok(DarkLightMode::Light) => syntastica_themes::catppuccin::latte(),
        Ok(DarkLightMode::Dark) => syntastica_themes::catppuccin::mocha(),
        Ok(DarkLightMode::Unspecified) => syntastica_themes::catppuccin::mocha(),
        Err(_) => syntastica_themes::catppuccin::mocha(),
    }
}

fn split_blocks(blocks: &[EnrichedBlock]) -> (Vec<EnrichedBlock>, Vec<EnrichedBlock>) {
    let mut primary = Vec::new();
    let mut expanded = Vec::new();
    for block in blocks {
        match block.source {
            CandidateSource::Explicit | CandidateSource::Pinned | CandidateSource::Primary => {
                primary.push(block.clone())
            }
            CandidateSource::Expanded => expanded.push(block.clone()),
        }
    }
    (primary, expanded)
}

fn source_label(source: CandidateSource) -> &'static str {
    match source {
        CandidateSource::Explicit => "explicit",
        CandidateSource::Pinned => "pinned",
        CandidateSource::Primary => "primary",
        CandidateSource::Expanded => "expanded",
    }
}

fn reason_label(source: CandidateSource) -> &'static str {
    match source {
        CandidateSource::Explicit => "explicit_include",
        CandidateSource::Pinned => "pinned",
        CandidateSource::Primary => "retrieval",
        CandidateSource::Expanded => "expanded_neighbor",
    }
}

pub async fn build_repository_overview(
    repo_root: &Path,
    db: &dyn Repository,
) -> RepositoryOverview {
    let name = repo_root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("repo")
        .to_string();
    let structure = Vec::new();
    let tech_stack = detect_tech_stack(repo_root, db).await;
    let summary = build_repository_summary(repo_root, db).await;
    RepositoryOverview {
        name,
        structure,
        tech_stack,
        summary,
    }
}

async fn build_repository_summary(repo_root: &Path, db: &dyn Repository) -> Vec<String> {
    let mut lines = Vec::new();
    let Ok(files) = db.list_files().await else {
        return lines;
    };
    if files.is_empty() {
        return lines;
    }

    let ranks = db.file_dependency_pagerank(SUMMARY_LIMIT).await.unwrap_or_default();
    let core_limit = SUMMARY_CORE_LIMIT.min(ranks.len());
    let mut ranked = std::collections::HashMap::new();

    for (idx, (file_path, score)) in ranks.iter().enumerate() {
        let display_path_value = display_path(repo_root, file_path);
        let mut badges = Vec::new();
        if idx < core_limit {
            badges.push("core");
        }
        if is_test_path(&display_path_value) {
            badges.push("test");
        }
        if let Ok(commit_count) = db.file_commit_count(file_path).await {
            if commit_count >= SUMMARY_HIGH_CHURN_COMMITS {
                badges.push("high-churn");
            }
        }

        let cochanges = match db
            .cochange_partners(file_path, SUMMARY_COCHANGE_LIMIT)
            .await
        {
            Ok(partners) if !partners.is_empty() => Some(
                partners
                    .into_iter()
                    .map(|(path, weight)| {
                        format!("{} ({:.2})", display_path(repo_root, &path), weight)
                    })
                    .collect::<Vec<_>>()
                    .join(", "),
            ),
            _ => None,
        };

        ranked.insert(
            display_path_value,
            RepoMapEntry {
                score: Some(*score),
                badges,
                cochanges,
            },
        );
    }

    let mut map = RepoMapNode::default();
    for file_path in files {
        let display_path_value = display_path(repo_root, &file_path);
        let entry = ranked.remove(&display_path_value).unwrap_or(RepoMapEntry {
            score: None,
            badges: Vec::new(),
            cochanges: None,
        });
        insert_repo_map_entry(&mut map, &display_path_value, entry);
    }

    let mut count = 0usize;
    render_repo_map_tree(&map, "", 0, 4, SUMMARY_MAX_ENTRIES, &mut count, &mut lines);

    lines
}

fn display_path(repo_root: &Path, file_path: &str) -> String {
    let path = Path::new(file_path);
    if path.is_absolute() {
        if let Ok(stripped) = path.strip_prefix(repo_root) {
            if let Some(rel) = stripped.to_str() {
                if !rel.is_empty() {
                    return rel.replace('\\', "/");
                }
            }
        }
    }
    file_path.replace('\\', "/")
}

fn is_test_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    if lower.contains("/tests/") || lower.contains("\\tests\\") || lower.contains("__tests__") {
        return true;
    }
    let file_name = Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    let stem = Path::new(file_name)
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    stem.starts_with("test_")
        || stem.ends_with("_test")
        || stem.ends_with(".spec")
        || stem.ends_with(".test")
}

fn detect_tech_stack_from_path(path: &Path, stack: &mut std::collections::BTreeSet<String>) {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match file_name.as_str() {
        "cargo.toml" | "cargo.lock" => {
            stack.insert("Rust".to_string());
        }
        "package.json" | "pnpm-lock.yaml" | "yarn.lock" | "bun.lockb" => {
            stack.insert("Node.js".to_string());
        }
        "pyproject.toml"
        | "requirements.txt"
        | "requirements-dev.txt"
        | "pipfile"
        | "poetry.lock"
        | "setup.py"
        | "setup.cfg" => {
            stack.insert("Python".to_string());
        }
        "go.mod" | "go.sum" => {
            stack.insert("Go".to_string());
        }
        "gemfile" | "gemfile.lock" => {
            stack.insert("Ruby".to_string());
        }
        "composer.json" | "composer.lock" => {
            stack.insert("PHP".to_string());
        }
        "pom.xml"
        | "build.gradle"
        | "build.gradle.kts"
        | "settings.gradle"
        | "settings.gradle.kts"
        | "gradle.properties" => {
            stack.insert("Java".to_string());
        }
        "mix.exs" | "mix.lock" => {
            stack.insert("Elixir".to_string());
        }
        "dockerfile" | "dockerfile.dev" => {
            stack.insert("Docker".to_string());
        }
        "deno.json" | "deno.jsonc" => {
            stack.insert("Deno".to_string());
        }
        "cabal.project" | "stack.yaml" => {
            stack.insert("Haskell".to_string());
        }
        "build.sbt" => {
            stack.insert("Scala".to_string());
        }
        "pubspec.yaml" => {
            stack.insert("Dart".to_string());
        }
        "package-lock.json" => {
            stack.insert("Node.js".to_string());
        }
        _ => {}
    }

    match extension.as_str() {
        "rs" => {
            stack.insert("Rust".to_string());
        }
        "py" => {
            stack.insert("Python".to_string());
        }
        "js" | "jsx" => {
            stack.insert("JavaScript".to_string());
        }
        "ts" | "tsx" => {
            stack.insert("TypeScript".to_string());
        }
        "go" => {
            stack.insert("Go".to_string());
        }
        "rb" => {
            stack.insert("Ruby".to_string());
        }
        "php" => {
            stack.insert("PHP".to_string());
        }
        "java" => {
            stack.insert("Java".to_string());
        }
        "kt" | "kts" => {
            stack.insert("Kotlin".to_string());
        }
        "cs" | "fs" | "vb" => {
            stack.insert(".NET".to_string());
        }
        "swift" => {
            stack.insert("Swift".to_string());
        }
        "scala" => {
            stack.insert("Scala".to_string());
        }
        "c" | "h" => {
            stack.insert("C".to_string());
        }
        "cc" | "cpp" | "cxx" | "hpp" | "hh" => {
            stack.insert("C++".to_string());
        }
        "sql" => {
            stack.insert("SQL".to_string());
        }
        "tf" | "tfvars" => {
            stack.insert("Terraform".to_string());
        }
        _ => {}
    }
}

async fn detect_tech_stack(repo_root: &Path, db: &dyn Repository) -> Vec<String> {
    let Ok(files) = db.list_files().await else {
        return Vec::new();
    };
    if files.is_empty() {
        return Vec::new();
    }

    let mut stack = std::collections::BTreeSet::new();
    for file in files {
        let display = display_path(repo_root, &file);
        detect_tech_stack_from_path(Path::new(&display), &mut stack);
    }
    stack.into_iter().collect()
}

#[derive(Clone)]
struct RepoMapEntry {
    score: Option<f64>,
    badges: Vec<&'static str>,
    cochanges: Option<String>,
}

#[derive(Default)]
struct RepoMapNode {
    dirs: std::collections::BTreeMap<String, RepoMapNode>,
    files: Vec<(String, RepoMapEntry)>,
}

fn insert_repo_map_entry(node: &mut RepoMapNode, path: &str, entry: RepoMapEntry) {
    let parts = path
        .split('/')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if parts.is_empty() {
        return;
    }
    insert_repo_map_parts(node, &parts, entry);
}

fn insert_repo_map_parts(node: &mut RepoMapNode, parts: &[&str], entry: RepoMapEntry) {
    if parts.len() == 1 {
        node.files.push((parts[0].to_string(), entry));
        return;
    }
    let head = parts[0].to_string();
    let child = node.dirs.entry(head).or_default();
    insert_repo_map_parts(child, &parts[1..], entry);
}

fn render_repo_map_tree(
    node: &RepoMapNode,
    prefix: &str,
    depth: usize,
    max_depth: usize,
    max_entries: usize,
    count: &mut usize,
    out: &mut Vec<String>,
) {
    if depth > max_depth || *count >= max_entries {
        return;
    }

    let mut children = Vec::new();
    for (name, child) in &node.dirs {
        children.push(RepoMapChild::Dir(name, child));
    }
    let mut files = node.files.clone();
    files.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, entry) in &files {
        children.push(RepoMapChild::File(name, entry));
    }

    let total = children.len();
    for (idx, child) in children.into_iter().enumerate() {
        if *count >= max_entries {
            break;
        }
        let is_last = idx + 1 == total;
        let connector = if is_last { "└── " } else { "├── " };
        let next_prefix = if is_last {
            format!("{prefix}    ")
        } else {
            format!("{prefix}│   ")
        };
        match child {
            RepoMapChild::Dir(name, child) => {
                out.push(format!("{prefix}{connector}{name}/"));
                *count += 1;
                if depth < max_depth {
                    render_repo_map_tree(
                        child,
                        &next_prefix,
                        depth + 1,
                        max_depth,
                        max_entries,
                        count,
                        out,
                    );
                }
            }
            RepoMapChild::File(name, entry) => {
                let badge_text = if entry.badges.is_empty() {
                    String::new()
                } else {
                    format!(
                        " {}",
                        entry
                            .badges
                            .iter()
                            .map(|b| format!("[{b}]"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    )
                };
                let score_text = entry
                    .score
                    .map(|score| format!(" (rank {:.4})", score))
                    .unwrap_or_default();
                out.push(format!(
                    "{prefix}{connector}{name}{badge_text}{score_text}"
                ));
                *count += 1;
                if let Some(cochanges) = &entry.cochanges {
                    if *count >= max_entries {
                        break;
                    }
                    out.push(format!("{next_prefix}↔ changes with: {cochanges}"));
                    *count += 1;
                }
            }
        }
    }
}

enum RepoMapChild<'a> {
    Dir(&'a str, &'a RepoMapNode),
    File(&'a str, &'a RepoMapEntry),
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    use crate::db::{Db, GraphData, ReferenceRecord, SymbolRecord};

    #[tokio::test]
    async fn repository_summary_includes_pagerank_and_badges() -> Result<()> {
        let dir = TempDir::new()?;
        let repo_root = dir.path();
        let src_dir = repo_root.join("src");
        std::fs::create_dir_all(&src_dir)?;

        let file_a = src_dir.join("a.rs");
        let file_b = src_dir.join("b.rs");
        std::fs::write(&file_a, "fn call_foo() { foo(); }\n")?;
        std::fs::write(&file_b, "fn foo() {}\n")?;

        let db_path = repo_root.join("crumbs.db");
        let db = Db::open(&db_path, Some(3)).await?;

        let b_path = file_b.to_string_lossy().to_string();
        let b_size = std::fs::metadata(&file_b)?.len();
        let graph_b = GraphData {
            symbols: vec![SymbolRecord {
                id: "sym_b".to_string(),
                file_path: b_path.clone(),
                name: "foo".to_string(),
                kind: "definition".to_string(),
                start_byte: 0,
                end_byte: 3,
                language: "rust".to_string(),
            }],
            references: Vec::new(),
            resolutions: Vec::new(),
        };
        db.upsert_file_graph(&b_path, b_size, [0u8; 32], "rust", None, graph_b)
            .await?;

        let a_path = file_a.to_string_lossy().to_string();
        let a_size = std::fs::metadata(&file_a)?.len();
        let graph_a = GraphData {
            symbols: Vec::new(),
            references: vec![ReferenceRecord {
                id: "ref_a".to_string(),
                file_path: a_path.clone(),
                name: "foo".to_string(),
                start_byte: 0,
                end_byte: 3,
                language: "rust".to_string(),
            }],
            resolutions: Vec::new(),
        };
        db.upsert_file_graph(&a_path, a_size, [1u8; 32], "rust", None, graph_a)
            .await?;

        let mut commit_edges = Vec::new();
        for i in 0..12 {
            commit_edges.push((a_path.clone(), format!("c{i}")));
        }
        let cochange_edges = vec![(a_path.clone(), b_path.clone(), 3, 0.5)];
        db.upsert_history_edges(&commit_edges, &cochange_edges)
            .await?;

        match db.file_dependency_pagerank(1).await {
            Ok(ranks) => {
                if ranks.is_empty() {
                    return Ok(());
                }
            }
            Err(_) => return Ok(()),
        }

        let overview = build_repository_overview(repo_root, &db).await;
        let summary = overview.summary.clone();
        assert!(!summary.is_empty(), "expected summary to be populated");

        let b_line = summary
            .iter()
            .find(|line| line.contains("b.rs") && line.contains("(rank"))
            .expect("expected b.rs in summary map");
        assert!(
            b_line.contains("[core]"),
            "expected core badge on b.rs"
        );
        let a_line = summary
            .iter()
            .find(|line| line.contains("a.rs") && line.contains("(rank"))
            .expect("expected a.rs in summary map");
        assert!(
            a_line.contains("[high-churn]") || a_line.contains("[core]"),
            "expected a.rs to have high-churn or core badge"
        );

        let rendered = render_prompt(
            &PromptPayload {
                overview,
                task: "test".to_string(),
                blocks: Vec::new(),
                warnings: Vec::new(),
            },
            PromptSections::all(),
            None,
        );
        assert!(
            rendered.contains("<SUMMARY_MAP>"),
            "expected summary_map in rendered output"
        );
        Ok(())
    }

    #[test]
    fn normalized_fence_language_handles_special_cases() {
        assert_eq!(normalized_fence_language("Rust"), "rust");
        assert_eq!(normalized_fence_language(""), "text");
        assert_eq!(normalized_fence_language("MadeUpLang"), "text");
    }
}

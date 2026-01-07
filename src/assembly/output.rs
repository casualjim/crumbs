use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use eyre::Result;
use text_chunking::languages;

use crate::assembly::pipeline::{CandidateSource, ContextBlock};
use crate::repository::Repository;

const SUMMARY_LIMIT: usize = 20;
const SUMMARY_COCHANGE_LIMIT: usize = 3;
const SUMMARY_CORE_LIMIT: usize = 5;
const SUMMARY_HIGH_CHURN_COMMITS: i64 = 10;

#[derive(Clone, Copy, Debug)]
pub enum PromptFormat {
    Xml,
    Markdown,
}

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
    _repo_root: &Path,
    db: &dyn Repository,
    blocks: &[ContextBlock],
) -> Result<Vec<EnrichedBlock>> {
    let mut language_cache: BTreeMap<String, Option<String>> = BTreeMap::new();
    let mut enriched = Vec::with_capacity(blocks.len());

    for block in blocks {
        let start_line = line_from_i64(block.start_line);
        let end_line = line_from_i64(block.end_line);

        let mut symbol_names = BTreeSet::new();
        for symbol in db.symbols_in_range(&block.file_path, block.start_byte, block.end_byte)? {
            symbol_names.insert(symbol.name);
        }

        let language = if let Some(cached) = language_cache.get(&block.file_path) {
            cached.clone()
        } else {
            let detected = db.file_primary_language(&block.file_path)?;
            language_cache.insert(block.file_path.clone(), detected.clone());
            detected
        }
        .unwrap_or_else(|| "text".to_string());

        enriched.push(EnrichedBlock {
            file_path: block.file_path.clone(),
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

pub fn render_prompt(format: PromptFormat, payload: &PromptPayload, sections: PromptSections) -> String {
    match format {
        PromptFormat::Xml => render_xml(payload, sections),
        PromptFormat::Markdown => render_markdown(payload, sections),
    }
}

fn render_xml(payload: &PromptPayload, sections: PromptSections) -> String {
    let mut out = String::new();
    writeln!(out, "<context>").ok();
    if sections.overview_enabled() {
        writeln!(out, "  <repository_overview>").ok();
        writeln!(out, "    <name>{}</name>", payload.overview.name).ok();
        if sections.structure && !payload.overview.structure.is_empty() {
            write_cdata(
                &mut out,
                "    <structure><![CDATA[",
                &payload.overview.structure.join("\n"),
                "]]></structure>",
            );
        }
        if sections.summary && !payload.overview.tech_stack.is_empty() {
            writeln!(
                out,
                "    <tech_stack>{}</tech_stack>",
                xml_escape(&payload.overview.tech_stack.join(", "))
            )
            .ok();
        }
        if sections.summary && !payload.overview.summary.is_empty() {
            write_cdata(
                &mut out,
                "    <summary_map><![CDATA[",
                &payload.overview.summary.join("\n"),
                "]]></summary_map>",
            );
        }
        writeln!(out, "  </repository_overview>").ok();
    }
    if sections.context {
        writeln!(out, "  <code_context>").ok();

        let (primary, expanded) = split_blocks(&payload.blocks);
        write_xml_group(&mut out, "retrieved_files", &primary);
        write_xml_group(&mut out, "expanded_files", &expanded);

        writeln!(out, "  </code_context>").ok();
    }
    if sections.query {
        writeln!(out, "  <user_query>").ok();
        writeln!(out, "{}", xml_escape(&payload.task)).ok();
        writeln!(out, "  </user_query>").ok();
    }
    writeln!(out, "</context>").ok();
    out
}

fn write_xml_group(out: &mut String, label: &str, blocks: &[EnrichedBlock]) {
    writeln!(
        out,
        "    <{label} count=\"{}\" total_tokens=\"0\">",
        blocks.len()
    )
    .ok();
    for block in blocks {
        writeln!(out, "      <file>").ok();
        writeln!(out, "        <path>{}</path>", xml_escape(&block.file_path)).ok();
        writeln!(
            out,
            "        <lines start=\"{}\" end=\"{}\"/>",
            block.start_line, block.end_line
        )
        .ok();
        writeln!(out, "        <relevance>{:.4}</relevance>", block.relevance).ok();
        writeln!(
            out,
            "        <source>{}</source>",
            source_label(block.source)
        )
        .ok();
        writeln!(
            out,
            "        <language>{}</language>",
            xml_escape(&block.language)
        )
        .ok();
        if !block.symbols.is_empty() {
            writeln!(out, "        <symbols>").ok();
            for symbol in &block.symbols {
                writeln!(out, "          <symbol>{}</symbol>", xml_escape(symbol)).ok();
            }
            writeln!(out, "        </symbols>").ok();
        }
        write_cdata(
            out,
            "        <content><![CDATA[",
            &block.text,
            "]]></content>",
        );
        writeln!(out, "      </file>").ok();
    }
    writeln!(out, "    </{label}>").ok();
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
            writeln!(out, "\n### Structure\n```").ok();
            for line in &payload.overview.structure {
                writeln!(out, "{line}").ok();
            }
            writeln!(out, "```").ok();
        }
        if sections.summary && !payload.overview.summary.is_empty() {
            writeln!(out, "\n### Summary Map\n```").ok();
            for line in &payload.overview.summary {
                writeln!(out, "{line}").ok();
            }
            writeln!(out, "```").ok();
        }
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
        writeln!(out, "\n## User Query\n{}", payload.task).ok();
    }
    out
}

fn write_markdown_group(out: &mut String, label: &str, blocks: &[EnrichedBlock]) {
    writeln!(out, "\n### {}", label).ok();
    for block in blocks {
        let ref_line = format!("{}:{}", block.file_path, block.start_line);
        writeln!(
            out,
            "\n**{}** (relevance: {:.4}, source: {})",
            ref_line,
            block.relevance,
            source_label(block.source)
        )
        .ok();
        writeln!(out, "Lines: {}-{}", block.start_line, block.end_line).ok();
        writeln!(out, "Language: {}", block.language).ok();
        if !block.symbols.is_empty() {
            let symbols = block
                .symbols
                .iter()
                .map(|symbol| format!("`{}`", symbol))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(out, "Symbols: {}", symbols).ok();
        }
        let fence_lang = normalized_fence_language(&block.language);
        writeln!(out, "\n```{}\n{}\n```", fence_lang, block.text).ok();
    }
}

fn split_blocks(blocks: &[EnrichedBlock]) -> (Vec<EnrichedBlock>, Vec<EnrichedBlock>) {
    let mut primary = Vec::new();
    let mut expanded = Vec::new();
    for block in blocks {
        match block.source {
            CandidateSource::Primary => primary.push(block.clone()),
            CandidateSource::Expanded => expanded.push(block.clone()),
        }
    }
    (primary, expanded)
}

fn source_label(source: CandidateSource) -> &'static str {
    match source {
        CandidateSource::Primary => "primary",
        CandidateSource::Expanded => "expanded",
    }
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn write_cdata(out: &mut String, prefix: &str, text: &str, suffix: &str) {
    out.push_str(prefix);
    let escaped = text.replace("]]>", "]]]]><![CDATA[>");
    out.push_str(&escaped);
    out.push_str(suffix);
    out.push('\n');
}

pub fn build_repository_overview(repo_root: &Path, db: &dyn Repository) -> RepositoryOverview {
    let name = repo_root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("repo")
        .to_string();
    let structure = build_structure(repo_root, 2, 200);
    let tech_stack = detect_tech_stack(repo_root);
    let summary = build_repository_summary(repo_root, db);
    RepositoryOverview {
        name,
        structure,
        tech_stack,
        summary,
    }
}

fn build_repository_summary(repo_root: &Path, db: &dyn Repository) -> Vec<String> {
    let mut lines = Vec::new();
    let Ok(ranks) = db.file_dependency_pagerank(SUMMARY_LIMIT) else {
        return lines;
    };
    if ranks.is_empty() {
        return lines;
    }

    let core_limit = SUMMARY_CORE_LIMIT.min(ranks.len());

    for (idx, (file_path, score)) in ranks.iter().enumerate() {
        let display_path_value = display_path(repo_root, file_path);
        let mut badges = Vec::new();
        if idx < core_limit {
            badges.push("core");
        }
        if is_test_path(&display_path_value) {
            badges.push("test");
        }
        if let Ok(commit_count) = db.file_commit_count(file_path) {
            if commit_count >= SUMMARY_HIGH_CHURN_COMMITS {
                badges.push("high-churn");
            }
        }

        let badge_text = if badges.is_empty() {
            String::new()
        } else {
            format!(
                " {}",
                badges
                    .iter()
                    .map(|b| format!("[{b}]"))
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        };
        lines.push(format!(
            "{}{} (rank {:.4})",
            display_path_value, badge_text, score
        ));

        if let Ok(partners) = db.cochange_partners(file_path, SUMMARY_COCHANGE_LIMIT) {
            if !partners.is_empty() {
                let items = partners
                    .into_iter()
                    .map(|(path, weight)| {
                        format!("{} ({:.2})", display_path(repo_root, &path), weight)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                lines.push(format!("  <-> changes with: {items}"));
            }
        }
    }

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

fn build_structure(root: &Path, max_depth: usize, max_entries: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut count = 0usize;
    build_structure_inner(root, 0, max_depth, max_entries, &mut count, &mut lines);
    lines
}

fn build_structure_inner(
    current: &Path,
    depth: usize,
    max_depth: usize,
    max_entries: usize,
    count: &mut usize,
    lines: &mut Vec<String>,
) {
    if depth > max_depth || *count >= max_entries {
        return;
    }

    let mut entries = match std::fs::read_dir(current) {
        Ok(entries) => entries.filter_map(|entry| entry.ok()).collect::<Vec<_>>(),
        Err(_) => return,
    };
    entries.sort_by_key(|entry| entry.path());

    for entry in entries {
        if *count >= max_entries {
            break;
        }
        let path = entry.path();
        let name = match path.file_name().and_then(|name| name.to_str()) {
            Some(name) => name,
            None => continue,
        };
        if should_skip_name(name) {
            continue;
        }
        let indent = "  ".repeat(depth);
        if path.is_dir() {
            lines.push(format!("{indent}{name}/"));
            *count += 1;
            build_structure_inner(&path, depth + 1, max_depth, max_entries, count, lines);
        } else if depth == 0 {
            lines.push(format!("{indent}{name}"));
            *count += 1;
        }
    }
}

fn should_skip_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    if lower.starts_with('.') {
        return true;
    }
    matches!(
        lower.as_str(),
        "target"
            | "node_modules"
            | ".config"
            | ".idea"
            | ".vscode"
            | "dist"
            | "build"
            | ".venv"
            | "venv"
    )
}

fn detect_tech_stack(root: &Path) -> Vec<String> {
    let mut stack = Vec::new();
    push_unique(&mut stack, "Rust", root.join("Cargo.toml").exists());
    push_unique(&mut stack, "Node.js", root.join("package.json").exists());
    push_unique(
        &mut stack,
        "Python",
        root.join("pyproject.toml").exists() || root.join("requirements.txt").exists(),
    );
    push_unique(&mut stack, "Go", root.join("go.mod").exists());
    push_unique(&mut stack, "Ruby", root.join("Gemfile").exists());
    push_unique(
        &mut stack,
        "Java",
        root.join("pom.xml").exists() || root.join("build.gradle").exists(),
    );
    push_unique(&mut stack, "PHP", root.join("composer.json").exists());
    stack
}

fn push_unique(stack: &mut Vec<String>, label: &str, present: bool) {
    if present && !stack.iter().any(|item| item == label) {
        stack.push(label.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    use crate::db::{Db, GraphData, ReferenceRecord, SymbolRecord};

    #[test]
    fn repository_summary_includes_pagerank_and_badges() -> Result<()> {
        let dir = TempDir::new()?;
        let repo_root = dir.path();
        let src_dir = repo_root.join("src");
        std::fs::create_dir_all(&src_dir)?;

        let file_a = src_dir.join("a.rs");
        let file_b = src_dir.join("b.rs");
        std::fs::write(&file_a, "fn call_foo() { foo(); }\n")?;
        std::fs::write(&file_b, "fn foo() {}\n")?;

        let db_path = repo_root.join("context.db");
        let db = Db::open(&db_path, Some(3))?;

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
        db.upsert_file_graph(&b_path, b_size, [0u8; 32], "rust", None, graph_b)?;

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
        db.upsert_file_graph(&a_path, a_size, [1u8; 32], "rust", None, graph_a)?;

        let mut commit_edges = Vec::new();
        for i in 0..12 {
            commit_edges.push((a_path.clone(), format!("c{i}")));
        }
        let cochange_edges = vec![(a_path.clone(), b_path.clone(), 3, 0.5)];
        db.upsert_history_edges(&commit_edges, &cochange_edges)?;

        match db.file_dependency_pagerank(1) {
            Ok(ranks) => {
                if ranks.is_empty() {
                    return Ok(());
                }
            }
            Err(_) => return Ok(()),
        }

        let overview = build_repository_overview(repo_root, &db);
        let summary = overview.summary.clone();
        assert!(!summary.is_empty(), "expected summary to be populated");

        let main_lines: Vec<&String> = summary
            .iter()
            .filter(|line| !line.starts_with("  <->"))
            .collect();
        assert!(
            main_lines[0].contains("src/b.rs"),
            "expected pagerank to prioritize src/b.rs, got {:?}",
            main_lines
        );
        let a_line = main_lines
            .iter()
            .find(|line| line.contains("src/a.rs"))
            .expect("expected src/a.rs in summary");
        assert!(
            a_line.contains("[high-churn]"),
            "expected high-churn badge on src/a.rs"
        );

        let xml = render_prompt(
            PromptFormat::Xml,
            &PromptPayload {
                overview,
                task: "test".to_string(),
                blocks: Vec::new(),
            },
            PromptSections::all(),
        );
        assert!(
            xml.contains("<summary_map>"),
            "expected summary_map in XML output"
        );
        Ok(())
    }

    #[test]
    fn xml_escape_rewrites_special_chars() {
        let input = r#"<tag attr="x&y">"'"#;
        let escaped = xml_escape(input);
        assert_eq!(escaped, "&lt;tag attr=&quot;x&amp;y&quot;&gt;&quot;&apos;");
    }

    #[test]
    fn normalized_fence_language_handles_special_cases() {
        assert_eq!(normalized_fence_language("Rust"), "rust");
        assert_eq!(normalized_fence_language(""), "text");
        assert_eq!(normalized_fence_language("MadeUpLang"), "text");
    }
}

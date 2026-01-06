use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use eyre::Result;
use text_chunking::languages;

use crate::assembly::pipeline::{CandidateSource, ContextBlock};
use crate::repository::Repository;

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

#[derive(Clone)]
pub struct RepositoryOverview {
    pub name: String,
    pub structure: Vec<String>,
    pub tech_stack: Vec<String>,
}

fn line_from_i64(value: i64) -> usize {
    if value <= 0 {
        1
    } else {
        value as usize
    }
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

pub fn render_prompt(format: PromptFormat, payload: &PromptPayload) -> String {
    match format {
        PromptFormat::Xml => render_xml(payload),
        PromptFormat::Markdown => render_markdown(payload),
    }
}

fn render_xml(payload: &PromptPayload) -> String {
    let mut out = String::new();
    writeln!(out, "<context>").ok();
    writeln!(out, "  <repository_overview>").ok();
    writeln!(out, "    <name>{}</name>", payload.overview.name).ok();
    if !payload.overview.structure.is_empty() {
        write_cdata(
            &mut out,
            "    <structure><![CDATA[",
            &payload.overview.structure.join("\n"),
            "]]></structure>",
        );
    }
    if !payload.overview.tech_stack.is_empty() {
        writeln!(
            out,
            "    <tech_stack>{}</tech_stack>",
            xml_escape(&payload.overview.tech_stack.join(", "))
        )
        .ok();
    }
    writeln!(out, "  </repository_overview>").ok();
    writeln!(out, "  <code_context>").ok();

    let (primary, expanded) = split_blocks(&payload.blocks);
    write_xml_group(&mut out, "retrieved_files", &primary);
    write_xml_group(&mut out, "expanded_files", &expanded);

    writeln!(out, "  </code_context>").ok();
    writeln!(out, "  <user_query>").ok();
    writeln!(out, "{}", xml_escape(&payload.task)).ok();
    writeln!(out, "  </user_query>").ok();
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

fn render_markdown(payload: &PromptPayload) -> String {
    let mut out = String::new();
    writeln!(out, "## Repository: {}", payload.overview.name).ok();
    if !payload.overview.tech_stack.is_empty() {
        writeln!(
            out,
            "Tech stack: {}",
            payload.overview.tech_stack.join(", ")
        )
        .ok();
    }
    if !payload.overview.structure.is_empty() {
        writeln!(out, "\n### Structure\n```").ok();
        for line in &payload.overview.structure {
            writeln!(out, "{line}").ok();
        }
        writeln!(out, "```").ok();
    }
    writeln!(
        out,
        "\n## Retrieved Context ({} blocks)",
        payload.blocks.len()
    )
    .ok();

    let (primary, expanded) = split_blocks(&payload.blocks);
    write_markdown_group(&mut out, "Primary Results", &primary);
    write_markdown_group(&mut out, "Expanded Context", &expanded);

    writeln!(out, "\n## User Query\n{}", payload.task).ok();
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

pub fn build_repository_overview(repo_root: &Path) -> RepositoryOverview {
    let name = repo_root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("repo")
        .to_string();
    let structure = build_structure(repo_root, 2, 200);
    let tech_stack = detect_tech_stack(repo_root);
    RepositoryOverview {
        name,
        structure,
        tech_stack,
    }
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

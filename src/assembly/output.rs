use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::io::Cursor;
use std::path::Path;

use eyre::{Result, WrapErr};
use text_chunking::languages::{self, PeekableReader, detect};

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
    pub repo_name: String,
    pub task: String,
    pub blocks: Vec<EnrichedBlock>,
}

struct FileContext {
    bytes: Vec<u8>,
    line_index: LineIndex,
    language: Option<String>,
}

struct LineIndex {
    line_starts: Vec<usize>,
}

impl LineIndex {
    fn new(bytes: &[u8]) -> Self {
        let mut line_starts = Vec::with_capacity(bytes.len() / 24 + 1);
        line_starts.push(0);
        for (idx, byte) in bytes.iter().enumerate() {
            if *byte == b'\n' {
                line_starts.push(idx + 1);
            }
        }
        Self { line_starts }
    }

    fn line_for_byte(&self, byte: usize) -> usize {
        if self.line_starts.is_empty() {
            return 1;
        }
        match self.line_starts.binary_search(&byte) {
            Ok(index) => index + 1,
            Err(index) => index.max(1),
        }
    }
}

fn clamp_byte(byte: i64, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let clamped = if byte < 0 { 0 } else { byte as usize };
    clamped.min(len - 1)
}

async fn detect_language(path: &Path, bytes: &[u8]) -> Option<String> {
    let cursor = Cursor::new(bytes.to_vec());
    let peekable = PeekableReader::new(cursor, 51200);
    let (detection, _) = detect(path, peekable).await.ok()?;
    detection.map(|detection| detection.language().to_string())
}

fn fallback_language(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
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

async fn load_file_context(repo_root: &Path, file_path: &str) -> Result<FileContext> {
    let full_path = repo_root.join(file_path);
    let bytes = tokio::fs::read(&full_path)
        .await
        .wrap_err_with(|| format!("failed to read {}", full_path.display()))?;
    let line_index = LineIndex::new(&bytes);
    let language = detect_language(&full_path, &bytes)
        .await
        .or_else(|| fallback_language(&full_path));
    Ok(FileContext {
        bytes,
        line_index,
        language,
    })
}

pub async fn enrich_blocks(
    repo_root: &Path,
    db: &dyn Repository,
    blocks: &[ContextBlock],
) -> Result<Vec<EnrichedBlock>> {
    let mut contexts: BTreeMap<String, FileContext> = BTreeMap::new();
    let mut enriched = Vec::with_capacity(blocks.len());

    for block in blocks {
        let ctx = if let Some(ctx) = contexts.get(&block.file_path) {
            ctx
        } else {
            let loaded = load_file_context(repo_root, &block.file_path).await?;
            contexts.insert(block.file_path.clone(), loaded);
            contexts.get(&block.file_path).expect("inserted context")
        };

        let start_byte = clamp_byte(block.start_byte, ctx.bytes.len());
        let end_byte = if block.end_byte > 0 {
            clamp_byte(block.end_byte - 1, ctx.bytes.len())
        } else {
            start_byte
        };

        let start_line = ctx.line_index.line_for_byte(start_byte);
        let end_line = ctx.line_index.line_for_byte(end_byte);

        let mut symbol_names = BTreeSet::new();
        for symbol in db.symbols_in_range(&block.file_path, block.start_byte, block.end_byte)? {
            symbol_names.insert(symbol.name);
        }

        let language = ctx.language.clone().unwrap_or_else(|| "text".to_string());

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
    writeln!(out, "    <name>{}</name>", payload.repo_name).ok();
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
    writeln!(out, "## Repository: {}", payload.repo_name).ok();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_index_maps_bytes_to_lines() {
        let bytes = b"first\nsecond\nthird";
        let index = LineIndex::new(bytes);
        assert_eq!(index.line_for_byte(0), 1);
        assert_eq!(index.line_for_byte(6), 2);
        assert_eq!(index.line_for_byte(bytes.len() - 1), 3);
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

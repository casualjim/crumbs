use std::collections::{HashMap, HashSet};
use std::path::Path;

use cupido::collector::config::{Collect, Config as CupidoConfig, get_collector};
use eyre::Result;
use syntastica_queries::{
    ASM_LOCALS_CRATES_IO, BASH_LOCALS_CRATES_IO, BIBTEX_LOCALS_CRATES_IO, C_LOCALS_CRATES_IO,
    CLOJURE_LOCALS_CRATES_IO, CMAKE_LOCALS_CRATES_IO, COMMENT_LOCALS_CRATES_IO,
    CPP_LOCALS_CRATES_IO, CSS_LOCALS_CRATES_IO, C_SHARP_LOCALS_CRATES_IO, DART_LOCALS_CRATES_IO,
    DIFF_LOCALS_CRATES_IO, DOCKERFILE_LOCALS_CRATES_IO, EBNF_LOCALS_CRATES_IO,
    EJS_LOCALS_CRATES_IO, ELIXIR_LOCALS_CRATES_IO, ERB_LOCALS_CRATES_IO, FISH_LOCALS_CRATES_IO,
    GLEAM_LOCALS_CRATES_IO, GO_LOCALS_CRATES_IO, HASKELL_LOCALS_CRATES_IO,
    HEXDUMP_LOCALS_CRATES_IO, HTML_LOCALS_CRATES_IO, JAVA_LOCALS_CRATES_IO,
    JAVASCRIPT_LOCALS_CRATES_IO, JSON5_LOCALS_CRATES_IO, JSONC_LOCALS_CRATES_IO,
    JSON_LOCALS_CRATES_IO, JSDOC_LOCALS_CRATES_IO, JULIA_LOCALS_CRATES_IO,
    KOTLIN_LOCALS_CRATES_IO, LALRPOP_LOCALS_CRATES_IO, LATEX_LOCALS_CRATES_IO,
    LLVM_LOCALS_CRATES_IO, LUA_LOCALS_CRATES_IO, LUAP_LOCALS_CRATES_IO, MAKE_LOCALS_CRATES_IO,
    MARKDOWN_INLINE_LOCALS_CRATES_IO, MARKDOWN_LOCALS_CRATES_IO, NIX_LOCALS_CRATES_IO,
    OCAML_INTERFACE_LOCALS_CRATES_IO, OCAML_LOCALS_CRATES_IO, PHP_LOCALS_CRATES_IO,
    PHP_ONLY_LOCALS_CRATES_IO, PRINTF_LOCALS_CRATES_IO, PYTHON_LOCALS_CRATES_IO,
    QL_LOCALS_CRATES_IO, REGEX_LOCALS_CRATES_IO, RUBY_LOCALS_CRATES_IO, RUSH_LOCALS_CRATES_IO,
    RUST_LOCALS_CRATES_IO, SCALA_LOCALS_CRATES_IO, SCSS_LOCALS_CRATES_IO,
    SQL_LOCALS_CRATES_IO, SWIFT_LOCALS_CRATES_IO, TOML_LOCALS_CRATES_IO,
    TSX_LOCALS_CRATES_IO, TYPESCRIPT_LOCALS_CRATES_IO, TYPST_LOCALS_CRATES_IO,
    URSA_LOCALS_CRATES_IO, VERILOG_LOCALS_CRATES_IO, WAT_LOCALS_CRATES_IO,
    YAML_LOCALS_CRATES_IO, ZIG_LOCALS_CRATES_IO,
};
use tracing::{info, warn};
use tree_sitter::{Query, QueryCursor, StreamingIterator};
use uuid::Uuid;

use crate::db::{GraphData, ReferenceRecord, SymbolRecord};
use crate::repository::Repository;

pub struct HistoryConfig {
    pub depth: u32,
    pub commit_size_limit_ratio: f32,
    pub multi_parents: bool,
    pub issue_regex: String,
    pub commit_exclude_regex: Option<String>,
    pub author_exclude_regex: Option<String>,
    pub path_specs: Vec<String>,
}

pub(crate) fn index_history(
    db: &dyn Repository,
    repo_path: &Path,
    config: &HistoryConfig,
) -> Result<()> {
    if !repo_path.join(".git").exists() {
        warn!(
            "history indexing skipped; no git repository at {}",
            repo_path.display()
        );
        return Ok(());
    }

    let known_files = db.list_files()?;
    if known_files.is_empty() {
        return Ok(());
    }
    let known_set: HashSet<String> = known_files.into_iter().collect();

    let mut conf = CupidoConfig::default();
    conf.repo_path = repo_path.to_string_lossy().to_string();
    conf.depth = config.depth;
    conf.multi_parents = config.multi_parents;
    conf.issue_regex = config.issue_regex.clone();
    conf.commit_exclude_regex = config.commit_exclude_regex.clone();
    conf.author_exclude_regex = config.author_exclude_regex.clone();
    conf.path_specs = config.path_specs.clone();
    conf.progress = false;

    let collector = get_collector();
    let graph = collector.walk(conf);

    let file_count = known_set.len().max(1) as f32;
    let max_files_per_commit = if config.commit_size_limit_ratio >= 1.0 {
        usize::MAX
    } else {
        (file_count * config.commit_size_limit_ratio).ceil() as usize
    }
    .max(1);

    let mut file_commit_edges: HashSet<(String, String)> = HashSet::new();
    let mut cochange_map: HashMap<(String, String), (u64, f64)> = HashMap::new();

    for commit_id in graph.commits() {
        let files = match graph.commit_related_files(&commit_id) {
            Ok(files) => files,
            Err(_) => continue,
        };
        let total_files = files.len();
        if total_files == 0 || total_files > max_files_per_commit {
            continue;
        }

        let weight = 1.0 / total_files as f64;
        let mut filtered: Vec<String> = files
            .into_iter()
            .filter(|file| known_set.contains(file))
            .collect();
        if filtered.is_empty() {
            continue;
        }

        filtered.sort();
        filtered.dedup();

        for file in &filtered {
            file_commit_edges.insert((file.clone(), commit_id.clone()));
        }

        if filtered.len() < 2 {
            continue;
        }

        for i in 0..filtered.len() {
            for j in (i + 1)..filtered.len() {
                let (left, right) = (&filtered[i], &filtered[j]);
                let key = if left <= right {
                    (left.clone(), right.clone())
                } else {
                    (right.clone(), left.clone())
                };
                let entry = cochange_map.entry(key).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += weight;
            }
        }
    }

    let mut commit_edges: Vec<(String, String)> = file_commit_edges.into_iter().collect();
    commit_edges.sort();

    let mut cochange_edges: Vec<(String, String, i64, f64)> = cochange_map
        .into_iter()
        .map(|((src, dst), (count, weight))| (src, dst, count as i64, weight))
        .collect();
    cochange_edges.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    db.replace_history_edges(&commit_edges, &cochange_edges)?;
    info!(
        "history indexing complete: commits={}, cochanges={}",
        commit_edges.len(),
        cochange_edges.len()
    );
    Ok(())
}

pub(crate) fn extract_graph_from_tree(
    language: &str,
    ts_language: tree_sitter::Language,
    tree: &tree_sitter::Tree,
    source: &str,
) -> Result<Option<GraphData>> {
    let Some(query_source) = graph_query_for_language(language) else {
        return Ok(None);
    };

    extract_graph_from_tree_inner(ts_language, tree, source, query_source)
}

fn extract_graph_from_tree_inner(
    ts_language: tree_sitter::Language,
    tree: &tree_sitter::Tree,
    source: &str,
    query_source: &'static str,
) -> Result<Option<GraphData>> {
    if query_source.trim().is_empty() {
        return Ok(None);
    }

    let query = Query::new(&ts_language, query_source)?;
    let capture_names = query.capture_names();
    let mut cursor = QueryCursor::new();

    let mut symbols = Vec::new();
    let mut references = Vec::new();
    let mut seen = HashSet::new();

    let mut matches = cursor.matches(&query, tree.root_node(), source.as_bytes());
    while let Some(m) = matches.next() {
        for capture in m.captures {
            let name = capture_names
                .get(capture.index as usize)
                .map(|name| name.as_ref())
                .unwrap_or("");
            if !(name.starts_with("local.definition") || name.starts_with("local.reference")) {
                continue;
            }

            let node = capture.node;
            let start_byte = node.start_byte();
            let end_byte = node.end_byte();
            let key = (start_byte, end_byte, name.to_string());
            if !seen.insert(key) {
                continue;
            }

            let text = source
                .get(start_byte..end_byte)
                .unwrap_or("")
                .trim()
                .to_string();
            if text.is_empty() {
                continue;
            }

            if name.starts_with("local.definition") {
                symbols.push(SymbolRecord {
                    id: Uuid::now_v7().to_string(),
                    file_path: String::new(),
                    name: text,
                    kind: "definition".to_string(),
                    start_byte,
                    end_byte,
                    language: String::new(),
                });
            } else {
                references.push(ReferenceRecord {
                    id: Uuid::now_v7().to_string(),
                    file_path: String::new(),
                    name: text,
                    start_byte,
                    end_byte,
                    language: String::new(),
                });
            }
        }
    }

    Ok(Some(GraphData {
        symbols,
        references,
        resolutions: Vec::new(),
    }))
}

fn graph_query_for_language(language: &str) -> Option<&'static str> {
    let lang = language.to_ascii_lowercase();
    match lang.as_str() {
        "asm" => Some(ASM_LOCALS_CRATES_IO),
        "bash" => Some(BASH_LOCALS_CRATES_IO),
        "bibtex" => Some(BIBTEX_LOCALS_CRATES_IO),
        "c" => Some(C_LOCALS_CRATES_IO),
        "csharp" | "c_sharp" | "c#" => Some(C_SHARP_LOCALS_CRATES_IO),
        "clojure" => Some(CLOJURE_LOCALS_CRATES_IO),
        "cmake" => Some(CMAKE_LOCALS_CRATES_IO),
        "comment" => Some(COMMENT_LOCALS_CRATES_IO),
        "cpp" | "c++" => Some(CPP_LOCALS_CRATES_IO),
        "css" => Some(CSS_LOCALS_CRATES_IO),
        "dart" => Some(DART_LOCALS_CRATES_IO),
        "diff" => Some(DIFF_LOCALS_CRATES_IO),
        "dockerfile" => Some(DOCKERFILE_LOCALS_CRATES_IO),
        "ebnf" => Some(EBNF_LOCALS_CRATES_IO),
        "ejs" => Some(EJS_LOCALS_CRATES_IO),
        "elixir" => Some(ELIXIR_LOCALS_CRATES_IO),
        "erb" => Some(ERB_LOCALS_CRATES_IO),
        "fish" => Some(FISH_LOCALS_CRATES_IO),
        "gleam" => Some(GLEAM_LOCALS_CRATES_IO),
        "rust" => Some(RUST_LOCALS_CRATES_IO),
        "python" => Some(PYTHON_LOCALS_CRATES_IO),
        "go" => Some(GO_LOCALS_CRATES_IO),
        "haskell" => Some(HASKELL_LOCALS_CRATES_IO),
        "hexdump" => Some(HEXDUMP_LOCALS_CRATES_IO),
        "html" => Some(HTML_LOCALS_CRATES_IO),
        "java" => Some(JAVA_LOCALS_CRATES_IO),
        "javascript" | "js" | "javascriptreact" | "jsx" => Some(JAVASCRIPT_LOCALS_CRATES_IO),
        "jsdoc" => Some(JSDOC_LOCALS_CRATES_IO),
        "json" => Some(JSON_LOCALS_CRATES_IO),
        "json5" => Some(JSON5_LOCALS_CRATES_IO),
        "jsonc" => Some(JSONC_LOCALS_CRATES_IO),
        "julia" => Some(JULIA_LOCALS_CRATES_IO),
        "kotlin" => Some(KOTLIN_LOCALS_CRATES_IO),
        "lalrpop" => Some(LALRPOP_LOCALS_CRATES_IO),
        "latex" => Some(LATEX_LOCALS_CRATES_IO),
        "llvm" => Some(LLVM_LOCALS_CRATES_IO),
        "lua" => Some(LUA_LOCALS_CRATES_IO),
        "luap" => Some(LUAP_LOCALS_CRATES_IO),
        "make" => Some(MAKE_LOCALS_CRATES_IO),
        "markdown" => Some(MARKDOWN_LOCALS_CRATES_IO),
        "markdown_inline" | "markdown-inline" => Some(MARKDOWN_INLINE_LOCALS_CRATES_IO),
        "nix" => Some(NIX_LOCALS_CRATES_IO),
        "ocaml" => Some(OCAML_LOCALS_CRATES_IO),
        "ocaml_interface" | "ocaml-interface" => Some(OCAML_INTERFACE_LOCALS_CRATES_IO),
        "php" => Some(PHP_LOCALS_CRATES_IO),
        "php_only" | "php-only" => Some(PHP_ONLY_LOCALS_CRATES_IO),
        "printf" => Some(PRINTF_LOCALS_CRATES_IO),
        "ql" => Some(QL_LOCALS_CRATES_IO),
        "regex" => Some(REGEX_LOCALS_CRATES_IO),
        "ruby" => Some(RUBY_LOCALS_CRATES_IO),
        "rush" => Some(RUSH_LOCALS_CRATES_IO),
        "scala" => Some(SCALA_LOCALS_CRATES_IO),
        "scss" => Some(SCSS_LOCALS_CRATES_IO),
        "sql" => Some(SQL_LOCALS_CRATES_IO),
        "swift" => Some(SWIFT_LOCALS_CRATES_IO),
        "toml" => Some(TOML_LOCALS_CRATES_IO),
        "tsx" | "typescriptreact" => Some(TSX_LOCALS_CRATES_IO),
        "typescript" | "ts" => Some(TYPESCRIPT_LOCALS_CRATES_IO),
        "typst" => Some(TYPST_LOCALS_CRATES_IO),
        "ursa" => Some(URSA_LOCALS_CRATES_IO),
        "verilog" => Some(VERILOG_LOCALS_CRATES_IO),
        "wat" => Some(WAT_LOCALS_CRATES_IO),
        "yaml" | "yml" => Some(YAML_LOCALS_CRATES_IO),
        "zig" => Some(ZIG_LOCALS_CRATES_IO),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::HistoryConfig;
    use tempfile::TempDir;
    use text_chunking::Tokenizer;

    use crate::db::Db;
    use crate::indexer::{Indexer, IndexerConfig};
    use crate::test_support::{load_test_embedder, write_fixture_repo};

    #[tokio::test]
    async fn graph_build_populates_symbols_and_references() -> eyre::Result<()> {
        let (embedder, embedding_dim) = load_test_embedder()?;
        let dir = TempDir::new()?;
        write_fixture_repo(dir.path())?;

        let db_path = dir.path().join("context.duckdb");
        let db = Db::open(&db_path, Some(embedding_dim))?;

        let config = IndexerConfig {
            repo_path: dir.path().to_path_buf(),
            max_chunk_size: 1500,
            overlap_percentage: 0.2,
            tokenizer: Tokenizer::Characters,
            max_parallel: 4,
            max_file_size: Some(5 * 1024 * 1024),
            large_file_threads: 2,
            history: HistoryConfig {
                depth: 10240,
                commit_size_limit_ratio: 1.0,
                multi_parents: false,
                issue_regex: "(#\\d+)".to_string(),
                commit_exclude_regex: None,
                author_exclude_regex: None,
                path_specs: Vec::new(),
            },
        };
        let indexer = Indexer::new(&db, embedder, config);
        indexer.index().await?;

        let conn = duckdb::Connection::open(&db_path)?;
        let symbols: i64 = conn.query_row("SELECT COUNT(*) FROM symbols", [], |row| row.get(0))?;
        let references: i64 =
            conn.query_row("SELECT COUNT(*) FROM symbol_references", [], |row| {
                row.get(0)
            })?;

        assert!(symbols > 0, "expected symbols to be populated");
        assert!(references > 0, "expected references to be populated");
        Ok(())
    }
}

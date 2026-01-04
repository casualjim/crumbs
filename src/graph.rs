use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::path::{Path, PathBuf};

use cupido::collector::config::{Collect, Config as CupidoConfig, get_collector};
use eyre::{Result, eyre};
use futures::StreamExt;
use syntastica_queries::{
    GO_LOCALS_CRATES_IO, JAVASCRIPT_LOCALS_CRATES_IO, PYTHON_LOCALS_CRATES_IO,
    RUST_LOCALS_CRATES_IO, TYPESCRIPT_LOCALS_CRATES_IO,
};
use text_chunking::languages::{PeekableReader, detect, get_language};
use text_chunking::{Chunk, Tokenizer, WalkOptions, walk_project};
use tracing::{info, warn};
use tree_sitter::{Parser, Query, QueryCursor, StreamingIterator};
use uuid::Uuid;

use crate::db::{GraphData, ReferenceRecord, SymbolRecord};
use crate::repository::Repository;

pub struct GraphConfig {
    pub repo_path: PathBuf,
    pub max_chunk_size: usize,
    pub overlap_percentage: f32,
    pub tokenizer: Tokenizer,
    pub max_parallel: usize,
    pub max_file_size: Option<u64>,
    pub large_file_threads: usize,
    pub history_depth: u32,
    pub history_commit_size_limit_ratio: f32,
    pub history_multi_parents: bool,
    pub history_issue_regex: String,
    pub history_commit_exclude_regex: Option<String>,
    pub history_author_exclude_regex: Option<String>,
    pub history_path_specs: Vec<String>,
}

pub struct GraphIndexer {
    db: Box<dyn Repository>,
    config: GraphConfig,
}

impl GraphIndexer {
    pub fn new<R: Repository + 'static>(db: R, config: GraphConfig) -> Self {
        Self {
            db: Box::new(db),
            config,
        }
    }

    pub async fn index(self) -> Result<()> {
        let existing_hashes = self.db.load_existing_hashes()?;
        let options = WalkOptions {
            max_chunk_size: self.config.max_chunk_size,
            tokenizer: self.config.tokenizer.clone(),
            overlap_percentage: self.config.overlap_percentage,
            max_parallel: self.config.max_parallel,
            max_file_size: self.config.max_file_size,
            large_file_threads: self.config.large_file_threads,
            existing_hashes,
            cancel_token: None,
        };

        let mut stream = walk_project(&self.config.repo_path, options);
        while let Some(item) = stream.next().await {
            let project_chunk = item?;
            match project_chunk.chunk {
                Chunk::Delete { file_path } => {
                    self.db.delete_file(&file_path)?;
                }
                Chunk::EndOfFile {
                    file_path,
                    content,
                    content_hash,
                    ..
                } => {
                    let Some(source) = content else {
                        return Err(eyre!("missing content for {}", file_path));
                    };
                    let Some(hash) = content_hash else {
                        return Err(eyre!("missing content hash for {}", file_path));
                    };

                    if let Some(language) = detect_language(&file_path, &source).await?
                        && let Some(graph) = extract_graph(&language, &source)?
                    {
                        self.db.replace_file_graph(
                            &file_path,
                            project_chunk.file_size,
                            hash,
                            &language,
                            graph,
                        )?;
                    }
                }
                _ => {}
            }
        }

        index_history(self.db.as_ref(), &self.config)?;

        Ok(())
    }
}

fn index_history(db: &dyn Repository, config: &GraphConfig) -> Result<()> {
    if !repo_has_git_dir(&config.repo_path) {
        warn!(
            "history indexing skipped; no git repository at {}",
            config.repo_path.display()
        );
        return Ok(());
    }

    let known_files = db.list_files()?;
    if known_files.is_empty() {
        return Ok(());
    }
    let known_set: HashSet<String> = known_files.into_iter().collect();

    let mut conf = CupidoConfig::default();
    conf.repo_path = config.repo_path.to_string_lossy().to_string();
    conf.depth = config.history_depth;
    conf.multi_parents = config.history_multi_parents;
    conf.issue_regex = config.history_issue_regex.clone();
    conf.commit_exclude_regex = config.history_commit_exclude_regex.clone();
    conf.author_exclude_regex = config.history_author_exclude_regex.clone();
    conf.path_specs = config.history_path_specs.clone();
    conf.progress = false;

    let collector = get_collector();
    let graph = collector.walk(conf);

    let file_count = known_set.len().max(1) as f32;
    let max_files_per_commit = if config.history_commit_size_limit_ratio >= 1.0 {
        usize::MAX
    } else {
        (file_count * config.history_commit_size_limit_ratio).ceil() as usize
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

fn repo_has_git_dir(path: &Path) -> bool {
    path.join(".git").exists()
}

fn extract_graph(language: &str, source: &str) -> Result<Option<GraphData>> {
    let query_source = match language {
        "rust" => RUST_LOCALS_CRATES_IO,
        "python" => PYTHON_LOCALS_CRATES_IO,
        "go" => GO_LOCALS_CRATES_IO,
        "javascript" => JAVASCRIPT_LOCALS_CRATES_IO,
        "typescript" => TYPESCRIPT_LOCALS_CRATES_IO,
        _ => return Ok(None),
    };

    if query_source.trim().is_empty() {
        return Ok(None);
    }

    let language_fn = match get_language(language) {
        Some(lang) => lang,
        None => return Ok(None),
    };
    let ts_language: tree_sitter::Language = language_fn.into();

    let mut parser = Parser::new();
    parser.set_language(&ts_language)?;
    let tree = parser
        .parse(source, None)
        .ok_or_else(|| eyre!("failed to parse source for {}", language))?;

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

async fn detect_language(file_path: &str, source: &str) -> Result<Option<String>> {
    let cursor = Cursor::new(source.as_bytes().to_vec());
    let peekable = PeekableReader::new(cursor, 51200);
    let (detection, _reader) = detect(std::path::Path::new(file_path), peekable)
        .await
        .map_err(|(err, _reader)| err)?;

    let Some(detection) = detection else {
        return Ok(None);
    };

    let raw = detection.language().to_ascii_lowercase();
    let normalized = match raw.as_str() {
        "typescriptreact" | "tsx" => "typescript".to_string(),
        "javascriptreact" | "jsx" => "javascript".to_string(),
        other => other.to_string(),
    };
    Ok(Some(normalized))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use text_chunking::Tokenizer;

    use crate::db::Db;
    use crate::test_support::write_fixture_repo;

    #[tokio::test]
    async fn graph_build_populates_symbols_and_references() -> Result<()> {
        let dir = TempDir::new()?;
        write_fixture_repo(dir.path())?;

        let db_path = dir.path().join("context.duckdb");
        let db = Db::open(&db_path, Some(2))?;

        let config = GraphConfig {
            repo_path: dir.path().to_path_buf(),
            max_chunk_size: 1500,
            overlap_percentage: 0.2,
            tokenizer: Tokenizer::Characters,
            max_parallel: 4,
            max_file_size: Some(5 * 1024 * 1024),
            large_file_threads: 2,
            history_depth: 10240,
            history_commit_size_limit_ratio: 1.0,
            history_multi_parents: false,
            history_issue_regex: "(#\\d+)".to_string(),
            history_commit_exclude_regex: None,
            history_author_exclude_regex: None,
            history_path_specs: Vec::new(),
        };
        let indexer = GraphIndexer::new(db, config);
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

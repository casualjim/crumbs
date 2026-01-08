use std::collections::{HashMap, HashSet};
use std::path::Path;

use cupido::collector::config::{get_collector, Collect, Config as CupidoConfig};
use eyre::Result;
use tracing::{info, warn};

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

pub(crate) async fn index_history(
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

    let known_files = db.list_files().await?;
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

    db.upsert_history_edges(&commit_edges, &cochange_edges).await?;
    info!(
        "history indexing complete: commits={}, cochanges={}",
        commit_edges.len(),
        cochange_edges.len()
    );
    Ok(())
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

        let db_path = dir.path().join("context.db");
        let db = Db::open(&db_path, Some(embedding_dim)).await?;

        let config = IndexerConfig {
            repo_path: dir.path().to_path_buf(),
            max_chunk_size: 1500,
            overlap_percentage: 0.2,
            tokenizer: Tokenizer::Tiktoken("cl100k_base".to_string()),
            max_parallel: 4,
            max_file_size: Some(5 * 1024 * 1024),
            large_file_threads: 2,
            max_batch_size: 16,
            max_tokens: 8_192,
            embedding_workers: 1,
            cancel_token: None,
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

        let test_db = libsql::Builder::new_local(&db_path).build().await?;
        let conn = test_db.connect()?;
        let mut rows = conn.query("SELECT COUNT(*) FROM symbols", ()).await?;
        let row = rows.next().await?.ok_or_else(|| eyre::eyre!("missing row"))?;
        let symbols: i64 = row.get(0)?;
        let mut rows = conn
            .query("SELECT COUNT(*) FROM symbol_references", ())
            .await?;
        let row = rows.next().await?.ok_or_else(|| eyre::eyre!("missing row"))?;
        let references: i64 = row.get(0)?;

        assert!(symbols > 0, "expected symbols to be populated");
        assert!(references > 0, "expected references to be populated");
        Ok(())
    }
}

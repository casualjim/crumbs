mod batcher;
mod embedder;
mod observer;
mod processor;
mod state;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use eyre::Result;
use futures::StreamExt;
use text_chunking::{Tokenizer, WalkOptions, walk_project_with_observer};

use crate::embedding::EmbeddingProvider;
use crate::graph::{HistoryConfig, index_history};
use crate::repository::Repository;

use self::observer::{GraphObserver, ObservedGraph};
use self::processor::IndexProcessor;

pub struct IndexerConfig {
    pub repo_path: PathBuf,
    pub max_chunk_size: usize,
    pub overlap_percentage: f32,
    pub tokenizer: Tokenizer,
    pub max_parallel: usize,
    pub max_file_size: Option<u64>,
    pub large_file_threads: usize,
    pub stream_batch_size: usize,
    pub max_batch_size: usize,
    pub max_tokens: usize,
    pub history: HistoryConfig,
}

pub struct Indexer<'a> {
    db: &'a dyn Repository,
    embedder: Box<dyn EmbeddingProvider>,
    config: IndexerConfig,
}

impl<'a> Indexer<'a> {
    pub fn new<E: EmbeddingProvider + 'static>(
        db: &'a dyn Repository,
        embedder: E,
        config: IndexerConfig,
    ) -> Self {
        Self {
            db,
            embedder: Box::new(embedder),
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

        let observed_graphs: Arc<Mutex<HashMap<String, ObservedGraph>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let observer = Arc::new(GraphObserver::new(observed_graphs.clone()));
        let mut stream = walk_project_with_observer(&self.config.repo_path, options, observer);
        let mut processor = IndexProcessor::new(
            self.db,
            self.embedder.as_ref(),
            &self.config,
            observed_graphs,
        );

        while let Some(item) = stream.next().await {
            let project_chunk = item?;
            processor.handle_chunk(project_chunk).await?;
        }

        processor.finish().await?;

        self.db.refresh_fts_index()?;
        index_history(self.db, &self.config.repo_path, &self.config.history)?;
        self.db.refresh_file_dependency_edges()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use text_chunking::Tokenizer;

    use crate::db::Db;
    use crate::graph::HistoryConfig;
    use crate::search;
    use crate::test_support::{load_test_embedder, write_fixture_repo};

    #[tokio::test]
    async fn end_to_end_index_and_search() -> Result<()> {
        let (embedder, embedding_dim) = load_test_embedder()?;
        let dir = TempDir::new()?;
        write_fixture_repo(dir.path())?;

        let db_path = dir.path().join("context.duckdb");
        let db = Db::open(&db_path, Some(embedding_dim))?;
        let tokenizer = Tokenizer::Tiktoken("cl100k_base".to_string());
        let config = IndexerConfig {
            repo_path: dir.path().to_path_buf(),
            max_chunk_size: 512,
            overlap_percentage: 0.1,
            tokenizer: tokenizer.clone(),
            max_parallel: 2,
            max_file_size: Some(5 * 1024 * 1024),
            large_file_threads: 2,
            stream_batch_size: 256,
            max_batch_size: 16,
            max_tokens: 8_192,
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
        let indexer = Indexer::new(&db, embedder.clone(), config);
        indexer.index().await?;

        let db = Db::open(&db_path, Some(embedding_dim))?;
        let mut search_config = search::SearchConfig::new(5, 0.6);
        search_config.min_score = 0.0;
        let results =
            search::search(&db, &embedder, None, &tokenizer, "add numbers", search_config).await?;

        assert!(!results.is_empty(), "expected search to return results");
        Ok(())
    }
}

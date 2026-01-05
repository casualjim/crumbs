use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use eyre::{Result, eyre};
use futures::StreamExt;
use text_chunking::{
    Chunk, ChunkError, CodeParseInfo, CodeParseObserver, Tokenizer, WalkOptions,
    walk_project_with_observer,
};
use tracing::warn;
use uuid::Uuid;

use crate::db::{ChunkRecord, GraphData};
use crate::embedding::{EmbeddingInput, EmbeddingProvider};
use crate::graph::{HistoryConfig, extract_graph_from_tree, index_history};
use crate::repository::Repository;

pub struct IndexerConfig {
    pub repo_path: PathBuf,
    pub max_chunk_size: usize,
    pub overlap_percentage: f32,
    pub tokenizer: Tokenizer,
    pub max_parallel: usize,
    pub max_file_size: Option<u64>,
    pub large_file_threads: usize,
    pub history: HistoryConfig,
}

pub struct Indexer<'a> {
    db: &'a dyn Repository,
    embedder: Box<dyn EmbeddingProvider>,
    config: IndexerConfig,
}

struct PendingFile {
    file_size: u64,
    chunks: Vec<ChunkRecord>,
    embeddings: Vec<Vec<f32>>,
}

impl PendingFile {
    fn new(file_size: u64) -> Self {
        Self {
            file_size,
            chunks: Vec::new(),
            embeddings: Vec::new(),
        }
    }
}

struct ObservedGraph {
    language: String,
    graph: GraphData,
}

struct GraphObserver {
    graphs: Arc<Mutex<HashMap<String, ObservedGraph>>>,
}

impl GraphObserver {
    fn new(graphs: Arc<Mutex<HashMap<String, ObservedGraph>>>) -> Self {
        Self { graphs }
    }
}

impl CodeParseObserver for GraphObserver {
    fn on_parse(&self, info: CodeParseInfo) -> Result<(), ChunkError> {
        let language = info.language_id.clone();

        match extract_graph_from_tree(
            &language,
            info.language,
            info.tree.as_ref(),
            info.source.as_ref(),
        ) {
            Ok(Some(graph)) => {
                let mut guard = self.graphs.lock().expect("graph observer lock poisoned");
                guard.insert(info.file_path.clone(), ObservedGraph { language, graph });
            }
            Ok(None) => {}
            Err(err) => {
                warn!("graph extraction failed for {}: {}", info.file_path, err);
            }
        }

        Ok(())
    }
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
        let mut pending: HashMap<String, PendingFile> = HashMap::new();

        while let Some(item) = stream.next().await {
            let project_chunk = item?;
            match project_chunk.chunk {
                Chunk::Delete { file_path } => {
                    self.db.delete_file(&file_path)?;
                    pending.remove(&file_path);
                    observed_graphs
                        .lock()
                        .expect("graph observer lock poisoned")
                        .remove(&file_path);
                }
                Chunk::Semantic(chunk) => {
                    let file_path = project_chunk.file_path.clone();
                    let entry = pending
                        .entry(file_path.clone())
                        .or_insert_with(|| PendingFile::new(project_chunk.file_size));
                    entry.file_size = project_chunk.file_size;
                    let ordinal = entry.chunks.len();
                    let token_count = chunk.tokens.as_ref().map(|tokens| tokens.len());
                    let embedding =
                        embed_text(self.embedder.as_ref(), chunk.text.clone(), token_count).await?;

                    let record = ChunkRecord {
                        id: Uuid::now_v7().to_string(),
                        file_path,
                        start_byte: chunk.start_byte,
                        end_byte: chunk.end_byte,
                        text: chunk.text,
                        kind: "semantic".to_string(),
                        ordinal,
                        tokens: chunk.tokens.clone(),
                    };

                    entry.chunks.push(record);
                    entry.embeddings.push(embedding);
                }
                Chunk::Text(chunk) => {
                    let file_path = project_chunk.file_path.clone();
                    let entry = pending
                        .entry(file_path.clone())
                        .or_insert_with(|| PendingFile::new(project_chunk.file_size));
                    entry.file_size = project_chunk.file_size;
                    let ordinal = entry.chunks.len();
                    let token_count = chunk.tokens.as_ref().map(|tokens| tokens.len());
                    let embedding =
                        embed_text(self.embedder.as_ref(), chunk.text.clone(), token_count).await?;

                    let record = ChunkRecord {
                        id: Uuid::now_v7().to_string(),
                        file_path,
                        start_byte: chunk.start_byte,
                        end_byte: chunk.end_byte,
                        text: chunk.text,
                        kind: "text".to_string(),
                        ordinal,
                        tokens: chunk.tokens.clone(),
                    };

                    entry.chunks.push(record);
                    entry.embeddings.push(embedding);
                }
                Chunk::EndOfFile {
                    file_path,
                    content_hash,
                    ..
                } => {
                    let Some(hash) = content_hash else {
                        return Err(eyre!("missing content hash for {}", file_path));
                    };

                    let entry = pending
                        .remove(&file_path)
                        .unwrap_or_else(|| PendingFile::new(project_chunk.file_size));

                    let observed = observed_graphs
                        .lock()
                        .expect("graph observer lock poisoned")
                        .remove(&file_path);

                    if let Some(observed) = observed {
                        self.db.replace_file_graph(
                            &file_path,
                            entry.file_size,
                            hash,
                            &observed.language,
                            observed.graph,
                        )?;
                    }

                    self.db.replace_file_chunks(
                        &file_path,
                        entry.file_size,
                        hash,
                        &entry.chunks,
                        &entry.embeddings,
                    )?;
                }
            }
        }

        if !pending.is_empty() {
            let incomplete = pending.keys().cloned().collect::<Vec<_>>().join(", ");
            return Err(eyre!(
                "indexing ended with {} file(s) missing EOF: {}",
                pending.len(),
                incomplete
            ));
        }

        self.db.refresh_fts_index()?;
        index_history(self.db, &self.config.repo_path, &self.config.history)?;

        Ok(())
    }
}

async fn embed_text(
    client: &dyn EmbeddingProvider,
    text: String,
    token_count: Option<usize>,
) -> Result<Vec<f32>> {
    let input = EmbeddingInput { text, token_count };
    let output = client.embed(&[input]).await?;
    let mut embeddings = output.embeddings;
    if embeddings.is_empty() {
        return Err(eyre!("embedder returned no embeddings"));
    }
    Ok(embeddings.remove(0))
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
        let config = IndexerConfig {
            repo_path: dir.path().to_path_buf(),
            max_chunk_size: 512,
            overlap_percentage: 0.1,
            tokenizer: Tokenizer::Characters,
            max_parallel: 2,
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
        let indexer = Indexer::new(&db, embedder.clone(), config);
        indexer.index().await?;

        let db = Db::open(&db_path, Some(embedding_dim))?;
        let results = search::search(&db, &embedder, "add numbers", 5, 0.6).await?;

        assert!(!results.is_empty(), "expected search to return results");
        Ok(())
    }
}

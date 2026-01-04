use std::collections::HashMap;
use std::path::PathBuf;

use eyre::{Result, eyre};
use futures::StreamExt;
use text_chunking::{Chunk, Tokenizer, WalkOptions, walk_project};
use uuid::Uuid;

use crate::db::{ChunkRecord, Db};
use crate::embedding::{EmbeddingInput, EmbeddingProvider};

pub struct IndexerConfig {
    pub repo_path: PathBuf,
    pub max_chunk_size: usize,
    pub overlap_percentage: f32,
    pub tokenizer: Tokenizer,
    pub max_parallel: usize,
    pub max_file_size: Option<u64>,
    pub large_file_threads: usize,
}

pub struct Indexer {
    db: Db,
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

impl Indexer {
    pub fn new<E: EmbeddingProvider + 'static>(db: Db, embedder: E, config: IndexerConfig) -> Self {
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

        let mut stream = walk_project(&self.config.repo_path, options);
        let mut pending: HashMap<String, PendingFile> = HashMap::new();

        while let Some(item) = stream.next().await {
            let project_chunk = item?;
            match project_chunk.chunk {
                Chunk::Delete { file_path } => {
                    self.db.delete_file(&file_path)?;
                    pending.remove(&file_path);
                }
                Chunk::Semantic(chunk) => {
                    let file_path = project_chunk.file_path.clone();
                    let entry = pending
                        .entry(file_path.clone())
                        .or_insert_with(|| PendingFile::new(project_chunk.file_size));
                    entry.file_size = project_chunk.file_size;
                    let ordinal = entry.chunks.len();
                    let embedding = embed_text(self.embedder.as_ref(), chunk.text.clone()).await?;

                    let record = ChunkRecord {
                        id: Uuid::now_v7().to_string(),
                        file_path,
                        start_byte: chunk.start_byte,
                        end_byte: chunk.end_byte,
                        text: chunk.text,
                        kind: "semantic".to_string(),
                        ordinal,
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
                    let embedding = embed_text(self.embedder.as_ref(), chunk.text.clone()).await?;

                    let record = ChunkRecord {
                        id: Uuid::now_v7().to_string(),
                        file_path,
                        start_byte: chunk.start_byte,
                        end_byte: chunk.end_byte,
                        text: chunk.text,
                        kind: "text".to_string(),
                        ordinal,
                    };

                    entry.chunks.push(record);
                    entry.embeddings.push(embedding);
                }
                Chunk::EndOfFile {
                    file_path,
                    content_hash,
                    ..
                } => {
                    if let Some(hash) = content_hash {
                        let entry = pending
                            .remove(&file_path)
                            .unwrap_or_else(|| PendingFile::new(project_chunk.file_size));
                        self.db.replace_file_chunks(
                            &file_path,
                            entry.file_size,
                            hash,
                            &entry.chunks,
                            &entry.embeddings,
                        )?;
                    } else {
                        return Err(eyre!("missing content hash for {}", file_path));
                    }
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

        Ok(())
    }
}

async fn embed_text(client: &dyn EmbeddingProvider, text: String) -> Result<Vec<f32>> {
    let input = EmbeddingInput {
        text,
        token_count: None,
    };
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
    use async_trait::async_trait;
    use tempfile::TempDir;
    use text_chunking::Tokenizer;

    use crate::search;
    use crate::test_support::write_fixture_repo;

    #[derive(Clone)]
    struct FakeEmbedder {
        embedding: Vec<f32>,
    }

    #[async_trait]
    impl EmbeddingProvider for FakeEmbedder {
        async fn embed(&self, input: &[EmbeddingInput]) -> Result<crate::embedding::EmbedOutput> {
            Ok(crate::embedding::EmbedOutput {
                embeddings: vec![self.embedding.clone(); input.len()],
            })
        }
    }

    #[tokio::test]
    async fn end_to_end_index_and_search() -> Result<()> {
        let dir = TempDir::new()?;
        write_fixture_repo(dir.path())?;

        let db_path = dir.path().join("context.duckdb");
        let embedder = FakeEmbedder {
            embedding: vec![0.1, 0.2],
        };
        let db = Db::open(&db_path, Some(2))?;
        let config = IndexerConfig {
            repo_path: dir.path().to_path_buf(),
            max_chunk_size: 512,
            overlap_percentage: 0.1,
            tokenizer: Tokenizer::Characters,
            max_parallel: 2,
            max_file_size: Some(5 * 1024 * 1024),
            large_file_threads: 2,
        };
        let indexer = Indexer::new(db, embedder, config);
        indexer.index().await?;

        let embedder = FakeEmbedder {
            embedding: vec![0.1, 0.2],
        };
        let db = Db::open(&db_path, Some(2))?;
        let results = search::search(&db, &embedder, "add numbers", 5, 0.6).await?;

        assert!(!results.is_empty(), "expected search to return results");
        Ok(())
    }
}

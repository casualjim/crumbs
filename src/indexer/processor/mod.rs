mod finalize;
mod handlers;

use std::sync::{Arc, Mutex};

use eyre::{Result, eyre};
use text_chunking::{Chunk, ProjectChunk};

use crate::embedding::EmbeddingProvider;
use crate::repository::Repository;

use super::IndexerConfig;
use super::batcher::{ChunkBatch, StreamBatcher, split_by_tokens};
use super::observer::ObservedGraph;
use super::state::IndexerState;

pub(crate) struct IndexProcessor<'a> {
    db: &'a dyn Repository,
    embedder: &'a dyn EmbeddingProvider,
    state: IndexerState,
    stream_batcher: StreamBatcher,
    max_tokens: usize,
    max_batch_size: usize,
    observed_graphs: Arc<Mutex<std::collections::HashMap<String, ObservedGraph>>>,
}

impl<'a> IndexProcessor<'a> {
    pub(crate) fn new(
        db: &'a dyn Repository,
        embedder: &'a dyn EmbeddingProvider,
        config: &'a IndexerConfig,
        observed_graphs: Arc<Mutex<std::collections::HashMap<String, ObservedGraph>>>,
    ) -> Self {
        Self {
            db,
            embedder,
            state: IndexerState::new(),
            stream_batcher: StreamBatcher::new(config.stream_batch_size),
            max_tokens: config.max_tokens,
            max_batch_size: config.max_batch_size,
            observed_graphs,
        }
    }

    pub(crate) async fn handle_chunk(&mut self, project_chunk: ProjectChunk) -> Result<()> {
        let ProjectChunk {
            file_path,
            chunk,
            file_size,
        } = project_chunk;
        match chunk {
            Chunk::Delete { file_path } => self.handle_delete(&file_path),
            Chunk::Semantic(chunk) => {
                self.handle_chunk_like(file_path, file_size, chunk, "semantic")
                    .await
            }
            Chunk::Text(chunk) => {
                self.handle_chunk_like(file_path, file_size, chunk, "text")
                    .await
            }
            Chunk::EndOfFile {
                file_path,
                content_hash,
                file_metadata,
                ..
            } => self.handle_eof(file_size, &file_path, content_hash, file_metadata),
        }
    }

    pub(crate) async fn finish(&mut self) -> Result<()> {
        if let Some(batch) = self.stream_batcher.flush() {
            self.process_chunk_batch(batch).await?;
        }

        if !self.state.pending_eof.is_empty() {
            let remaining = self.state.pending_eof.keys().cloned().collect::<Vec<_>>();
            self.try_finalize_files(remaining)?;
        }

        if !self.state.pending.is_empty() {
            let incomplete = self
                .state
                .pending
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            return Err(eyre!(
                "indexing ended with {} file(s) missing EOF: {}",
                self.state.pending.len(),
                incomplete
            ));
        }

        Ok(())
    }

    pub(super) async fn process_chunk_batch(&mut self, batch: ChunkBatch) -> Result<()> {
        let batches = split_by_tokens(batch.items, self.max_tokens, self.max_batch_size);
        for embedding_batch in batches {
            self.process_batch(embedding_batch).await?;
        }
        Ok(())
    }
}

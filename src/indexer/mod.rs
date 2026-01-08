mod batcher;
mod embedder;
mod processor;
mod state;

use std::path::PathBuf;
use std::sync::Arc;

use crate::embedding::EmbeddingProvider;
use crate::graph::{HistoryConfig, index_history};
use crate::progress;
use crate::repository::Repository;
use eyre::{Result, eyre};
use futures::StreamExt;
use text_chunking::{Tokenizer, WalkOptions, walk_project};
use tokio_util::sync::CancellationToken;

use self::embedder::EmbedderService;
use self::processor::{IndexProcessor, ProcessorOutput};

pub struct IndexerConfig {
    pub repo_path: PathBuf,
    pub max_chunk_size: usize,
    pub overlap_percentage: f32,
    pub tokenizer: Tokenizer,
    pub max_parallel: usize,
    pub max_file_size: Option<u64>,
    pub large_file_threads: usize,
    pub max_batch_size: usize,
    pub max_tokens: usize,
    pub embedding_workers: usize,
    pub cancel_token: Option<CancellationToken>,
    pub history: HistoryConfig,
}

pub struct Indexer<'a> {
    db: &'a dyn Repository,
    embedder: Arc<dyn EmbeddingProvider>,
    config: IndexerConfig,
}

impl<'a> Indexer<'a> {
    pub fn new<E: EmbeddingProvider + 'static>(
        db: &'a dyn Repository,
        embedder: E,
        config: IndexerConfig,
    ) -> Self {
        let embedder = Arc::new(embedder);
        Self {
            db,
            embedder,
            config,
        }
    }

    pub async fn index(self) -> Result<()> {
        let progress = progress::spinner("indexing files");
        let mut files_processed = 0usize;
        let update_progress = |files_processed: usize, pending_batches: usize, stream_done: bool| {
            if let Some(progress) = &progress {
                let message = if stream_done {
                    format!(
                        "processed {files_processed} files; embedding {pending_batches} batches"
                    )
                } else if pending_batches > 0 {
                    format!(
                        "processed {files_processed} files; pending {pending_batches} batches"
                    )
                } else {
                    format!("processed {files_processed} files")
                };
                progress.set_message(message);
            }
        };

        let existing_hashes = self.db.load_existing_hashes().await?;
        let options = WalkOptions {
            max_chunk_size: self.config.max_chunk_size,
            tokenizer: self.config.tokenizer.clone(),
            overlap_percentage: self.config.overlap_percentage,
            max_parallel: self.config.max_parallel,
            max_file_size: self.config.max_file_size,
            large_file_threads: self.config.large_file_threads,
            existing_hashes,
            cancel_token: self.config.cancel_token.clone(),
        };

        let mut stream = walk_project(&self.config.repo_path, options);
        let mut processor = IndexProcessor::new(self.db);
        let (mut embedder_service, mut result_rx) = EmbedderService::new(
            Arc::clone(&self.embedder),
            self.config.max_tokens,
            self.config.max_batch_size,
            self.config.embedding_workers,
        );
        let mut pending_batches = 0usize;

        let mut stream_done = false;
        loop {
            tokio::select! {
                item = stream.next(), if !stream_done => {
                    match item {
                        Some(Ok(project_chunk)) => {
                            let file_done = matches!(
                                &project_chunk.chunk,
                                text_chunking::Chunk::EndOfFile { .. }
                                    | text_chunking::Chunk::Delete { .. }
                            );
                            match processor.handle_chunk(project_chunk).await? {
                                ProcessorOutput::Batch(batch_item) => {
                                    let enqueued = embedder_service.enqueue(batch_item).await?;
                                    if enqueued {
                                        pending_batches = pending_batches.saturating_add(1);
                                        update_progress(
                                            files_processed,
                                            pending_batches,
                                            stream_done,
                                        );
                                    }
                                }
                                ProcessorOutput::RemoveFile(file_path) => {
                                    embedder_service.remove_file(&file_path);
                                }
                                ProcessorOutput::None => {}
                            }
                            if file_done {
                                files_processed = files_processed.saturating_add(1);
                                update_progress(files_processed, pending_batches, stream_done);
                            }
                        }
                        Some(Err(err)) => {
                            return Err(eyre!(err));
                        }
                        None => {
                            stream_done = true;
                            if embedder_service.flush().await? {
                                pending_batches = pending_batches.saturating_add(1);
                            }
                            update_progress(files_processed, pending_batches, stream_done);
                        }
                    }
                }
                result = result_rx.recv(), if pending_batches > 0 || stream_done => {
                    match result {
                        Some(Ok(result)) => {
                            pending_batches = pending_batches.saturating_sub(1);
                            processor
                                .apply_embeddings(result.items, result.embeddings)
                                .await?;
                            update_progress(files_processed, pending_batches, stream_done);
                        }
                        Some(Err(err)) => {
                            return Err(err);
                        }
                        None => {
                            if pending_batches > 0 {
                                return Err(eyre!(
                                    "embedder result channel closed with {} pending batches",
                                    pending_batches
                                ));
                            }
                            if stream_done {
                                break;
                            }
                        }
                    }
                }
            }

            if stream_done && pending_batches == 0 {
                break;
            }
        }

        processor.finish().await?;

        index_history(self.db, &self.config.repo_path, &self.config.history).await?;

        if let Some(progress) = progress {
            progress.finish_and_clear();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests;

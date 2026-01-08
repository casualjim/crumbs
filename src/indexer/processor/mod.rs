mod finalize;
mod handlers;

use eyre::{Result, eyre};
use text_chunking::{Chunk, ProjectChunk};

use crate::repository::Repository;

use super::batcher::BatchItem;
use super::state::IndexerState;

pub(crate) struct IndexProcessor<'a> {
    db: &'a dyn Repository,
    state: IndexerState,
}

pub(crate) enum ProcessorOutput {
    Batch(BatchItem),
    RemoveFile(String),
    None,
}

impl<'a> IndexProcessor<'a> {
    pub(crate) fn new(db: &'a dyn Repository) -> Self {
        Self {
            db,
            state: IndexerState::new(),
        }
    }

    pub(crate) async fn handle_chunk(
        &mut self,
        project_chunk: ProjectChunk,
    ) -> Result<ProcessorOutput> {
        let ProjectChunk {
            file_path,
            chunk,
            file_size,
        } = project_chunk;
        match chunk {
            Chunk::Delete { file_path } => {
                self.handle_delete(&file_path).await?;
                Ok(ProcessorOutput::RemoveFile(file_path))
            }
            Chunk::Semantic(chunk) => {
                let item = self
                    .handle_content_chunk(file_path, file_size, chunk, "semantic")
                    .await?;
                Ok(item.map_or(ProcessorOutput::None, ProcessorOutput::Batch))
            }
            Chunk::Text(chunk) => {
                let item = self
                    .handle_content_chunk(file_path, file_size, chunk, "text")
                    .await?;
                Ok(item.map_or(ProcessorOutput::None, ProcessorOutput::Batch))
            }
            Chunk::EndOfFile {
                file_path,
                content_hash,
                file_metadata,
                ..
            } => {
                self.handle_eof(file_size, &file_path, content_hash, file_metadata)
                    .await?;
                Ok(ProcessorOutput::None)
            }
        }
    }

    pub(crate) async fn finish(&mut self) -> Result<()> {
        if !self.state.pending_eof.is_empty() {
            let remaining = self.state.pending_eof.keys().cloned().collect::<Vec<_>>();
            self.try_finalize_files(remaining).await?;
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
}

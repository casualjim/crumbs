use eyre::{Result, eyre};
use text_chunking::{FileMetadata, SemanticChunk};
use uuid::Uuid;

use crate::db::ChunkRecord;

use super::IndexProcessor;
use super::super::batcher::BatchItem;
use super::super::state::{PendingEof, PendingFile};

impl<'a> IndexProcessor<'a> {
    pub(super) fn handle_delete(&mut self, file_path: &str) -> Result<()> {
        self.db.delete_file(file_path)?;
        if let Some(entry) = self.state.pending.remove(file_path) {
            for chunk in entry.chunks {
                self.state.embeddings.remove(&chunk.id);
            }
        }
        self.state.pending_eof.remove(file_path);
        self.stream_batcher.remove_file(file_path);
        self.observed_graphs
            .lock()
            .expect("graph observer lock poisoned")
            .remove(file_path);
        Ok(())
    }

    pub(super) async fn handle_chunk_like(
        &mut self,
        file_path: String,
        file_size: u64,
        chunk: SemanticChunk,
        kind: &str,
    ) -> Result<()> {
        let SemanticChunk {
            start_byte,
            end_byte,
            chunk_hash,
            start_line,
            end_line,
            text,
            tokens,
        } = chunk;
        let entry = self
            .state
            .pending
            .entry(file_path.clone())
            .or_insert_with(|| PendingFile::new(file_size));
        entry.file_size = file_size;

        let ordinal = entry.chunks.len();
        let token_count = tokens.as_ref().map(|tokens| tokens.len()).ok_or_else(|| {
            eyre!(
                "missing token counts for chunk; configure tokenizer to provide tokens"
            )
        })?;

        let record = ChunkRecord {
            id: Uuid::now_v7().to_string(),
            file_path,
            start_byte,
            end_byte,
            chunk_hash,
            start_line,
            end_line,
            text,
            kind: kind.to_string(),
            ordinal,
            tokens,
        };

        let batch_item = BatchItem {
            chunk_id: record.id.clone(),
            file_path: record.file_path.clone(),
            text: record.text.clone(),
            token_count,
        };
        entry.chunks.push(record);
        self.maybe_process_batch(batch_item).await
    }

    pub(super) fn handle_eof(
        &mut self,
        file_size: u64,
        file_path: &str,
        content_hash: Option<[u8; 32]>,
        file_metadata: Option<FileMetadata>,
    ) -> Result<()> {
        let Some(hash) = content_hash else {
            return Err(eyre!("missing content hash for {}", file_path));
        };
        let primary_language = file_metadata
            .and_then(|metadata| metadata.primary_language)
            .map(|lang| lang.trim().to_string())
            .filter(|lang| !lang.is_empty());

        let entry = self
            .state
            .pending
            .entry(file_path.to_string())
            .or_insert_with(|| PendingFile::new(file_size));
        entry.file_size = file_size;

        let observed = self
            .observed_graphs
            .lock()
            .expect("graph observer lock poisoned")
            .remove(file_path);

        if let Some(observed) = observed {
            self.db.replace_file_graph(
                file_path,
                entry.file_size,
                hash,
                &observed.language,
                primary_language.clone(),
                observed.graph,
            )?;
        }

        self.state.pending_eof.insert(
            file_path.to_string(),
            PendingEof {
                content_hash: hash,
                primary_language: primary_language.clone(),
            },
        );
        self.try_finalize_files(vec![file_path.to_string()])
    }

    pub(super) async fn maybe_process_batch(&mut self, item: BatchItem) -> Result<()> {
        if let Some(batch) = self.stream_batcher.add(item) {
            self.process_chunk_batch(batch).await?;
        }
        Ok(())
    }
}

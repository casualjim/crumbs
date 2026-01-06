use std::collections::HashSet;

use eyre::{Result, eyre};

use super::IndexProcessor;
use super::super::batcher::BatchItem;
use super::super::embedder::embed_batch;
use super::super::state::{PendingEof, PendingFile};

impl<'a> IndexProcessor<'a> {
    pub(super) async fn process_batch(&mut self, batch: Vec<BatchItem>) -> Result<()> {
        let touched = embed_batch(self.embedder, batch, &mut self.state.embeddings).await?;
        self.try_finalize_files(touched)
    }

    pub(super) fn try_finalize_files(
        &mut self,
        file_paths: impl IntoIterator<Item = String>,
    ) -> Result<()> {
        let mut seen = HashSet::new();
        for file_path in file_paths {
            if !seen.insert(file_path.clone()) {
                continue;
            }

            let Some(eof) = self.state.pending_eof.get(&file_path) else {
                continue;
            };
            let Some(entry) = self.state.pending.get(&file_path) else {
                continue;
            };
            let ready = entry
                .chunks
                .iter()
                .all(|chunk| self.state.embeddings.contains_key(&chunk.id));
            if !ready {
                continue;
            }

            let eof = PendingEof {
                content_hash: eof.content_hash,
                primary_language: eof.primary_language.clone(),
            };
            let entry = self
                .state
                .pending
                .remove(&file_path)
                .unwrap_or_else(|| PendingFile::new(0));

            let mut ordered = Vec::with_capacity(entry.chunks.len());
            for chunk in &entry.chunks {
                let embedding = self
                    .state
                    .embeddings
                    .remove(&chunk.id)
                    .ok_or_else(|| eyre!("missing embedding for {}", chunk.id))?;
                ordered.push(embedding);
            }

            self.db.replace_file_chunks(
                &file_path,
                entry.file_size,
                eof.content_hash,
                eof.primary_language.clone(),
                &entry.chunks,
                &ordered,
            )?;
            self.state.pending_eof.remove(&file_path);
        }

        Ok(())
    }
}

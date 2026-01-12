use std::collections::HashSet;

use eyre::{Result, eyre};

use super::super::EmbedMeta;
use super::super::state::{PendingEof, PendingFile};
use super::IndexProcessor;

impl<'a> IndexProcessor<'a> {
    pub(crate) async fn apply_embeddings(
        &mut self,
        batch: Vec<EmbedMeta>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<()> {
        if embeddings.len() != batch.len() {
            return Err(eyre!(
                "embedder returned {} embeddings for {} inputs",
                embeddings.len(),
                batch.len()
            ));
        }

        let mut touched = HashSet::new();
        let mut records = Vec::with_capacity(batch.len());
        let mut embed_values = Vec::with_capacity(batch.len());
        let mut pending = Vec::with_capacity(batch.len());

        for (item, embedding) in batch.into_iter().zip(embeddings.into_iter()) {
            let Some(entry) = self.state.pending.get_mut(&item.file_path) else {
                continue;
            };
            let Some(idx) = entry.chunk_index.get(&item.chunk_id).copied() else {
                continue;
            };
            if let Some(record) = entry.chunks.get(idx) {
                records.push(record.clone());
                embed_values.push(embedding);
                pending.push((item.file_path, item.chunk_id));
            }
        }

        if !records.is_empty() {
            self.db
                .upsert_chunks_with_embeddings(&records, &embed_values)
                .await?;
            for (file_path, chunk_id) in pending {
                if let Some(entry) = self.state.pending.get_mut(&file_path) {
                    entry.pending_embeddings.remove(&chunk_id);
                }
                touched.insert(file_path);
            }
        }

        self.try_finalize_files(touched).await
    }

    pub(super) async fn try_finalize_files(
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
            if !entry.pending_embeddings.is_empty() {
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

            let keep_ids: Vec<String> = entry.chunks.iter().map(|chunk| chunk.id.clone()).collect();
            self.db
                .upsert_file_metadata(
                    &file_path,
                    entry.file_size,
                    eof.content_hash,
                    eof.primary_language.clone(),
                )
                .await?;
            self.db.delete_missing_chunks(&file_path, &keep_ids).await?;
            self.state.pending_eof.remove(&file_path);
        }

        Ok(())
    }
}

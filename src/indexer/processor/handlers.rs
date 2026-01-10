use blake3::Hasher;
use eyre::{Result, eyre};
use text_chunking::{FileMetadata, SemanticChunk};
use uuid::Uuid;

use crate::db::{ChunkRecord, GraphData, ReferenceRecord, SymbolRecord, build_fts_text};

use super::super::batcher::BatchItem;
use super::super::state::{PendingEof, PendingFile};
use super::IndexProcessor;

impl<'a> IndexProcessor<'a> {
    pub(super) async fn handle_delete(&mut self, file_path: &str) -> Result<()> {
        self.db.delete_file(file_path).await?;
        self.state.pending.remove(file_path);
        self.state.pending_eof.remove(file_path);
        Ok(())
    }

    pub(super) async fn handle_content_chunk(
        &mut self,
        file_path: String,
        file_size: u64,
        chunk: SemanticChunk,
        kind: &str,
    ) -> Result<Option<BatchItem>> {
        let SemanticChunk {
            start_byte,
            end_byte,
            chunk_hash,
            start_line,
            end_line,
            text,
            tokens,
            metadata,
        } = chunk;
        let file_path_value = file_path.clone();
        let entry = self
            .state
            .pending
            .entry(file_path.clone())
            .or_insert_with(|| PendingFile::new(file_size));
        entry.file_size = file_size;
        if !entry.file_row_ensured {
            self.db.ensure_file_row(&file_path, file_size, None).await?;
            entry.file_row_ensured = true;
        }

        let ordinal = entry.chunks.len();
        let token_count = tokens.as_ref().map(|tokens| tokens.len()).ok_or_else(|| {
            eyre!("missing token counts for chunk; configure tokenizer to provide tokens")
        })?;

        let existing_id = self
            .db
            .find_chunk_id(&file_path, start_byte, end_byte, kind, chunk_hash)
            .await?;
        let fts_text = build_fts_text(&text);
        let record = ChunkRecord {
            id: existing_id
                .clone()
                .unwrap_or_else(|| Uuid::now_v7().to_string()),
            file_path,
            start_byte,
            end_byte,
            chunk_hash,
            start_line,
            end_line,
            text,
            fts_text,
            kind: kind.to_string(),
            ordinal,
            tokens,
        };

        let is_existing = existing_id.is_some();
        let chunk_id = record.id.clone();
        let record_file_path = record.file_path.clone();
        let record_text = record.text.clone();
        let record = {
            let idx = entry.chunks.len();
            entry.chunks.push(record);
            entry.chunk_index.insert(chunk_id.clone(), idx);
            entry.chunks[idx].clone()
        };

        if kind == "semantic" {
            let language = metadata.language.trim().to_string();
            if !language.is_empty() {
                entry.graph_language = Some(language.clone());
            }
            let graph_language = entry
                .graph_language
                .as_deref()
                .unwrap_or("text")
                .to_string();
            for name in metadata.definitions {
                let id = stable_id(&file_path_value, "definition", start_byte, end_byte, &name);
                if entry.symbol_ids.insert(id.clone()) {
                    entry.symbols.push(SymbolRecord {
                        id,
                        file_path: file_path_value.clone(),
                        name,
                        kind: "definition".to_string(),
                        start_byte,
                        end_byte,
                        language: graph_language.clone(),
                    });
                }
            }
            for name in metadata.references {
                let id = stable_id(&file_path_value, "reference", start_byte, end_byte, &name);
                if entry.reference_ids.insert(id.clone()) {
                    entry.references.push(ReferenceRecord {
                        id,
                        file_path: file_path_value.clone(),
                        name,
                        start_byte,
                        end_byte,
                        language: graph_language.clone(),
                    });
                }
            }
        }

        if is_existing {
            self.db.update_chunk_without_embedding(&record).await?;
            return Ok(None);
        }

        entry.pending_embeddings.insert(chunk_id.clone());

        let batch_item = BatchItem {
            chunk_id,
            file_path: record_file_path,
            text: record_text,
            token_count,
        };
        Ok(Some(batch_item))
    }

    pub(super) async fn handle_eof(
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

        let graph_language = entry
            .graph_language
            .clone()
            .or_else(|| primary_language.clone())
            .unwrap_or_else(|| "text".to_string());
        let graph = GraphData {
            symbols: entry.symbols.clone(),
            references: entry.references.clone(),
            resolutions: Vec::new(),
        };
        self.db
            .upsert_file_graph(
                file_path,
                entry.file_size,
                hash,
                &graph_language,
                primary_language.clone(),
                graph,
            )
            .await?;

        self.state.pending_eof.insert(
            file_path.to_string(),
            PendingEof {
                content_hash: hash,
                primary_language: primary_language.clone(),
            },
        );
        self.try_finalize_files(vec![file_path.to_string()]).await
    }
}

fn stable_id(
    file_path: &str,
    kind: &str,
    start_byte: usize,
    end_byte: usize,
    name: &str,
) -> String {
    let mut hasher = Hasher::new();
    hasher.update(file_path.as_bytes());
    hasher.update(kind.as_bytes());
    hasher.update(&start_byte.to_le_bytes());
    hasher.update(&end_byte.to_le_bytes());
    hasher.update(name.as_bytes());
    let hash = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in hash.as_bytes() {
        use std::fmt::Write as _;
        let _ = write!(out, "{:02x}", byte);
    }
    out
}

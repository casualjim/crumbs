use std::collections::HashMap;

use crate::db::ChunkRecord;

pub(crate) struct PendingFile {
    pub(crate) file_size: u64,
    pub(crate) chunks: Vec<ChunkRecord>,
}

impl PendingFile {
    pub(crate) fn new(file_size: u64) -> Self {
        Self {
            file_size,
            chunks: Vec::new(),
        }
    }
}

pub(crate) struct PendingEof {
    pub(crate) content_hash: [u8; 32],
    pub(crate) primary_language: Option<String>,
}

pub(crate) struct IndexerState {
    pub(crate) pending: HashMap<String, PendingFile>,
    pub(crate) pending_eof: HashMap<String, PendingEof>,
    pub(crate) embeddings: HashMap<String, Vec<f32>>,
}

impl IndexerState {
    pub(crate) fn new() -> Self {
        Self {
            pending: HashMap::new(),
            pending_eof: HashMap::new(),
            embeddings: HashMap::new(),
        }
    }
}

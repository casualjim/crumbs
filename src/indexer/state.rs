use std::collections::{HashMap, HashSet};

use crate::db::ChunkRecord;

pub(crate) struct PendingFile {
    pub(crate) file_size: u64,
    pub(crate) chunks: Vec<ChunkRecord>,
    pub(crate) chunk_index: HashMap<String, usize>,
    pub(crate) pending_embeddings: HashSet<String>,
    pub(crate) file_row_ensured: bool,
}

impl PendingFile {
    pub(crate) fn new(file_size: u64) -> Self {
        Self {
            file_size,
            chunks: Vec::new(),
            chunk_index: HashMap::new(),
            pending_embeddings: HashSet::new(),
            file_row_ensured: false,
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
}

impl IndexerState {
    pub(crate) fn new() -> Self {
        Self {
            pending: HashMap::new(),
            pending_eof: HashMap::new(),
        }
    }
}

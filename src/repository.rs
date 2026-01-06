use std::collections::BTreeMap;
use std::path::PathBuf;

use eyre::Result;

use crate::db::{ChunkRecord, FtsRow, GraphData, SearchRow, SymbolRecord};

pub trait Repository {
    fn load_existing_hashes(&self) -> Result<BTreeMap<PathBuf, [u8; 32]>>;
    fn delete_file(&self, file_path: &str) -> Result<()>;
    fn replace_file_chunks(
        &self,
        file_path: &str,
        file_size: u64,
        content_hash: [u8; 32],
        chunks: &[ChunkRecord],
        embeddings: &[Vec<f32>],
    ) -> Result<()>;
    fn refresh_fts_index(&self) -> Result<()>;
    fn replace_file_graph(
        &self,
        file_path: &str,
        file_size: u64,
        content_hash: [u8; 32],
        language: &str,
        graph: GraphData,
    ) -> Result<()>;
    fn list_files(&self) -> Result<Vec<String>>;
    fn replace_history_edges(
        &self,
        file_commit_edges: &[(String, String)],
        cochange_edges: &[(String, String, i64, f64)],
    ) -> Result<()>;
    fn vss_loaded(&self) -> bool;
    fn fts_loaded(&self) -> bool;
    fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<SearchRow>>;
    fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<FtsRow>>;
    fn cochange_neighbors(&self, seeds: &[String], limit: usize) -> Result<Vec<String>>;
    fn chunks_for_files(
        &self,
        file_paths: &[String],
        limit_per_file: usize,
    ) -> Result<Vec<SearchRow>>;
    fn symbols_in_range(
        &self,
        file_path: &str,
        start_byte: i64,
        end_byte: i64,
    ) -> Result<Vec<SymbolRecord>>;
}

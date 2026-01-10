use std::collections::BTreeMap;
use std::path::PathBuf;

use async_trait::async_trait;
use eyre::Result;

use crate::db::{ChunkRecord, FtsRow, GraphData, SearchRow, SymbolRecord};

#[derive(Clone, Debug)]
pub struct DependencyEdge {
    pub src_path: String,
    pub dst_path: String,
    pub reference_count: i64,
}

#[derive(Clone, Debug)]
pub struct CochangeEdge {
    pub src_path: String,
    pub dst_path: String,
    pub weight: f64,
    #[allow(dead_code)]
    pub commit_count: i64,
}

#[async_trait]
pub trait Repository: Send + Sync {
    async fn load_existing_hashes(&self) -> Result<BTreeMap<PathBuf, [u8; 32]>>;
    async fn find_chunk_id(
        &self,
        file_path: &str,
        start_byte: usize,
        end_byte: usize,
        kind: &str,
        chunk_hash: [u8; 32],
    ) -> Result<Option<String>>;
    async fn upsert_file_metadata(
        &self,
        file_path: &str,
        file_size: u64,
        content_hash: [u8; 32],
        primary_language: Option<String>,
    ) -> Result<()>;
    async fn ensure_file_row(
        &self,
        file_path: &str,
        file_size: u64,
        primary_language: Option<String>,
    ) -> Result<()>;
    async fn upsert_chunks_with_embeddings(
        &self,
        records: &[ChunkRecord],
        embeddings: &[Vec<f32>],
    ) -> Result<()>;
    async fn update_chunk_without_embedding(&self, record: &ChunkRecord) -> Result<()>;
    async fn delete_missing_chunks(&self, file_path: &str, keep_ids: &[String]) -> Result<()>;
    async fn upsert_file_graph(
        &self,
        file_path: &str,
        file_size: u64,
        content_hash: [u8; 32],
        language: &str,
        primary_language: Option<String>,
        graph: GraphData,
    ) -> Result<()>;
    async fn delete_file(&self, file_path: &str) -> Result<()>;
    async fn list_files(&self) -> Result<Vec<String>>;
    async fn file_primary_language(&self, file_path: &str) -> Result<Option<String>>;
    async fn upsert_history_edges(
        &self,
        file_commit_edges: &[(String, String)],
        cochange_edges: &[(String, String, i64, f64)],
    ) -> Result<()>;
    async fn upsert_commit_issue_edges(
        &self,
        commit_issue_edges: &[(String, String)],
    ) -> Result<()>;
    fn vss_loaded(&self) -> bool;
    fn fts_loaded(&self) -> bool;
    async fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<SearchRow>>;
    async fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<FtsRow>>;
    async fn cochange_neighbors(&self, seeds: &[String], limit: usize) -> Result<Vec<String>>;
    async fn cochange_partners(&self, file_path: &str, limit: usize) -> Result<Vec<(String, f64)>>;
    async fn file_commit_count(&self, file_path: &str) -> Result<i64>;
    async fn update_file_dependency_edges(&self, file_path: &str) -> Result<()>;
    async fn file_dependency_pagerank(&self, limit: usize) -> Result<Vec<(String, f64)>>;
    async fn list_dependency_edges(&self) -> Result<Vec<DependencyEdge>>;
    async fn list_cochange_edges(&self) -> Result<Vec<CochangeEdge>>;
    async fn chunks_for_files(
        &self,
        file_paths: &[String],
        limit_per_file: usize,
    ) -> Result<Vec<SearchRow>>;
    async fn symbols_in_range(
        &self,
        file_path: &str,
        start_byte: i64,
        end_byte: i64,
    ) -> Result<Vec<SymbolRecord>>;
}

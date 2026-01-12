use std::collections::HashMap;

use eyre::{Result, eyre};
use tracing::warn;

use std::path::Path;

use crate::db::{FtsRow, SearchRow};
use crate::repository::Repository;
use niblits::Tokenizer as ChunkTokenizer;
use seasoning::{
    EmbeddingInput, EmbeddingProvider, RerankDocument, RerankQuery, RerankingProvider,
};
use tiktoken_rs::{cl100k_base, o200k_base, p50k_base, p50k_edit, r50k_base};
use tokenizers::Tokenizer as HfTokenizer;

pub type ProgressCallback = dyn Fn(&'static str) + Send + Sync;

#[derive(Clone)]
pub struct SearchResult {
    pub id: String,
    pub file_path: String,
    pub start_byte: i64,
    pub end_byte: i64,
    pub chunk_hash: [u8; 32],
    pub start_line: i64,
    pub end_line: i64,
    pub text: String,
    pub score: f64,
    pub vector_score: Option<f64>,
    pub fts_score: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub limit: usize,
    pub hybrid_weight: f32,
    pub min_score: f64,
    pub path_prefixes: Vec<String>,
    pub file_exts: Vec<String>,
}

impl SearchConfig {
    pub fn new(limit: usize, hybrid_weight: f32) -> Self {
        Self {
            limit,
            hybrid_weight,
            min_score: 0.25,
            path_prefixes: Vec::new(),
            file_exts: Vec::new(),
        }
    }

    fn normalize(mut self) -> Self {
        self.path_prefixes = self
            .path_prefixes
            .into_iter()
            .map(|item| item.trim().to_string())
            .filter(|item| !item.is_empty())
            .collect();
        self.file_exts = self
            .file_exts
            .into_iter()
            .map(|item| item.trim().trim_start_matches('.').to_ascii_lowercase())
            .filter(|item| !item.is_empty())
            .collect();
        self
    }

    fn matches_path(&self, file_path: &str) -> bool {
        if !self.path_prefixes.is_empty()
            && !self
                .path_prefixes
                .iter()
                .any(|prefix| file_path.starts_with(prefix))
        {
            return false;
        }
        if !self.file_exts.is_empty() {
            let ext = Path::new(file_path)
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if !self.file_exts.iter().any(|item| item == &ext) {
                return false;
            }
        }
        true
    }
}

pub struct SearchContext<'a> {
    pub db: &'a dyn Repository,
    pub embedder: &'a dyn EmbeddingProvider,
    pub reranker: &'a dyn RerankingProvider,
    pub tokenizer: &'a ChunkTokenizer,
    pub progress: Option<&'a ProgressCallback>,
}

pub async fn search(
    ctx: &SearchContext<'_>,
    query: &str,
    config: SearchConfig,
) -> Result<Vec<SearchResult>> {
    search_internal(ctx, query, config, true).await
}

pub async fn search_without_rerank(
    ctx: &SearchContext<'_>,
    query: &str,
    config: SearchConfig,
) -> Result<Vec<SearchResult>> {
    search_internal(ctx, query, config, false).await
}

async fn search_internal(
    ctx: &SearchContext<'_>,
    query: &str,
    config: SearchConfig,
    rerank: bool,
) -> Result<Vec<SearchResult>> {
    let config = config.normalize();
    report_progress(ctx.progress, "preparing query");
    let mut combined = search_single(ctx, query, config.limit, config.hybrid_weight)
        .await?
        .into_iter()
        .filter(|result| config.matches_path(&result.file_path))
        .collect::<Vec<_>>();
    combined.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    combined.truncate(config.limit);
    if rerank {
        let documents: Vec<String> = combined.iter().map(|item| item.text.clone()).collect();
        report_progress(ctx.progress, "reranking results");
        let token_count = count_tokens(ctx.tokenizer, query)?;
        let query = RerankQuery {
            text: query.to_string(),
            token_count,
        };
        let documents = documents
            .iter()
            .map(|text| {
                Ok(RerankDocument {
                    token_count: count_tokens(ctx.tokenizer, text)?,
                    text: text.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let scores = ctx
            .reranker
            .rerank(&query, &documents)
            .await
            .map_err(|err| eyre!("reranking failed: {err}"))?;
        if scores.len() != combined.len() {
            return Err(eyre!(
                "reranker returned {} scores for {} results",
                scores.len(),
                combined.len()
            ));
        }
        for (result, score) in combined.iter_mut().zip(scores.iter()) {
            result.score = *score;
        }
        combined.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    combined.retain(|item| item.score >= config.min_score);
    Ok(combined)
}

async fn search_single(
    ctx: &SearchContext<'_>,
    query: &str,
    limit: usize,
    hybrid_weight: f32,
) -> Result<Vec<SearchResult>> {
    let vector_future = async {
        if !ctx.db.vss_loaded() {
            return Ok::<Option<Vec<SearchRow>>, eyre::Report>(None);
        }
        report_progress(ctx.progress, "embedding query");
        let token_count = count_tokens(ctx.tokenizer, query)?;
        let input = EmbeddingInput {
            text: query.to_string(),
            token_count,
        };
        let output = ctx.embedder.embed(&[input]).await?;
        let mut embeddings = output.embeddings;
        if embeddings.is_empty() {
            return Err(eyre!("embedder returned no embeddings for query"));
        }
        report_progress(ctx.progress, "searching vector index");
        Ok(Some(ctx.db.search(&embeddings.remove(0), limit).await?))
    };

    let fts_future = async {
        if !ctx.db.fts_loaded() {
            return Ok::<Option<Vec<FtsRow>>, eyre::Report>(None);
        }
        report_progress(ctx.progress, "searching full-text");
        match ctx.db.search_fts(query, limit).await {
            Ok(rows) => Ok(Some(rows)),
            Err(err) => {
                warn!("full-text search unavailable: {err}");
                Ok(None)
            }
        }
    };

    let (fts_results, vector_results) = tokio::join!(fts_future, vector_future);
    let fts_results = fts_results?;
    let vector_results = vector_results?;

    merge_results(vector_results, fts_results, limit, hybrid_weight)
}

pub(crate) fn count_tokens(tokenizer: &ChunkTokenizer, text: &str) -> Result<usize> {
    match tokenizer {
        ChunkTokenizer::Characters => Ok(text.chars().count()),
        ChunkTokenizer::Tiktoken(encoding) => {
            let encoder = match encoding.as_str() {
                "cl100k_base" => cl100k_base(),
                "p50k_base" => p50k_base(),
                "p50k_edit" => p50k_edit(),
                "r50k_base" => r50k_base(),
                "o200k_base" => o200k_base(),
                other => {
                    return Err(eyre::eyre!("Unknown tiktoken encoding: {other}"));
                }
            }
            .map_err(|err| eyre::eyre!("Failed to create tiktoken: {err}"))?;
            Ok(encoder.encode_ordinary(text).len())
        }
        ChunkTokenizer::PreloadedTiktoken(encoder) => Ok(encoder.encode_ordinary(text).len()),
        ChunkTokenizer::HuggingFace(model_id) => {
            let tokenizer = HfTokenizer::from_pretrained(model_id, None)
                .map_err(|err| eyre::eyre!("Failed to load HF tokenizer {model_id}: {err}"))?;
            tokenizer
                .encode(text, false)
                .map(|encoding| encoding.len())
                .map_err(|err| eyre::eyre!("tokenizer encode failed: {err}"))
        }
        ChunkTokenizer::PreloadedHuggingFace(tokenizer) => tokenizer
            .encode(text, false)
            .map(|encoding| encoding.len())
            .map_err(|err| eyre::eyre!("tokenizer encode failed: {err}")),
    }
}

fn report_progress(progress: Option<&ProgressCallback>, message: &'static str) {
    if let Some(callback) = progress {
        callback(message);
    }
}

struct PartialResult {
    id: String,
    file_path: String,
    start_byte: i64,
    end_byte: i64,
    chunk_hash: [u8; 32],
    start_line: i64,
    end_line: i64,
    text: String,
    vector_score: Option<f64>,
    fts_score: Option<f64>,
}

fn merge_results(
    vector_results: Option<Vec<SearchRow>>,
    fts_results: Option<Vec<FtsRow>>,
    limit: usize,
    hybrid_weight: f32,
) -> Result<Vec<SearchResult>> {
    let mut by_id: HashMap<String, PartialResult> = HashMap::new();

    if let Some(results) = vector_results {
        for row in results {
            let score = vector_score(row.distance);
            by_id
                .entry(row.id.clone())
                .or_insert_with(|| PartialResult {
                    id: row.id,
                    file_path: row.file_path,
                    start_byte: row.start_byte,
                    end_byte: row.end_byte,
                    chunk_hash: row.chunk_hash,
                    start_line: row.start_line,
                    end_line: row.end_line,
                    text: row.text,
                    vector_score: Some(score),
                    fts_score: None,
                })
                .vector_score = Some(score);
        }
    }

    if let Some(results) = fts_results {
        let max_score = results.iter().map(|row| row.score).fold(0.0_f64, f64::max);
        for row in results {
            let normalized = if max_score > 0.0 {
                row.score / max_score
            } else {
                0.0
            };
            by_id
                .entry(row.id.clone())
                .or_insert_with(|| PartialResult {
                    id: row.id,
                    file_path: row.file_path,
                    start_byte: row.start_byte,
                    end_byte: row.end_byte,
                    chunk_hash: row.chunk_hash,
                    start_line: row.start_line,
                    end_line: row.end_line,
                    text: row.text,
                    vector_score: None,
                    fts_score: Some(normalized),
                })
                .fts_score = Some(normalized);
        }
    }

    let mut combined = Vec::with_capacity(by_id.len());
    let weight = hybrid_weight.clamp(0.0, 1.0) as f64;
    for entry in by_id.into_values() {
        let score = match (entry.vector_score, entry.fts_score) {
            (Some(v), Some(t)) => weight * v + (1.0 - weight) * t,
            (Some(v), None) => v,
            (None, Some(t)) => t,
            (None, None) => 0.0,
        };
        combined.push(SearchResult {
            id: entry.id,
            file_path: entry.file_path,
            start_byte: entry.start_byte,
            end_byte: entry.end_byte,
            chunk_hash: entry.chunk_hash,
            start_line: entry.start_line,
            end_line: entry.end_line,
            text: entry.text,
            score,
            vector_score: entry.vector_score,
            fts_score: entry.fts_score,
        });
    }

    combined.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    combined.truncate(limit);
    Ok(combined)
}

fn vector_score(distance: f64) -> f64 {
    (1.0 - (distance / 2.0)).clamp(0.0, 1.0)
}

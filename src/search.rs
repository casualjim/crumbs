use std::collections::HashMap;

use eyre::{Result, eyre};
use tracing::warn;

use std::path::Path;

use crate::db::{FtsRow, SearchRow};
use crate::embedding::{EmbeddingInput, EmbeddingProvider};
use crate::reranker::RerankingProvider;
use crate::repository::Repository;

#[derive(Clone)]
pub struct SearchResult {
    pub file_path: String,
    pub start_byte: i64,
    pub end_byte: i64,
    pub text: String,
    pub score: f64,
    pub vector_score: Option<f64>,
    pub fts_score: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub limit: usize,
    pub hybrid_weight: f32,
    pub path_prefixes: Vec<String>,
    pub file_exts: Vec<String>,
    pub decompose: bool,
    pub rerank: bool,
}

impl SearchConfig {
    pub fn new(limit: usize, hybrid_weight: f32) -> Self {
        Self {
            limit,
            hybrid_weight,
            path_prefixes: Vec::new(),
            file_exts: Vec::new(),
            decompose: false,
            rerank: false,
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

pub async fn search(
    db: &dyn Repository,
    embedder: &dyn EmbeddingProvider,
    reranker: Option<&dyn RerankingProvider>,
    query: &str,
    config: SearchConfig,
) -> Result<Vec<SearchResult>> {
    let config = config.normalize();
    let query_text = query;
    let queries = if config.decompose {
        split_query(query)
    } else {
        vec![query.to_string()]
    };
    let mut combined: HashMap<String, SearchResult> = HashMap::new();

    for query in queries {
        let results = search_single(
            db,
            embedder,
            &query,
            config.limit,
            config.hybrid_weight,
        )
        .await?;
        for result in results {
            if !config.matches_path(&result.file_path) {
                continue;
            }
            let id = format!("{}:{}-{}", result.file_path, result.start_byte, result.end_byte);
            match combined.get_mut(&id) {
                Some(existing) => {
                    if result.score > existing.score {
                        *existing = result;
                    }
                }
                None => {
                    combined.insert(id, result);
                }
            }
        }
    }

    let mut combined = combined.into_values().collect::<Vec<_>>();
    combined.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    combined.truncate(config.limit);
    if config.rerank {
        if let Some(reranker) = reranker {
            let documents: Vec<String> = combined.iter().map(|item| item.text.clone()).collect();
            match reranker.rerank(query_text, &documents).await {
                Ok(scores) if scores.len() == combined.len() => {
                    for (result, score) in combined.iter_mut().zip(scores.iter()) {
                        result.score = *score;
                    }
                    combined.sort_by(|a, b| {
                        b.score
                            .partial_cmp(&a.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                Ok(_) => {
                    warn!("Reranker returned unexpected score count; using hybrid scores");
                }
                Err(err) => {
                    warn!("Reranking failed, using hybrid scores: {err}");
                }
            }
        } else {
            warn!("Reranking requested but no reranker configured");
        }
    }
    Ok(combined)
}

async fn search_single(
    db: &dyn Repository,
    embedder: &dyn EmbeddingProvider,
    query: &str,
    limit: usize,
    hybrid_weight: f32,
) -> Result<Vec<SearchResult>> {
    let vector_results = if db.vss_loaded() {
        let input = EmbeddingInput {
            text: query.to_string(),
            token_count: None,
        };
        let output = embedder.embed(&[input]).await?;
        let mut embeddings = output.embeddings;
        if embeddings.is_empty() {
            return Err(eyre!("embedder returned no embeddings for query"));
        }
        Some(db.search(&embeddings.remove(0), limit)?)
    } else {
        None
    };

    let fts_results = if db.fts_loaded() {
        match db.search_fts(query, limit) {
            Ok(rows) => Some(rows),
            Err(err) => {
                warn!("full-text search unavailable: {err}");
                None
            }
        }
    } else {
        None
    };

    merge_results(vector_results, fts_results, limit, hybrid_weight)
}

struct PartialResult {
    file_path: String,
    start_byte: i64,
    end_byte: i64,
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
                    file_path: row.file_path,
                    start_byte: row.start_byte,
                    end_byte: row.end_byte,
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
                    file_path: row.file_path,
                    start_byte: row.start_byte,
                    end_byte: row.end_byte,
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
            file_path: entry.file_path,
            start_byte: entry.start_byte,
            end_byte: entry.end_byte,
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

fn split_query(query: &str) -> Vec<String> {
    let cleaned = query
        .replace(';', " and ")
        .replace('\n', " and ")
        .replace('\t', " ");
    if !cleaned.contains(" and ") {
        return vec![query.trim().to_string()];
    }
    let mut parts = Vec::new();
    for part in cleaned.split(" and ") {
        let trimmed = part.trim();
        if trimmed.len() > 2 {
            parts.push(trimmed.to_string());
        }
    }
    if parts.is_empty() {
        vec![query.trim().to_string()]
    } else {
        parts.truncate(4);
        parts
    }
}

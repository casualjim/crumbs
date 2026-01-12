#![allow(dead_code)]

use async_trait::async_trait;
use eyre::{Result, eyre};
use niblits::Tokenizer;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use super::{Arena, AssemblyContext, Handle};
use crate::config::AppConfig;
use crate::db::SearchRow;
use crate::issues::Issue;
use crate::search::{SearchConfig, search_without_rerank};
use crate::topology::TopologySnapshot;

/// Input query for assembly.
pub struct QueryInput {
    pub text: String,
    pub issue_context: Option<IssueContext>,
}

#[derive(Clone)]
pub struct IssueContext {
    pub issue: Issue,
    pub dependency_issues: Vec<Issue>,
    pub related_issues: Vec<Issue>,
    pub duplicate_issues: Vec<Issue>,
}

/// Initial candidate pool with evidence from multiple sources.
pub struct UnifiedCandidateSet {
    pub query: String,
    pub candidates: Vec<UnifiedCandidate>,
    pub warnings: Vec<String>,
}

/// Selected and scored blocks ready for budgeting.
pub struct SelectedBlocks {
    pub blocks: Vec<ContextBlock>,
    pub warnings: Vec<String>,
}

/// Budgeted/merged blocks ready for prompt assembly.
pub struct BudgetedBlocks {
    pub blocks: Vec<ContextBlock>,
    pub warnings: Vec<String>,
}

/// Final assembled context payload.
pub struct AssembledContext {
    pub blocks: Vec<ContextBlock>,
    pub warnings: Vec<String>,
}

/// A candidate chunk with evidence from multiple sources.
#[derive(Clone)]
pub struct UnifiedCandidate {
    pub id: String,
    pub file_path: String,
    pub start_byte: i64,
    pub end_byte: i64,
    pub chunk_hash: [u8; 32],
    pub start_line: i64,
    pub end_line: i64,
    pub text: String,

    pub semantic_score: Option<f64>,
    pub topology_distance: Option<usize>,
    pub topology_weight: Option<f64>,
    pub cochange_weight: Option<f64>,
    pub recency_score: Option<f64>,
    pub centrality_score: Option<f64>,

    pub label_match_count: usize,
    pub dependency_match: bool,
    pub relates_to_match: bool,
    pub duplicate_origin: bool,

    pub feature_score: f64,
    pub rerank_score: Option<f64>,
    pub final_score: f64,

    pub sources: Vec<CandidateSource>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CandidateSource {
    Explicit,
    Pinned,
    SemanticRetrieval,
    TopologyNeighbor { depth: usize },
    CochangeExpansion,
    DependencyIssue { issue_id: String },
    RelatedIssue { issue_id: String },
    DuplicateIssue { issue_id: String },
}

/// A context block carried forward for assembly.
#[derive(Clone)]
pub struct ContextBlock {
    pub id: String,
    pub file_path: String,
    pub start_byte: i64,
    pub end_byte: i64,
    pub chunk_hash: [u8; 32],
    pub start_line: i64,
    pub end_line: i64,
    pub text: String,
    pub score: f64,
    pub source: CandidateSource,
    pub sources: Vec<CandidateSource>,
}

#[async_trait(?Send)]
pub trait RetrieveCandidates: Send + Sync {
    async fn retrieve(
        &self,
        ctx: &AssemblyContext<'_>,
        arena: &mut Arena,
        input: Handle<QueryInput>,
    ) -> Result<Handle<UnifiedCandidateSet>>;
}

#[async_trait(?Send)]
pub trait ScoreAndSelect: Send + Sync {
    async fn score_and_select(
        &self,
        ctx: &AssemblyContext<'_>,
        arena: &mut Arena,
        input: Handle<UnifiedCandidateSet>,
    ) -> Result<Handle<SelectedBlocks>>;
}

#[async_trait(?Send)]
pub trait BudgetAndMerge: Send + Sync {
    async fn budget(
        &self,
        ctx: &AssemblyContext<'_>,
        arena: &mut Arena,
        input: Handle<SelectedBlocks>,
    ) -> Result<Handle<BudgetedBlocks>>;
}

#[async_trait(?Send)]
pub trait AssembleContext: Send + Sync {
    async fn assemble(
        &self,
        ctx: &AssemblyContext<'_>,
        arena: &mut Arena,
        input: Handle<BudgetedBlocks>,
    ) -> Result<Handle<AssembledContext>>;
}

/// Concrete pipeline wiring that connects independent stages.
pub struct AssemblyPipeline<R, S, B, F> {
    retrieve: R,
    score: S,
    budget: B,
    assemble: F,
}

impl<R, S, B, F> AssemblyPipeline<R, S, B, F> {
    pub fn new(retrieve: R, score: S, budget: B, assemble: F) -> Self {
        Self {
            retrieve,
            score,
            budget,
            assemble,
        }
    }
}

impl<R, S, B, F> AssemblyPipeline<R, S, B, F>
where
    R: RetrieveCandidates,
    S: ScoreAndSelect,
    B: BudgetAndMerge,
    F: AssembleContext,
{
    pub async fn run(
        &self,
        ctx: &AssemblyContext<'_>,
        arena: &mut Arena,
        input: Handle<QueryInput>,
    ) -> Result<Handle<AssembledContext>> {
        let candidates = self.retrieve.retrieve(ctx, arena, input).await?;
        let selected = self.score.score_and_select(ctx, arena, candidates).await?;
        let budgeted = self.budget.budget(ctx, arena, selected).await?;
        self.assemble.assemble(ctx, arena, budgeted).await
    }

    pub async fn run_with_progress<Notify>(
        &self,
        ctx: &AssemblyContext<'_>,
        arena: &mut Arena,
        input: Handle<QueryInput>,
        mut notify: Notify,
    ) -> Result<Handle<AssembledContext>>
    where
        Notify: FnMut(&'static str),
    {
        notify("retrieving candidates");
        let candidates = self.retrieve.retrieve(ctx, arena, input).await?;
        notify("scoring and selecting");
        let selected = self.score.score_and_select(ctx, arena, candidates).await?;
        notify("budgeting context");
        let budgeted = self.budget.budget(ctx, arena, selected).await?;
        notify("assembling prompt");
        self.assemble.assemble(ctx, arena, budgeted).await
    }
}

#[derive(Clone, Debug)]
pub struct UnifiedRetrieveConfig {
    pub semantic_limit: usize,
    pub topology_depth: usize,
    pub topology_limit: usize,
    pub cochange_limit: usize,
    pub dependency_issue_limit: usize,
    pub related_issue_limit: usize,
    pub duplicate_limit: usize,
    pub per_file_limit: usize,
}

impl Default for UnifiedRetrieveConfig {
    fn default() -> Self {
        Self {
            semantic_limit: 50,
            topology_depth: 2,
            topology_limit: 30,
            cochange_limit: 20,
            dependency_issue_limit: 10,
            related_issue_limit: 10,
            duplicate_limit: 3,
            per_file_limit: 5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScoringConfig {
    pub semantic_weight: f64,
    pub topology_weight: f64,
    pub cochange_weight: f64,
    pub recency_weight: f64,
    pub centrality_weight: f64,
    pub metadata_weight: f64,

    pub topology_distance_decay: f64,
    pub label_match_boost: f64,
    pub dependency_boost: f64,
    pub relates_to_boost: f64,
    pub duplicate_boost: f64,
    pub multi_source_bonus: f64,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            semantic_weight: 0.40,
            topology_weight: 0.25,
            cochange_weight: 0.15,
            recency_weight: 0.10,
            centrality_weight: 0.05,
            metadata_weight: 0.05,
            topology_distance_decay: 1.0,
            label_match_boost: 0.05,
            dependency_boost: 0.10,
            relates_to_boost: 0.08,
            duplicate_boost: 0.05,
            multi_source_bonus: 0.10,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SelectionConfig {
    pub max_blocks: usize,
    pub max_per_file: usize,
    pub diversity_lambda: f64,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            max_blocks: 40,
            max_per_file: 5,
            diversity_lambda: 0.7,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RerankConfig {
    pub top_n: usize,
    pub blend: f64,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            top_n: 30,
            blend: 0.7,
        }
    }
}

/// Unified retrieval stage: gather candidates from semantic, topology, cochange, and issue metadata.
pub struct UnifiedRetrieve {
    pub config: UnifiedRetrieveConfig,
    pub progress: Option<Arc<dyn Fn(&'static str) + Send + Sync>>,
}

#[async_trait(?Send)]
impl RetrieveCandidates for UnifiedRetrieve {
    async fn retrieve(
        &self,
        ctx: &AssemblyContext<'_>,
        arena: &mut Arena,
        input: Handle<QueryInput>,
    ) -> Result<Handle<UnifiedCandidateSet>> {
        let mut warnings = Vec::new();
        let mut candidates: HashMap<[u8; 32], UnifiedCandidate> = HashMap::new();
        let mut seed_paths: HashSet<String> = HashSet::new();

        let per_file_limit = self.config.per_file_limit.max(1);
        if !ctx.selection.explicit_includes.is_empty() {
            let include_paths =
                resolve_input_paths(ctx.repo_path, &ctx.selection.explicit_includes);
            let rows = ctx
                .db
                .chunks_for_files(&include_paths, per_file_limit)
                .await?;
            let found: HashSet<String> = rows.iter().map(|row| row.file_path.clone()).collect();
            for path in &include_paths {
                if !found.contains(path) {
                    warnings.push(format!(
                        "explicit include not found: {}",
                        normalize_path(ctx.repo_path, path)
                    ));
                }
            }
            for row in rows {
                seed_paths.insert(row.file_path.clone());
                insert_candidate(
                    &mut candidates,
                    build_candidate(row, CandidateSource::Explicit, Some(1.0)),
                );
            }
        }

        if !ctx.selection.pinned_items.is_empty() {
            let pinned_paths = resolve_input_paths(ctx.repo_path, &ctx.selection.pinned_items);
            let rows = ctx
                .db
                .chunks_for_files(&pinned_paths, per_file_limit)
                .await?;
            let found: HashSet<String> = rows.iter().map(|row| row.file_path.clone()).collect();
            for path in &pinned_paths {
                if !found.contains(path) {
                    warnings.push(format!(
                        "pinned item not found: {}",
                        normalize_path(ctx.repo_path, path)
                    ));
                }
            }
            for row in rows {
                seed_paths.insert(row.file_path.clone());
                insert_candidate(
                    &mut candidates,
                    build_candidate(row, CandidateSource::Pinned, Some(1.0)),
                );
            }
        }

        let query = arena.get(input);
        if self.config.semantic_limit > 0 && !query.text.trim().is_empty() {
            if let Some(progress) = &self.progress {
                progress("loading tokenizer");
            }
            let embedder = ctx
                .embedder
                .ok_or_else(|| eyre!("embedding provider required for retrieval"))?;
            let tokenizer = crate::parse_tokenizer(&ctx.config.embedding.tokenizer)?;
            let progress = self.progress.as_ref().map(|progress| progress.as_ref());
            let search_ctx = crate::search::SearchContext {
                db: ctx.db,
                embedder,
                reranker: ctx.reranker,
                tokenizer: &tokenizer,
                progress,
            };
            let mut search_config =
                SearchConfig::new(self.config.semantic_limit, ctx.config.search.hybrid_weight);
            search_config.min_score = ctx.config.search.min_score;
            search_config.path_prefixes = ctx.config.search.path_prefixes.clone();
            search_config.file_exts = ctx.config.search.file_exts.clone();
            let results = search_without_rerank(&search_ctx, &query.text, search_config).await?;
            for result in results
                .into_iter()
                .filter(|result| is_allowed_path(ctx.repo_path, &ctx.selection, &result.file_path))
            {
                seed_paths.insert(result.file_path.clone());
                insert_candidate(
                    &mut candidates,
                    UnifiedCandidate {
                        id: result.id,
                        file_path: result.file_path,
                        start_byte: result.start_byte,
                        end_byte: result.end_byte,
                        chunk_hash: result.chunk_hash,
                        start_line: result.start_line,
                        end_line: result.end_line,
                        text: result.text,
                        semantic_score: Some(result.score),
                        topology_distance: None,
                        topology_weight: None,
                        cochange_weight: None,
                        recency_score: None,
                        centrality_score: None,
                        label_match_count: 0,
                        dependency_match: false,
                        relates_to_match: false,
                        duplicate_origin: false,
                        feature_score: 0.0,
                        rerank_score: None,
                        final_score: 0.0,
                        sources: vec![CandidateSource::SemanticRetrieval],
                    },
                );
            }
        }

        let mut topology_seeds = HashSet::new();
        if let Some(issue_context) = &query.issue_context {
            collect_issue_seeds(
                ctx.repo_path,
                &issue_context.issue.affected_symbols,
                &mut topology_seeds,
            );
            for issue in &issue_context.dependency_issues {
                collect_issue_seeds(ctx.repo_path, &issue.affected_symbols, &mut topology_seeds);
            }
            for issue in &issue_context.related_issues {
                collect_issue_seeds(ctx.repo_path, &issue.affected_symbols, &mut topology_seeds);
            }
            for issue in &issue_context.duplicate_issues {
                collect_issue_seeds(ctx.repo_path, &issue.affected_symbols, &mut topology_seeds);
            }
        }
        for path in &seed_paths {
            topology_seeds.insert(normalize_seed_path(ctx.repo_path, path));
        }

        if self.config.topology_limit > 0 && !topology_seeds.is_empty() {
            if let Some(progress) = &self.progress {
                progress("loading topology snapshot");
            }
            let cwd = std::env::current_dir().unwrap_or_else(|_| ctx.repo_path.to_path_buf());
            let snapshot =
                TopologySnapshot::load_with_workspace(ctx.db, ctx.repo_path, &cwd).await?;
            let topology = select_topology_files(
                &snapshot,
                ctx.repo_path,
                &ctx.selection,
                &topology_seeds,
                self.config.topology_depth,
                self.config.topology_limit,
                &mut warnings,
            )?;
            if !topology.file_paths.is_empty() {
                let rows = ctx
                    .db
                    .chunks_for_files(&topology.file_paths, per_file_limit)
                    .await?;
                for row in rows {
                    seed_paths.insert(row.file_path.clone());
                    let weight = topology.weights.get(&row.file_path).copied();
                    let distance = topology.distances.get(&row.file_path).copied();
                    insert_candidate(
                        &mut candidates,
                        build_topology_candidate(row, weight, distance),
                    );
                }
            }
        }

        if let Some(issue_context) = &query.issue_context {
            let mut issue_candidates = Vec::new();
            issue_candidates.extend(issue_context.dependency_issues.iter().map(|issue| {
                (
                    issue.summary_query(),
                    CandidateSource::DependencyIssue {
                        issue_id: issue.id.clone(),
                    },
                )
            }));
            issue_candidates.extend(issue_context.related_issues.iter().map(|issue| {
                (
                    issue.summary_query(),
                    CandidateSource::RelatedIssue {
                        issue_id: issue.id.clone(),
                    },
                )
            }));
            issue_candidates.extend(issue_context.duplicate_issues.iter().map(|issue| {
                (
                    issue.summary_query(),
                    CandidateSource::DuplicateIssue {
                        issue_id: issue.id.clone(),
                    },
                )
            }));

            for (summary, source) in issue_candidates {
                if summary.trim().is_empty() {
                    continue;
                }
                let dependency_match = matches!(source, CandidateSource::DependencyIssue { .. });
                let relates_to_match = matches!(source, CandidateSource::RelatedIssue { .. });
                let duplicate_origin = matches!(source, CandidateSource::DuplicateIssue { .. });
                let limit = match source {
                    CandidateSource::DependencyIssue { .. } => self.config.dependency_issue_limit,
                    CandidateSource::RelatedIssue { .. } => self.config.related_issue_limit,
                    CandidateSource::DuplicateIssue { .. } => self.config.duplicate_limit,
                    _ => self.config.semantic_limit,
                };
                if limit == 0 {
                    continue;
                }
                let embedder = ctx
                    .embedder
                    .ok_or_else(|| eyre!("embedding provider required for retrieval"))?;
                let tokenizer = crate::parse_tokenizer(&ctx.config.embedding.tokenizer)?;
                let search_ctx = crate::search::SearchContext {
                    db: ctx.db,
                    embedder,
                    reranker: ctx.reranker,
                    tokenizer: &tokenizer,
                    progress: None,
                };

                let mut search_config = SearchConfig::new(limit, ctx.config.search.hybrid_weight);
                search_config.min_score = ctx.config.search.min_score;
                search_config.path_prefixes = ctx.config.search.path_prefixes.clone();
                search_config.file_exts = ctx.config.search.file_exts.clone();

                let results = search_without_rerank(&search_ctx, &summary, search_config).await?;
                for result in results.into_iter().filter(|result| {
                    is_allowed_path(ctx.repo_path, &ctx.selection, &result.file_path)
                }) {
                    seed_paths.insert(result.file_path.clone());
                    insert_candidate(
                        &mut candidates,
                        UnifiedCandidate {
                            id: result.id,
                            file_path: result.file_path,
                            start_byte: result.start_byte,
                            end_byte: result.end_byte,
                            chunk_hash: result.chunk_hash,
                            start_line: result.start_line,
                            end_line: result.end_line,
                            text: result.text,
                            semantic_score: Some(result.score),
                            topology_distance: None,
                            topology_weight: None,
                            cochange_weight: None,
                            recency_score: None,
                            centrality_score: None,
                            label_match_count: 0,
                            dependency_match,
                            relates_to_match,
                            duplicate_origin,
                            feature_score: 0.0,
                            rerank_score: None,
                            final_score: 0.0,
                            sources: vec![source.clone()],
                        },
                    );
                }
            }
        }

        if self.config.cochange_limit > 0 && !seed_paths.is_empty() {
            if let Some(progress) = &self.progress {
                progress("expanding cochange neighbors");
            }
            let mut cochange_weights: HashMap<String, f64> = HashMap::new();
            for seed in &seed_paths {
                let partners = ctx
                    .db
                    .cochange_partners(seed, self.config.cochange_limit)
                    .await?;
                for (path, weight) in partners {
                    *cochange_weights.entry(path).or_insert(0.0) += weight.max(0.0);
                }
            }

            let mut cochange_paths: Vec<(String, f64)> = cochange_weights
                .iter()
                .map(|(path, weight)| (path.clone(), *weight))
                .collect();
            cochange_paths.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            if cochange_paths.len() > self.config.cochange_limit {
                cochange_paths.truncate(self.config.cochange_limit);
            }
            let cochange_files: Vec<String> = cochange_paths
                .iter()
                .map(|(path, _)| path.clone())
                .collect();
            if !cochange_files.is_empty() {
                let rows = ctx
                    .db
                    .chunks_for_files(&cochange_files, per_file_limit)
                    .await?;
                for row in rows {
                    let weight = cochange_weights.get(&row.file_path).copied();
                    insert_candidate(&mut candidates, build_cochange_candidate(row, weight));
                }
            }
        }

        if let Some(issue_context) = &query.issue_context {
            apply_issue_metadata(ctx.repo_path, issue_context, &mut candidates);
        }

        Ok(arena.insert(UnifiedCandidateSet {
            query: query.text.clone(),
            candidates: candidates.into_values().collect(),
            warnings,
        }))
    }
}

pub struct DefaultScoreAndSelect {
    pub scoring: ScoringConfig,
    pub selection: SelectionConfig,
    pub rerank: RerankConfig,
}

#[async_trait(?Send)]
impl ScoreAndSelect for DefaultScoreAndSelect {
    async fn score_and_select(
        &self,
        ctx: &AssemblyContext<'_>,
        arena: &mut Arena,
        input: Handle<UnifiedCandidateSet>,
    ) -> Result<Handle<SelectedBlocks>> {
        let set = arena.get(input);
        let mut candidates = set.candidates.clone();
        let warnings = set.warnings.clone();
        let query = set.query.clone();

        normalize_weights(&mut candidates);
        if self.scoring.centrality_weight > 0.0 {
            apply_centrality_scores(ctx, &mut candidates).await?;
        }
        compute_feature_scores(&mut candidates, &self.scoring);

        if self.rerank.top_n > 0 && !candidates.is_empty() && !query.trim().is_empty() {
            rerank_candidates(
                &query,
                ctx,
                &mut candidates,
                self.rerank.top_n,
                self.rerank.blend,
            )
            .await?;
        } else {
            for candidate in &mut candidates {
                candidate.final_score = candidate.feature_score;
            }
        }

        candidates.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let selected = select_with_mmr(&candidates, &self.selection);
        let mut blocks = Vec::new();
        for candidate in selected {
            blocks.push(ContextBlock {
                id: candidate.id.clone(),
                file_path: candidate.file_path.clone(),
                start_byte: candidate.start_byte,
                end_byte: candidate.end_byte,
                chunk_hash: candidate.chunk_hash,
                start_line: candidate.start_line,
                end_line: candidate.end_line,
                text: candidate.text.clone(),
                score: candidate.final_score,
                source: primary_source(&candidate.sources),
                sources: candidate.sources.clone(),
            });
        }

        Ok(arena.insert(SelectedBlocks { blocks, warnings }))
    }
}

fn insert_candidate(
    candidates: &mut HashMap<[u8; 32], UnifiedCandidate>,
    candidate: UnifiedCandidate,
) {
    if let Some(existing) = candidates.get_mut(&candidate.chunk_hash) {
        merge_candidate(existing, candidate);
    } else {
        candidates.insert(candidate.chunk_hash, candidate);
    }
}

fn merge_candidate(existing: &mut UnifiedCandidate, incoming: UnifiedCandidate) {
    for source in incoming.sources {
        if !existing.sources.contains(&source) {
            existing.sources.push(source);
        }
    }
    if let Some(score) = incoming.semantic_score {
        existing.semantic_score = Some(
            existing
                .semantic_score
                .map(|current| current.max(score))
                .unwrap_or(score),
        );
    }
    if let Some(weight) = incoming.topology_weight {
        existing.topology_weight = Some(
            existing
                .topology_weight
                .map(|current| current.max(weight))
                .unwrap_or(weight),
        );
    }
    if let Some(distance) = incoming.topology_distance {
        existing.topology_distance = Some(
            existing
                .topology_distance
                .map(|current| current.min(distance))
                .unwrap_or(distance),
        );
    }
    if let Some(weight) = incoming.cochange_weight {
        existing.cochange_weight = Some(
            existing
                .cochange_weight
                .map(|current| current.max(weight))
                .unwrap_or(weight),
        );
    }
    if let Some(score) = incoming.recency_score {
        existing.recency_score = Some(
            existing
                .recency_score
                .map(|current| current.max(score))
                .unwrap_or(score),
        );
    }
    if let Some(score) = incoming.centrality_score {
        existing.centrality_score = Some(
            existing
                .centrality_score
                .map(|current| current.max(score))
                .unwrap_or(score),
        );
    }
    existing.label_match_count = existing.label_match_count.max(incoming.label_match_count);
    existing.dependency_match |= incoming.dependency_match;
    existing.relates_to_match |= incoming.relates_to_match;
    existing.duplicate_origin |= incoming.duplicate_origin;
}

fn build_candidate(
    row: SearchRow,
    source: CandidateSource,
    semantic_score: Option<f64>,
) -> UnifiedCandidate {
    UnifiedCandidate {
        id: row.id,
        file_path: row.file_path,
        start_byte: row.start_byte,
        end_byte: row.end_byte,
        chunk_hash: row.chunk_hash,
        start_line: row.start_line,
        end_line: row.end_line,
        text: row.text,
        semantic_score,
        topology_distance: None,
        topology_weight: None,
        cochange_weight: None,
        recency_score: None,
        centrality_score: None,
        label_match_count: 0,
        dependency_match: false,
        relates_to_match: false,
        duplicate_origin: false,
        feature_score: 0.0,
        rerank_score: None,
        final_score: 0.0,
        sources: vec![source],
    }
}

fn build_topology_candidate(
    row: SearchRow,
    weight: Option<f64>,
    distance: Option<usize>,
) -> UnifiedCandidate {
    let depth = distance.unwrap_or(0);
    let mut candidate = build_candidate(row, CandidateSource::TopologyNeighbor { depth }, None);
    candidate.topology_weight = weight;
    candidate.topology_distance = Some(depth);
    candidate
}

fn build_cochange_candidate(row: SearchRow, weight: Option<f64>) -> UnifiedCandidate {
    let mut candidate = build_candidate(row, CandidateSource::CochangeExpansion, None);
    candidate.cochange_weight = weight;
    candidate
}

fn normalize_seed_path(repo_root: &Path, raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let path = Path::new(trimmed);
    let full = if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo_root.join(path)
    };
    let full_str = full.to_string_lossy();
    normalize_path(repo_root, full_str.as_ref())
}

fn collect_issue_seeds(repo_root: &Path, symbols: &[String], output: &mut HashSet<String>) {
    for symbol in symbols {
        let normalized = normalize_seed_path(repo_root, symbol);
        if !normalized.is_empty() {
            output.insert(normalized);
        }
    }
}

struct TopologySelection {
    file_paths: Vec<String>,
    weights: HashMap<String, f64>,
    distances: HashMap<String, usize>,
}

fn select_topology_files(
    snapshot: &TopologySnapshot,
    repo_root: &Path,
    selection: &super::SelectionOptions,
    seeds: &HashSet<String>,
    depth: usize,
    limit: usize,
    warnings: &mut Vec<String>,
) -> Result<TopologySelection> {
    if limit == 0 {
        return Ok(TopologySelection {
            file_paths: Vec::new(),
            weights: HashMap::new(),
            distances: HashMap::new(),
        });
    }

    let mut seed_paths = Vec::new();
    for seed in seeds {
        if !snapshot.has_path(seed) {
            warnings.push(format!("seed not found in index: {}", seed));
            continue;
        }
        if !is_allowed_path(repo_root, selection, seed) {
            warnings.push(format!("seed outside scope/excludes: {}", seed));
            continue;
        }
        seed_paths.push(seed.clone());
    }

    if seed_paths.is_empty() {
        return Ok(TopologySelection {
            file_paths: Vec::new(),
            weights: HashMap::new(),
            distances: HashMap::new(),
        });
    }

    let seed_set: HashSet<String> = seed_paths.iter().cloned().collect();
    let mut neighbor_weights: HashMap<String, f64> = HashMap::new();
    let mut neighbor_distances: HashMap<String, usize> = HashMap::new();

    if depth > 0 {
        for seed in &seed_paths {
            let neighbors = snapshot.star_neighborhood(seed, depth)?;
            for neighbor in neighbors {
                if seed_set.contains(&neighbor.path) {
                    continue;
                }
                if !is_allowed_path(repo_root, selection, &neighbor.path) {
                    continue;
                }
                let entry = neighbor_weights.entry(neighbor.path.clone()).or_insert(0.0);
                *entry += neighbor.total_weight.max(0.0);
                let distance_entry = neighbor_distances
                    .entry(neighbor.path.clone())
                    .or_insert(neighbor.distance);
                if neighbor.distance < *distance_entry {
                    *distance_entry = neighbor.distance;
                }
            }
        }
    }

    let mut neighbors: Vec<(String, f64, usize)> = neighbor_weights
        .iter()
        .map(|(path, weight)| {
            let distance = neighbor_distances.get(path).copied().unwrap_or(usize::MAX);
            (path.clone(), *weight, distance)
        })
        .collect();
    neighbors.sort_by(|a, b| {
        a.2.cmp(&b.2)
            .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut file_paths = seed_paths;
    file_paths.sort();
    file_paths.dedup();
    if file_paths.len() > limit {
        warnings.push(format!(
            "seed count {} exceeds limit {}; neighbors omitted",
            file_paths.len(),
            limit
        ));
    } else {
        for (path, _weight, _distance) in neighbors {
            if file_paths.len() >= limit {
                break;
            }
            file_paths.push(path);
        }
    }

    let mut weights = neighbor_weights;
    let mut distances = neighbor_distances;
    for seed in &file_paths {
        if seed_set.contains(seed) {
            weights
                .entry(seed.clone())
                .and_modify(|value| {
                    if *value < 1.0 {
                        *value = 1.0;
                    }
                })
                .or_insert(1.0);
            distances.entry(seed.clone()).or_insert(0);
        }
    }

    Ok(TopologySelection {
        file_paths,
        weights,
        distances,
    })
}

fn apply_issue_metadata(
    repo_root: &Path,
    issue_context: &IssueContext,
    candidates: &mut HashMap<[u8; 32], UnifiedCandidate>,
) {
    let labels = &issue_context.issue.labels;
    let dependency_paths = collect_issue_paths(repo_root, &issue_context.dependency_issues);
    let related_paths = collect_issue_paths(repo_root, &issue_context.related_issues);
    let duplicate_paths = collect_issue_paths(repo_root, &issue_context.duplicate_issues);

    for candidate in candidates.values_mut() {
        candidate.label_match_count =
            count_label_matches(&candidate.text, &candidate.file_path, labels);
        let normalized = normalize_seed_path(repo_root, &candidate.file_path);
        if dependency_paths.contains(&normalized)
            || candidate
                .sources
                .iter()
                .any(|source| matches!(source, CandidateSource::DependencyIssue { .. }))
        {
            candidate.dependency_match = true;
        }
        if related_paths.contains(&normalized)
            || candidate
                .sources
                .iter()
                .any(|source| matches!(source, CandidateSource::RelatedIssue { .. }))
        {
            candidate.relates_to_match = true;
        }
        if duplicate_paths.contains(&normalized)
            || candidate
                .sources
                .iter()
                .any(|source| matches!(source, CandidateSource::DuplicateIssue { .. }))
        {
            candidate.duplicate_origin = true;
        }
    }
}

fn collect_issue_paths(repo_root: &Path, issues: &[Issue]) -> HashSet<String> {
    let mut paths = HashSet::new();
    for issue in issues {
        collect_issue_seeds(repo_root, &issue.affected_symbols, &mut paths);
    }
    paths
}

fn count_label_matches(text: &str, file_path: &str, labels: &[String]) -> usize {
    if labels.is_empty() {
        return 0;
    }
    let mut haystack = String::new();
    haystack.push_str(text);
    haystack.push('\n');
    haystack.push_str(file_path);
    let haystack = haystack.to_ascii_lowercase();
    labels
        .iter()
        .filter_map(|label| {
            let trimmed = label.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_ascii_lowercase())
            }
        })
        .filter(|label| haystack.contains(label))
        .count()
}

fn normalize_weights(candidates: &mut [UnifiedCandidate]) {
    let max_topology = candidates
        .iter()
        .filter_map(|candidate| candidate.topology_weight)
        .fold(0.0, f64::max);
    if max_topology > 0.0 {
        for candidate in candidates.iter_mut() {
            if let Some(weight) = candidate.topology_weight {
                candidate.topology_weight = Some(weight / max_topology);
            }
        }
    }

    let max_cochange = candidates
        .iter()
        .filter_map(|candidate| candidate.cochange_weight)
        .fold(0.0, f64::max);
    if max_cochange > 0.0 {
        for candidate in candidates.iter_mut() {
            if let Some(weight) = candidate.cochange_weight {
                candidate.cochange_weight = Some(weight / max_cochange);
            }
        }
    }
}

async fn apply_centrality_scores(
    ctx: &AssemblyContext<'_>,
    candidates: &mut [UnifiedCandidate],
) -> Result<()> {
    if candidates.is_empty() {
        return Ok(());
    }
    let limit = candidates.len().max(1);
    let scores = ctx.db.file_dependency_pagerank(limit).await?;
    let max_score = scores.iter().map(|(_, score)| *score).fold(0.0, f64::max);
    if max_score == 0.0 {
        return Ok(());
    }
    let mut map = HashMap::new();
    for (path, score) in scores {
        map.insert(path, score / max_score);
    }
    for candidate in candidates {
        if let Some(score) = map.get(&candidate.file_path) {
            candidate.centrality_score = Some(*score);
        }
    }
    Ok(())
}

fn compute_feature_scores(candidates: &mut [UnifiedCandidate], config: &ScoringConfig) {
    for candidate in candidates.iter_mut() {
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;

        if let Some(score) = candidate.semantic_score {
            weighted_sum += score * config.semantic_weight;
            weight_total += config.semantic_weight;
        }
        if let Some(weight) = candidate.topology_weight {
            let distance = candidate.topology_distance.unwrap_or(0) as f64;
            let decay_base = if config.topology_distance_decay > 0.0 {
                config.topology_distance_decay
            } else {
                1.0
            };
            let decay = 1.0 / (distance + decay_base);
            let score = weight * decay;
            weighted_sum += score * config.topology_weight;
            weight_total += config.topology_weight;
        }
        if let Some(score) = candidate.cochange_weight {
            weighted_sum += score * config.cochange_weight;
            weight_total += config.cochange_weight;
        }
        if let Some(score) = candidate.recency_score {
            weighted_sum += score * config.recency_weight;
            weight_total += config.recency_weight;
        }
        if let Some(score) = candidate.centrality_score {
            weighted_sum += score * config.centrality_weight;
            weight_total += config.centrality_weight;
        }

        let base = if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            0.0
        };

        let mut metadata = 0.0;
        if candidate.label_match_count > 0 {
            metadata += candidate.label_match_count as f64 * config.label_match_boost;
        }
        if candidate.dependency_match {
            metadata += config.dependency_boost;
        }
        if candidate.relates_to_match {
            metadata += config.relates_to_boost;
        }
        if candidate.duplicate_origin {
            metadata += config.duplicate_boost;
        }
        metadata *= config.metadata_weight;

        let mut score = (base + metadata).min(1.0);
        if candidate.sources.len() > 1 {
            score = (score + config.multi_source_bonus).min(1.0);
        }

        candidate.feature_score = score;
        candidate.final_score = score;
    }
}

async fn rerank_candidates(
    query: &str,
    ctx: &AssemblyContext<'_>,
    candidates: &mut [UnifiedCandidate],
    top_n: usize,
    blend: f64,
) -> Result<()> {
    if candidates.is_empty() || top_n == 0 {
        return Ok(());
    }
    let blend = blend.clamp(0.0, 1.0);
    let mut indices: Vec<usize> = (0..candidates.len()).collect();
    indices.sort_by(|a, b| {
        candidates[*b]
            .feature_score
            .partial_cmp(&candidates[*a].feature_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if indices.len() > top_n {
        indices.truncate(top_n);
    }
    let documents: Vec<String> = indices
        .iter()
        .map(|idx| candidates[*idx].text.clone())
        .collect();
    let tokenizer = crate::parse_tokenizer(&ctx.config.embedding.tokenizer)?;
    let query = seasoning::RerankQuery {
        text: query.to_string(),
        token_count: crate::search::count_tokens(&tokenizer, query)?,
    };
    let documents = documents
        .iter()
        .map(|text| {
            Ok(seasoning::RerankDocument {
                token_count: crate::search::count_tokens(&tokenizer, text)?,
                text: text.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let scores = ctx
        .reranker
        .rerank(&query, &documents)
        .await
        .map_err(|err| eyre!("reranking failed: {err}"))?;
    if scores.len() != indices.len() {
        return Err(eyre!(
            "reranker returned {} scores for {} results",
            scores.len(),
            indices.len()
        ));
    }
    for (idx, score) in indices.iter().zip(scores.iter()) {
        candidates[*idx].rerank_score = Some(*score);
    }
    for candidate in candidates.iter_mut() {
        if let Some(rerank_score) = candidate.rerank_score {
            candidate.final_score =
                (blend * rerank_score) + ((1.0 - blend) * candidate.feature_score);
        } else {
            candidate.final_score = candidate.feature_score;
        }
    }
    Ok(())
}

fn select_with_mmr(
    candidates: &[UnifiedCandidate],
    config: &SelectionConfig,
) -> Vec<UnifiedCandidate> {
    if candidates.is_empty() || config.max_blocks == 0 {
        return Vec::new();
    }
    let mut remaining = candidates.to_vec();
    let mut selected = Vec::new();
    let mut per_file: HashMap<String, usize> = HashMap::new();
    let lambda = config.diversity_lambda.clamp(0.0, 1.0);

    while selected.len() < config.max_blocks && !remaining.is_empty() {
        let mut best_index = None;
        let mut best_score = f64::MIN;
        for (idx, candidate) in remaining.iter().enumerate() {
            if config.max_per_file > 0 {
                let count = per_file.get(&candidate.file_path).copied().unwrap_or(0);
                if count >= config.max_per_file {
                    continue;
                }
            }
            let similarity = if selected
                .iter()
                .any(|selected: &UnifiedCandidate| selected.file_path == candidate.file_path)
            {
                1.0
            } else {
                0.0
            };
            let score = (lambda * candidate.final_score) - ((1.0 - lambda) * similarity);
            if score > best_score {
                best_score = score;
                best_index = Some(idx);
            }
        }
        let Some(idx) = best_index else {
            break;
        };
        let candidate = remaining.remove(idx);
        *per_file.entry(candidate.file_path.clone()).or_insert(0) += 1;
        selected.push(candidate);
    }
    selected
}

fn primary_source(sources: &[CandidateSource]) -> CandidateSource {
    sources
        .iter()
        .min_by_key(|source| source_rank(source))
        .cloned()
        .unwrap_or(CandidateSource::SemanticRetrieval)
}

/// Default stage: apply basic limits and de-duplication.
pub struct DefaultBudgetAndMerge {
    pub max_blocks: usize,
    pub max_bytes: Option<usize>,
    pub max_tokens: Option<usize>,
    pub reserved_output_tokens: usize,
    pub tokenizer: Option<Tokenizer>,
}

#[async_trait(?Send)]
impl BudgetAndMerge for DefaultBudgetAndMerge {
    async fn budget(
        &self,
        ctx: &AssemblyContext<'_>,
        arena: &mut Arena,
        input: Handle<SelectedBlocks>,
    ) -> Result<Handle<BudgetedBlocks>> {
        let selected = arena.get(input);
        let mut ordered = selected.blocks.clone();
        ordered.sort_by(|a, b| {
            source_rank(&a.source)
                .cmp(&source_rank(&b.source))
                .then_with(|| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.file_path.cmp(&b.file_path))
                .then_with(|| a.start_byte.cmp(&b.start_byte))
        });

        let token_counter = if let Some(tokenizer) = &self.tokenizer {
            Some(TokenCounter::new(tokenizer.clone())?)
        } else {
            None
        };
        let token_budget = self
            .max_tokens
            .map(|limit| limit.saturating_sub(self.reserved_output_tokens));
        let mut seen = std::collections::HashSet::new();
        let mut limited = Vec::new();
        let mut warnings = selected.warnings.clone();
        let mut bytes = 0usize;
        let mut tokens = 0usize;

        for block in &ordered {
            if !seen.insert(block.id.clone()) {
                continue;
            }
            let mut candidate = block.clone();
            let mut block_tokens = 0usize;

            if let Some(counter) = token_counter.as_ref() {
                block_tokens = counter.count(&candidate.text)?;
                if let Some(max_tokens) = token_budget
                    && tokens + block_tokens > max_tokens
                {
                    if is_expanded_source(&candidate.source) {
                        continue;
                    }
                    if let Some(trimmed) = trim_block_to_tokens(
                        &candidate,
                        max_tokens.saturating_sub(tokens),
                        counter,
                    )? {
                        candidate = trimmed;
                        block_tokens = counter.count(&candidate.text)?;
                    } else {
                        if matches!(
                            candidate.source,
                            CandidateSource::Explicit | CandidateSource::Pinned
                        ) {
                            warnings.push(format!(
                                "{} dropped due to token budget: {}",
                                source_label(&candidate.source),
                                normalize_path(ctx.repo_path, &candidate.file_path)
                            ));
                        }
                        continue;
                    }
                }
            }

            let mut next_bytes = bytes + candidate.text.len();
            if let Some(max) = self.max_bytes
                && next_bytes > max
            {
                if is_expanded_source(&candidate.source) {
                    continue;
                }
                if let Some(trimmed) = trim_block_to_bytes(&candidate, max.saturating_sub(bytes)) {
                    candidate = trimmed;
                    next_bytes = bytes + candidate.text.len();
                    if let Some(counter) = token_counter.as_ref() {
                        block_tokens = counter.count(&candidate.text)?;
                    }
                } else {
                    if matches!(
                        candidate.source,
                        CandidateSource::Explicit | CandidateSource::Pinned
                    ) {
                        warnings.push(format!(
                            "{} dropped due to byte budget: {}",
                            source_label(&candidate.source),
                            normalize_path(ctx.repo_path, &candidate.file_path)
                        ));
                    }
                    continue;
                }
            }

            tokens = tokens.saturating_add(block_tokens);
            limited.push(candidate);
            bytes = next_bytes;
            if limited.len() >= self.max_blocks {
                break;
            }
        }

        Ok(arena.insert(BudgetedBlocks {
            blocks: limited,
            warnings,
        }))
    }
}

/// Default stage: finalize assembly output.
pub struct DefaultAssembleContext;

#[async_trait(?Send)]
impl AssembleContext for DefaultAssembleContext {
    async fn assemble(
        &self,
        _ctx: &AssemblyContext<'_>,
        arena: &mut Arena,
        input: Handle<BudgetedBlocks>,
    ) -> Result<Handle<AssembledContext>> {
        let blocks = arena.get(input);
        Ok(arena.insert(AssembledContext {
            blocks: blocks.blocks.clone(),
            warnings: blocks.warnings.clone(),
        }))
    }
}

pub fn default_pipeline(
    config: &AppConfig,
    budget: BudgetOptions,
) -> AssemblyPipeline<
    UnifiedRetrieve,
    DefaultScoreAndSelect,
    DefaultBudgetAndMerge,
    DefaultAssembleContext,
> {
    default_pipeline_with_progress(config, budget, None)
}

pub fn default_pipeline_with_progress(
    config: &AppConfig,
    budget: BudgetOptions,
    progress: Option<Arc<dyn Fn(&'static str) + Send + Sync>>,
) -> AssemblyPipeline<
    UnifiedRetrieve,
    DefaultScoreAndSelect,
    DefaultBudgetAndMerge,
    DefaultAssembleContext,
> {
    let retrieve_config = UnifiedRetrieveConfig {
        semantic_limit: config.context.semantic_limit,
        topology_depth: config.context.topology_depth,
        topology_limit: config.context.topology_limit,
        cochange_limit: config.context.cochange_limit,
        dependency_issue_limit: config.context.dependency_issue_limit,
        related_issue_limit: config.context.related_issue_limit,
        duplicate_limit: config.context.duplicate_limit,
        per_file_limit: config.context.per_file_limit,
    };
    let scoring_config = ScoringConfig {
        semantic_weight: config.context.semantic_weight,
        topology_weight: config.context.topology_weight,
        cochange_weight: config.context.cochange_weight,
        recency_weight: config.context.recency_weight,
        centrality_weight: config.context.centrality_weight,
        metadata_weight: config.context.metadata_weight,
        topology_distance_decay: config.context.topology_distance_decay,
        label_match_boost: config.context.label_match_boost,
        dependency_boost: config.context.dependency_boost,
        relates_to_boost: config.context.relates_to_boost,
        duplicate_boost: config.context.duplicate_boost,
        multi_source_bonus: config.context.multi_source_bonus,
    };
    let selection_config = SelectionConfig {
        max_blocks: config.context.max_blocks,
        max_per_file: config.context.max_per_file,
        diversity_lambda: config.context.diversity_lambda,
    };
    let rerank_config = RerankConfig {
        top_n: config.context.rerank_top_n,
        blend: config.context.rerank_blend,
    };

    AssemblyPipeline::new(
        UnifiedRetrieve {
            config: retrieve_config,
            progress,
        },
        DefaultScoreAndSelect {
            scoring: scoring_config,
            selection: selection_config,
            rerank: rerank_config,
        },
        DefaultBudgetAndMerge {
            max_blocks: config.context.max_blocks,
            max_bytes: None,
            max_tokens: budget.max_tokens,
            reserved_output_tokens: budget.reserved_output_tokens,
            tokenizer: budget.tokenizer,
        },
        DefaultAssembleContext,
    )
}

pub struct BudgetOptions {
    pub max_tokens: Option<usize>,
    pub reserved_output_tokens: usize,
    pub tokenizer: Option<Tokenizer>,
}

struct TokenCounter {
    tokenizer: Tokenizer,
}

impl TokenCounter {
    fn new(tokenizer: Tokenizer) -> Result<Self> {
        let tokenizer = tokenizer
            .preload()
            .map_err(|err| eyre!("tokenizer preload failed: {err}"))?;
        Ok(Self { tokenizer })
    }

    fn count(&self, text: &str) -> Result<usize> {
        match &self.tokenizer {
            Tokenizer::Characters => Ok(text.chars().count()),
            Tokenizer::PreloadedTiktoken(bpe) => Ok(bpe.encode_ordinary(text).len()),
            Tokenizer::PreloadedHuggingFace(tokenizer) => tokenizer
                .encode(text, false)
                .map(|encoding| encoding.len())
                .map_err(|err| eyre!("tokenizer encode failed: {err}")),
            Tokenizer::Tiktoken(_) | Tokenizer::HuggingFace(_) => {
                Err(eyre!("tokenizer must be preloaded before counting tokens"))
            }
        }
    }
}

fn resolve_input_paths(repo_root: &Path, inputs: &[String]) -> Vec<String> {
    inputs
        .iter()
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            let path = Path::new(value);
            if path.is_absolute() {
                value.clone()
            } else {
                repo_root.join(path).to_string_lossy().to_string()
            }
        })
        .collect()
}

fn normalize_path(repo_root: &Path, file_path: &str) -> String {
    let path = Path::new(file_path);
    if path.is_absolute()
        && let Ok(stripped) = path.strip_prefix(repo_root)
        && let Some(rel) = stripped.to_str()
        && !rel.is_empty()
    {
        return rel.replace('\\', "/");
    }
    file_path.replace('\\', "/")
}

fn matches_any(path: &str, patterns: &[String]) -> bool {
    for pattern in patterns {
        let trimmed = pattern.trim();
        if trimmed.is_empty() {
            continue;
        }
        let prefix = trimmed.trim_end_matches('/');
        if path == prefix || path.starts_with(prefix) {
            return true;
        }
    }
    false
}

fn is_allowed_path(repo_root: &Path, selection: &super::SelectionOptions, file_path: &str) -> bool {
    let normalized = normalize_path(repo_root, file_path);
    if !selection.scope_paths.is_empty()
        && !matches_any(&normalized, &selection.scope_paths)
        && !matches_any(file_path, &selection.scope_paths)
    {
        return false;
    }
    if !selection.explicit_excludes.is_empty()
        && (matches_any(&normalized, &selection.explicit_excludes)
            || matches_any(file_path, &selection.explicit_excludes))
    {
        return false;
    }
    true
}

fn trim_block_to_tokens(
    block: &ContextBlock,
    max_tokens: usize,
    counter: &TokenCounter,
) -> Result<Option<ContextBlock>> {
    if max_tokens == 0 {
        return Ok(None);
    }
    let mut acc = String::new();
    let mut end_line = block.start_line.saturating_sub(1);
    for (idx, line) in block.text.lines().enumerate() {
        let candidate = if acc.is_empty() {
            line.to_string()
        } else {
            format!("{acc}\n{line}")
        };
        if counter.count(&candidate)? > max_tokens {
            break;
        }
        acc = candidate;
        end_line = block.start_line + idx as i64;
    }
    if acc.is_empty() {
        return Ok(None);
    }
    let mut trimmed = block.clone();
    trimmed.text = acc;
    trimmed.end_line = end_line;
    trimmed.end_byte = block.start_byte + trimmed.text.len() as i64;
    Ok(Some(trimmed))
}

fn trim_block_to_bytes(block: &ContextBlock, max_bytes: usize) -> Option<ContextBlock> {
    if max_bytes == 0 {
        return None;
    }
    let mut acc = String::new();
    let mut end_line = block.start_line.saturating_sub(1);
    for (idx, line) in block.text.lines().enumerate() {
        let candidate = if acc.is_empty() {
            line.to_string()
        } else {
            format!("{acc}\n{line}")
        };
        if candidate.len() > max_bytes {
            break;
        }
        acc = candidate;
        end_line = block.start_line + idx as i64;
    }
    if acc.is_empty() {
        return None;
    }
    let mut trimmed = block.clone();
    trimmed.text = acc;
    trimmed.end_line = end_line;
    trimmed.end_byte = block.start_byte + trimmed.text.len() as i64;
    Some(trimmed)
}

fn source_label(source: &CandidateSource) -> &'static str {
    match source {
        CandidateSource::Explicit => "explicit",
        CandidateSource::Pinned => "pinned",
        CandidateSource::SemanticRetrieval => "semantic",
        CandidateSource::TopologyNeighbor { .. } => "topology",
        CandidateSource::CochangeExpansion => "cochange",
        CandidateSource::DependencyIssue { .. } => "dependency",
        CandidateSource::RelatedIssue { .. } => "related",
        CandidateSource::DuplicateIssue { .. } => "duplicate",
    }
}

fn source_rank(source: &CandidateSource) -> u8 {
    match source {
        CandidateSource::Explicit => 0,
        CandidateSource::Pinned => 1,
        CandidateSource::SemanticRetrieval => 2,
        CandidateSource::DependencyIssue { .. } => 3,
        CandidateSource::RelatedIssue { .. } => 4,
        CandidateSource::DuplicateIssue { .. } => 5,
        CandidateSource::TopologyNeighbor { .. } => 6,
        CandidateSource::CochangeExpansion => 7,
    }
}

fn is_expanded_source(source: &CandidateSource) -> bool {
    matches!(
        source,
        CandidateSource::TopologyNeighbor { .. }
            | CandidateSource::CochangeExpansion
            | CandidateSource::DependencyIssue { .. }
            | CandidateSource::RelatedIssue { .. }
            | CandidateSource::DuplicateIssue { .. }
    )
}

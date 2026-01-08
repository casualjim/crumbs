#![allow(dead_code)]

use async_trait::async_trait;
use eyre::{Result, eyre};
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use text_chunking::Tokenizer;

use super::{Arena, AssemblyContext, Handle};
use crate::config::AppConfig;
use crate::search::SearchConfig;

/// Input query for assembly.
pub struct QueryInput {
    pub text: String,
}

/// Initial retrieval candidates (e.g., vector/fts hits).
pub struct CandidateSet {
    pub chunks: Vec<CandidateChunk>,
    pub warnings: Vec<String>,
}

/// Expanded candidate set after graph/history expansion.
pub struct ExpandedCandidates {
    pub chunks: Vec<CandidateChunk>,
    pub expanded_files: Vec<String>,
    pub warnings: Vec<String>,
}

/// AST-refined blocks (structured code blocks).
pub struct AstBlocks {
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

/// A candidate chunk returned from retrieval.
#[derive(Clone)]
pub struct CandidateChunk {
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateSource {
    Explicit,
    Pinned,
    Primary,
    Expanded,
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
}

#[async_trait(?Send)]
pub trait RetrieveCandidates: Send + Sync {
    async fn retrieve(
        &self,
        ctx: &AssemblyContext<'_>,
        arena: &mut Arena,
        input: Handle<QueryInput>,
    ) -> Result<Handle<CandidateSet>>;
}

#[async_trait(?Send)]
pub trait ExpandGraph: Send + Sync {
    async fn expand(
        &self,
        ctx: &AssemblyContext<'_>,
        arena: &mut Arena,
        input: Handle<CandidateSet>,
    ) -> Result<Handle<ExpandedCandidates>>;
}

#[async_trait(?Send)]
pub trait RefineAst: Send + Sync {
    async fn refine(
        &self,
        ctx: &AssemblyContext<'_>,
        arena: &mut Arena,
        input: Handle<ExpandedCandidates>,
    ) -> Result<Handle<AstBlocks>>;
}

#[async_trait(?Send)]
pub trait BudgetAndMerge: Send + Sync {
    async fn budget(
        &self,
        ctx: &AssemblyContext<'_>,
        arena: &mut Arena,
        input: Handle<AstBlocks>,
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
pub struct AssemblyPipeline<R, G, A, B, F> {
    retrieve: R,
    expand: G,
    refine: A,
    budget: B,
    assemble: F,
}

impl<R, G, A, B, F> AssemblyPipeline<R, G, A, B, F> {
    pub fn new(retrieve: R, expand: G, refine: A, budget: B, assemble: F) -> Self {
        Self {
            retrieve,
            expand,
            refine,
            budget,
            assemble,
        }
    }
}

impl<R, G, A, B, F> AssemblyPipeline<R, G, A, B, F>
where
    R: RetrieveCandidates,
    G: ExpandGraph,
    A: RefineAst,
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
        let expanded = self.expand.expand(ctx, arena, candidates).await?;
        let refined = self.refine.refine(ctx, arena, expanded).await?;
        let budgeted = self.budget.budget(ctx, arena, refined).await?;
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
        notify("expanding related files");
        let expanded = self.expand.expand(ctx, arena, candidates).await?;
        notify("fetching related chunks");
        let refined = self.refine.refine(ctx, arena, expanded).await?;
        notify("budgeting context");
        let budgeted = self.budget.budget(ctx, arena, refined).await?;
        notify("assembling prompt");
        self.assemble.assemble(ctx, arena, budgeted).await
    }
}

/// Default stage: retrieve candidates using embedding/FTS search.
pub struct DefaultRetrieve {
    pub config: SearchConfig,
    pub progress: Option<Arc<dyn Fn(&'static str) + Send + Sync>>,
}

#[async_trait(?Send)]
impl RetrieveCandidates for DefaultRetrieve {
    async fn retrieve(
        &self,
        ctx: &AssemblyContext<'_>,
        arena: &mut Arena,
        input: Handle<QueryInput>,
    ) -> Result<Handle<CandidateSet>> {
        if let Some(progress) = &self.progress {
            progress("loading tokenizer");
        }
        let embedder = ctx
            .embedder
            .ok_or_else(|| eyre!("embedding provider required for retrieval"))?;
        let tokenizer = crate::parse_tokenizer(&ctx.config.embedding.tokenizer)?;
        let progress = self.progress.as_ref().map(|progress| progress.as_ref());
        let query = arena.get(input);
        let search_ctx = crate::search::SearchContext {
            db: ctx.db,
            embedder,
            reranker: ctx.reranker,
            tokenizer: &tokenizer,
            progress,
        };
        let mut warnings = Vec::new();
        let mut chunks: Vec<CandidateChunk> = Vec::new();

        let per_file_limit = self.config.limit.max(1);
        if !ctx.selection.explicit_includes.is_empty() {
            let include_paths = resolve_input_paths(ctx.repo_path, &ctx.selection.explicit_includes);
            let rows = ctx
                .db
                .chunks_for_files(&include_paths, per_file_limit)
                .await?;
            let found: HashSet<String> =
                rows.iter().map(|row| row.file_path.clone()).collect();
            for path in &include_paths {
                if !found.contains(path) {
                    warnings.push(format!(
                        "explicit include not found: {}",
                        normalize_path(ctx.repo_path, path)
                    ));
                }
            }
            for row in rows {
                chunks.push(CandidateChunk {
                    id: row.id,
                    file_path: row.file_path,
                    start_byte: row.start_byte,
                    end_byte: row.end_byte,
                    chunk_hash: row.chunk_hash,
                    start_line: row.start_line,
                    end_line: row.end_line,
                    text: row.text,
                    score: 1.0,
                    source: CandidateSource::Explicit,
                });
            }
        }

        if !ctx.selection.pinned_items.is_empty() {
            let pinned_paths = resolve_input_paths(ctx.repo_path, &ctx.selection.pinned_items);
            let rows = ctx
                .db
                .chunks_for_files(&pinned_paths, per_file_limit)
                .await?;
            let found: HashSet<String> =
                rows.iter().map(|row| row.file_path.clone()).collect();
            for path in &pinned_paths {
                if !found.contains(path) {
                    warnings.push(format!(
                        "pinned item not found: {}",
                        normalize_path(ctx.repo_path, path)
                    ));
                }
            }
            for row in rows {
                chunks.push(CandidateChunk {
                    id: row.id,
                    file_path: row.file_path,
                    start_byte: row.start_byte,
                    end_byte: row.end_byte,
                    chunk_hash: row.chunk_hash,
                    start_line: row.start_line,
                    end_line: row.end_line,
                    text: row.text,
                    score: 1.0,
                    source: CandidateSource::Pinned,
                });
            }
        }

        let results =
            crate::search::search(&search_ctx, &query.text, self.config.clone()).await?;
        let mut retrieved = results
            .into_iter()
            .filter(|result| is_allowed_path(ctx.repo_path, &ctx.selection, &result.file_path))
            .map(|result| CandidateChunk {
                id: result.id,
                file_path: result.file_path,
                start_byte: result.start_byte,
                end_byte: result.end_byte,
                chunk_hash: result.chunk_hash,
                start_line: result.start_line,
                end_line: result.end_line,
                text: result.text,
                score: result.score,
                source: CandidateSource::Primary,
            })
            .collect();
        chunks.append(&mut retrieved);
        Ok(arena.insert(CandidateSet { chunks, warnings }))
    }
}

/// Default stage: expand with co-changed files from git history.
pub struct DefaultExpandGraph {
    pub max_expanded_files: usize,
}

#[async_trait(?Send)]
impl ExpandGraph for DefaultExpandGraph {
    async fn expand(
        &self,
        ctx: &AssemblyContext<'_>,
        arena: &mut Arena,
        input: Handle<CandidateSet>,
    ) -> Result<Handle<ExpandedCandidates>> {
        let candidates = arena.get(input);
        let mut seeds = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for chunk in &candidates.chunks {
            if seen.insert(chunk.file_path.clone()) {
                seeds.push(chunk.file_path.clone());
            }
        }

        let mut expanded = if seeds.is_empty() {
            Vec::new()
        } else {
            ctx.db.cochange_neighbors(&seeds, self.max_expanded_files).await?
        };
        if !expanded.is_empty() {
            expanded.retain(|path| is_allowed_path(ctx.repo_path, &ctx.selection, path));
        }

        Ok(arena.insert(ExpandedCandidates {
            chunks: candidates.chunks.clone(),
            expanded_files: expanded,
            warnings: candidates.warnings.clone(),
        }))
    }
}

/// Default stage: fetch additional chunks from expanded files.
pub struct DefaultRefineAst {
    pub per_file_limit: usize,
}

#[async_trait(?Send)]
impl RefineAst for DefaultRefineAst {
    async fn refine(
        &self,
        ctx: &AssemblyContext<'_>,
        arena: &mut Arena,
        input: Handle<ExpandedCandidates>,
    ) -> Result<Handle<AstBlocks>> {
        let expanded = arena.get(input);
        let mut blocks: Vec<ContextBlock> = expanded
            .chunks
            .iter()
            .map(|chunk| ContextBlock {
                id: chunk.id.clone(),
                file_path: chunk.file_path.clone(),
                start_byte: chunk.start_byte,
                end_byte: chunk.end_byte,
                chunk_hash: chunk.chunk_hash,
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                text: chunk.text.clone(),
                score: chunk.score,
                source: chunk.source,
            })
            .collect();

        if !expanded.expanded_files.is_empty() {
            let extra = ctx
                .db
                .chunks_for_files(&expanded.expanded_files, self.per_file_limit)
                .await?;
            for row in extra {
                blocks.push(ContextBlock {
                    id: row.id,
                    file_path: row.file_path,
                    start_byte: row.start_byte,
                    end_byte: row.end_byte,
                    chunk_hash: row.chunk_hash,
                    start_line: row.start_line,
                    end_line: row.end_line,
                    text: row.text,
                    score: 0.0,
                    source: CandidateSource::Expanded,
                });
            }
        }

        Ok(arena.insert(AstBlocks {
            blocks,
            warnings: expanded.warnings.clone(),
        }))
    }
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
        input: Handle<AstBlocks>,
    ) -> Result<Handle<BudgetedBlocks>> {
        let blocks = arena.get(input);
        let mut ordered = blocks.blocks.clone();
        ordered.sort_by(|a, b| {
            source_rank(a.source)
                .cmp(&source_rank(b.source))
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
        let mut warnings = blocks.warnings.clone();
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
                if let Some(max_tokens) = token_budget {
                    if tokens + block_tokens > max_tokens {
                        if candidate.source == CandidateSource::Expanded {
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
                            if candidate.source == CandidateSource::Explicit
                                || candidate.source == CandidateSource::Pinned
                            {
                                warnings.push(format!(
                                    "{} dropped due to token budget: {}",
                                    source_label(candidate.source),
                                    normalize_path(ctx.repo_path, &candidate.file_path)
                                ));
                            }
                            continue;
                        }
                    }
                }
            }

            let mut next_bytes = bytes + candidate.text.len();
            if let Some(max) = self.max_bytes
                && next_bytes > max
            {
                if candidate.source == CandidateSource::Expanded {
                    continue;
                }
                if let Some(trimmed) =
                    trim_block_to_bytes(&candidate, max.saturating_sub(bytes))
                {
                    candidate = trimmed;
                    next_bytes = bytes + candidate.text.len();
                    if let Some(counter) = token_counter.as_ref() {
                        block_tokens = counter.count(&candidate.text)?;
                    }
                } else {
                    if candidate.source == CandidateSource::Explicit
                        || candidate.source == CandidateSource::Pinned
                    {
                        warnings.push(format!(
                            "{} dropped due to byte budget: {}",
                            source_label(candidate.source),
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
    DefaultRetrieve,
    DefaultExpandGraph,
    DefaultRefineAst,
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
    DefaultRetrieve,
    DefaultExpandGraph,
    DefaultRefineAst,
    DefaultBudgetAndMerge,
    DefaultAssembleContext,
> {
    let mut search_config = SearchConfig::new(config.search.limit, config.search.hybrid_weight);
    search_config.min_score = config.search.min_score;
    search_config.path_prefixes = config.search.path_prefixes.clone();
    search_config.file_exts = config.search.file_exts.clone();
    AssemblyPipeline::new(
        DefaultRetrieve {
            config: search_config,
            progress,
        },
        DefaultExpandGraph {
            max_expanded_files: config.search.limit,
        },
        DefaultRefineAst { per_file_limit: 3 },
        DefaultBudgetAndMerge {
            max_blocks: config.search.limit,
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
                repo_root
                    .join(path)
                    .to_string_lossy()
                    .to_string()
            }
        })
        .collect()
}

fn normalize_path(repo_root: &Path, file_path: &str) -> String {
    let path = Path::new(file_path);
    if path.is_absolute() {
        if let Ok(stripped) = path.strip_prefix(repo_root) {
            if let Some(rel) = stripped.to_str() {
                if !rel.is_empty() {
                    return rel.replace('\\', "/");
                }
            }
        }
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

fn is_allowed_path(
    repo_root: &Path,
    selection: &super::SelectionOptions,
    file_path: &str,
) -> bool {
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

fn source_label(source: CandidateSource) -> &'static str {
    match source {
        CandidateSource::Explicit => "explicit",
        CandidateSource::Pinned => "pinned",
        CandidateSource::Primary => "primary",
        CandidateSource::Expanded => "expanded",
    }
}

fn source_rank(source: CandidateSource) -> u8 {
    match source {
        CandidateSource::Explicit => 0,
        CandidateSource::Pinned => 1,
        CandidateSource::Primary => 2,
        CandidateSource::Expanded => 3,
    }
}

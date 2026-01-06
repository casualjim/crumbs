#![allow(dead_code)]

use async_trait::async_trait;
use eyre::{Result, eyre};
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
}

/// Expanded candidate set after graph/history expansion.
pub struct ExpandedCandidates {
    pub chunks: Vec<CandidateChunk>,
    pub expanded_files: Vec<String>,
}

/// AST-refined blocks (structured code blocks).
pub struct AstBlocks {
    pub blocks: Vec<ContextBlock>,
}

/// Budgeted/merged blocks ready for prompt assembly.
pub struct BudgetedBlocks {
    pub blocks: Vec<ContextBlock>,
}

/// Final assembled context payload.
pub struct AssembledContext {
    pub blocks: Vec<ContextBlock>,
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
}

/// Default stage: retrieve candidates using embedding/FTS search.
pub struct DefaultRetrieve {
    pub config: SearchConfig,
}

#[async_trait(?Send)]
impl RetrieveCandidates for DefaultRetrieve {
    async fn retrieve(
        &self,
        ctx: &AssemblyContext<'_>,
        arena: &mut Arena,
        input: Handle<QueryInput>,
    ) -> Result<Handle<CandidateSet>> {
        let embedder = ctx
            .embedder
            .ok_or_else(|| eyre!("embedding provider required for retrieval"))?;
        let tokenizer = crate::parse_tokenizer(&ctx.config.embedding.tokenizer)?;
        let query = arena.get(input);
        let results = crate::search::search(
            ctx.db,
            embedder,
            ctx.reranker,
            &tokenizer,
            &query.text,
            self.config.clone(),
        )
        .await?;
        let chunks = results
            .into_iter()
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
        Ok(arena.insert(CandidateSet { chunks }))
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

        let expanded = if seeds.is_empty() {
            Vec::new()
        } else {
            ctx.db.cochange_neighbors(&seeds, self.max_expanded_files)?
        };

        Ok(arena.insert(ExpandedCandidates {
            chunks: candidates.chunks.clone(),
            expanded_files: expanded,
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
                .chunks_for_files(&expanded.expanded_files, self.per_file_limit)?;
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

        Ok(arena.insert(AstBlocks { blocks }))
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
        _ctx: &AssemblyContext<'_>,
        arena: &mut Arena,
        input: Handle<AstBlocks>,
    ) -> Result<Handle<BudgetedBlocks>> {
        let blocks = arena.get(input);
        let mut ordered = blocks.blocks.clone();
        ordered.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| source_rank(a.source).cmp(&source_rank(b.source)))
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
        let mut bytes = 0usize;
        let mut tokens = 0usize;

        for block in &ordered {
            if !seen.insert(block.id.clone()) {
                continue;
            }
            if let Some(counter) = token_counter.as_ref() {
                let block_tokens = counter.count(&block.text)?;
                if let Some(max_tokens) = token_budget {
                    if tokens + block_tokens > max_tokens {
                        continue;
                    }
                }
                tokens += block_tokens;
            }
            let next_bytes = bytes + block.text.len();
            if let Some(max) = self.max_bytes
                && next_bytes > max
            {
                continue;
            }
            limited.push(block.clone());
            bytes = next_bytes;
            if limited.len() >= self.max_blocks {
                break;
            }
        }

        Ok(arena.insert(BudgetedBlocks { blocks: limited }))
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
    let mut search_config = SearchConfig::new(config.search.limit, config.search.hybrid_weight);
    search_config.min_score = config.search.min_score;
    search_config.path_prefixes = config.search.path_prefixes.clone();
    search_config.file_exts = config.search.file_exts.clone();
    search_config.decompose = config.search.decompose;
    search_config.rerank = config.search.rerank;
    AssemblyPipeline::new(
        DefaultRetrieve { config: search_config },
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

fn source_rank(source: CandidateSource) -> u8 {
    match source {
        CandidateSource::Primary => 0,
        CandidateSource::Expanded => 1,
    }
}

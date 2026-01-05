#![allow(dead_code)]

use async_trait::async_trait;
use eyre::{Result, eyre};

use super::{Arena, AssemblyContext, Handle};
use crate::config::AppConfig;

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
    pub file_path: String,
    pub start_byte: i64,
    pub end_byte: i64,
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
    pub limit: usize,
    pub hybrid_weight: f32,
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
        let query = arena.get(input);
        let results = crate::search::search(
            ctx.db,
            embedder,
            &query.text,
            self.limit,
            self.hybrid_weight,
        )
        .await?;
        let chunks = results
            .into_iter()
            .map(|result| CandidateChunk {
                id: format!(
                    "{}:{}-{}",
                    result.file_path, result.start_byte, result.end_byte
                ),
                file_path: result.file_path,
                start_byte: result.start_byte,
                end_byte: result.end_byte,
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
                file_path: chunk.file_path.clone(),
                start_byte: chunk.start_byte,
                end_byte: chunk.end_byte,
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
                    file_path: row.file_path,
                    start_byte: row.start_byte,
                    end_byte: row.end_byte,
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
        let mut seen = std::collections::HashSet::new();
        let mut limited = Vec::new();
        let mut bytes = 0usize;

        for block in &blocks.blocks {
            let key = (block.file_path.clone(), block.start_byte, block.end_byte);
            if !seen.insert(key) {
                continue;
            }
            let next_bytes = bytes + block.text.len();
            if let Some(max) = self.max_bytes
                && next_bytes > max
            {
                break;
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
) -> AssemblyPipeline<
    DefaultRetrieve,
    DefaultExpandGraph,
    DefaultRefineAst,
    DefaultBudgetAndMerge,
    DefaultAssembleContext,
> {
    AssemblyPipeline::new(
        DefaultRetrieve {
            limit: config.search.limit,
            hybrid_weight: config.search.hybrid_weight,
        },
        DefaultExpandGraph {
            max_expanded_files: config.search.limit,
        },
        DefaultRefineAst { per_file_limit: 3 },
        DefaultBudgetAndMerge {
            max_blocks: config.search.limit,
            max_bytes: None,
        },
        DefaultAssembleContext,
    )
}

mod assembly;
mod config;
mod db;
mod embedding;
mod graph;
mod indexer;
mod logging;
mod progress;
mod reqwestx;
mod repository;
mod reranker;
mod search;
mod topology;
#[cfg(test)]
mod test_support;

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use async_trait::async_trait;
use clap::Parser;
use eyre::{Result, eyre};
use text_chunking::Tokenizer;

use crate::config::{Cli, Command};
use crate::db::Db;
use crate::embedding::{Client as EmbedClient, EmbedderConfig, ProviderDialect};
use crate::indexer::{Indexer, IndexerConfig};
use crate::assembly::pipeline::{AssembleContext, BudgetAndMerge, DefaultAssembleContext, DefaultBudgetAndMerge};
use crate::reranker::{Client as RerankerClient, RerankerConfig, RerankingProvider};
use crate::repository::Repository;
use crate::topology::RefactorOptions;

struct NoopReranker;

#[async_trait]
impl RerankingProvider for NoopReranker {
    async fn rerank(&self, _query: &str, documents: &[String]) -> Result<Vec<f64>> {
        Ok(vec![0.0; documents.len()])
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let _logging_guard = logging::init()?;
    let cli = Cli::parse();

    match &cli.command {
        Command::Init(init) => {
            let result = config::init_config(init)?;
            if result.wrote_config {
                println!("Wrote config to {}", result.config_path.display());
            } else {
                println!("Config already exists at {}", result.config_path.display());
            }
            if result.wrote_secrets {
                println!("Wrote secrets to {}", result.secrets_path.display());
            } else {
                println!("Secrets already exist at {}", result.secrets_path.display());
            }
        }
        Command::Index(_) => {
            let cfg = config::load_config(&cli)?;
            let project = config::resolve_project(&cfg, project_override(&cli.command))?;
            let tokenizer = parse_tokenizer(&cfg.embedding.tokenizer)?;

            let embedder = build_embedder(&cfg.embedding)?;
            let db = Db::open(&project.database_path, Some(cfg.embedding.embedding_dim)).await?;
            let index_config = IndexerConfig {
                repo_path: project.repo_path.clone(),
                max_chunk_size: cfg.chunking.max_chunk_size,
                overlap_percentage: cfg.chunking.overlap,
                tokenizer: tokenizer.clone(),
                max_parallel: cfg.chunking.max_parallel,
                max_file_size: Some(cfg.chunking.max_file_size),
                large_file_threads: cfg.chunking.large_file_threads,
                max_batch_size: cfg.embedding.max_batch_size,
                max_tokens: cfg.embedding.context_length,
                embedding_workers: cfg.embedding.workers,
                cancel_token: None,
                history: graph::HistoryConfig {
                    depth: cfg.history.depth,
                    commit_size_limit_ratio: cfg.history.commit_size_limit_ratio,
                    multi_parents: cfg.history.multi_parents,
                    issue_regex: cfg.history.issue_regex.clone(),
                    commit_exclude_regex: cfg.history.commit_exclude_regex.clone(),
                    author_exclude_regex: cfg.history.author_exclude_regex.clone(),
                    path_specs: split_history_path_specs(&cfg.history.path_specs),
                },
            };
            let indexer = Indexer::new(&db, embedder, index_config);
            indexer.index().await?;
        }
        Command::Search(cmd) => {
            let cfg = config::load_config(&cli)?;
            let tokenizer = parse_tokenizer(&cfg.embedding.tokenizer)?;
            let embedder = build_embedder(&cfg.embedding)?;
            let project = config::resolve_project(&cfg, project_override(&cli.command))?;
            let db = Db::open(&project.database_path, Some(cfg.embedding.embedding_dim)).await?;
            let reranker = build_reranker(&cfg)?;
            let mut search_config =
                search::SearchConfig::new(cfg.search.limit, cfg.search.hybrid_weight);
            search_config.path_prefixes = cfg.search.path_prefixes.clone();
            search_config.file_exts = cfg.search.file_exts.clone();
            let search_ctx = search::SearchContext {
                db: &db,
                embedder: &embedder,
                reranker: &reranker,
                tokenizer: &tokenizer,
                progress: None,
            };
            let results = search::search(&search_ctx, &cmd.query, search_config).await?;
            for (idx, result) in results.iter().enumerate() {
                let mut score_line = format!("score={:.4}", result.score);
                if let Some(vector) = result.vector_score {
                    score_line.push_str(&format!(" vec={vector:.4}"));
                }
                if let Some(fts) = result.fts_score {
                    score_line.push_str(&format!(" fts={fts:.4}"));
                }
                println!(
                    "{idx}. {path}:{start}-{end} {score_line}\n{text}\n",
                    idx = idx + 1,
                    path = result.file_path,
                    start = result.start_byte,
                    end = result.end_byte,
                    text = result.text
                );
            }
        }
        Command::Prompt(cmd) => {
            let cfg = config::load_config(&cli)?;
            let project = config::resolve_project(&cfg, project_override(&cli.command))?;
            let tokenizer = parse_tokenizer(&cfg.embedding.tokenizer)?;
            let db = Db::open(&project.database_path, Some(cfg.embedding.embedding_dim)).await?;
            let progress_watch = progress::watch_spinner("assembling prompt");
            let (spinner, progress_tx) = match progress_watch {
                Some((spinner, tx)) => (Some(spinner), Some(tx)),
                None => (None, None),
            };

            let selection_opts = assembly::SelectionOptions {
                scope_paths: cmd.scope.clone(),
                explicit_includes: cmd.include.clone(),
                explicit_excludes: cmd.exclude.clone(),
                pinned_items: cmd.pin.clone(),
            };

            let max_tokens = if cmd.max_tokens == 0 {
                Some(cfg.embedding.context_length)
            } else {
                Some(cmd.max_tokens)
            };
            let prompt_tokenizer_value = cmd
                .prompt
                .tokenizer
                .clone()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| cfg.prompt.tokenizer.clone());
            let prompt_tokenizer = if prompt_tokenizer_value.trim().is_empty() {
                tokenizer.clone()
            } else {
                parse_tokenizer(&prompt_tokenizer_value)?
            };
            let prompt_theme_value = cmd
                .prompt
                .theme
                .clone()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| cfg.prompt.theme.clone());
            let prompt_theme = if prompt_theme_value.trim().is_empty() {
                None
            } else {
                Some(prompt_theme_value)
            };
            let budget = assembly::pipeline::BudgetOptions {
                max_tokens,
                reserved_output_tokens: cmd.reserved_output_tokens,
                tokenizer: Some(prompt_tokenizer),
            };
            let sections = resolve_sections(&cmd.sections);

            let (blocks, warnings) = if cmd.topology {
                let snapshot = topology::TopologySnapshot::load(&db).await?;
                let seed_sources = collect_seed_sources(
                    &project.repo_path,
                    &snapshot,
                    &selection_opts,
                    &cmd.include,
                    &cmd.pin,
                )?;
                let selection = select_topology_files(
                    &snapshot,
                    &selection_opts,
                    seed_sources,
                    cmd.topology_depth,
                    cmd.topology_limit,
                )?;
                let blocks = build_topology_blocks(&db, &selection, cmd.topology_per_file).await?;
                let max_blocks = cmd
                    .topology_limit
                    .saturating_mul(cmd.topology_per_file.max(1));
                let noop = NoopReranker;
                let ctx = assembly::AssemblyContext {
                    repo_path: &project.repo_path,
                    db: &db,
                    embedder: None,
                    reranker: &noop,
                    config: &cfg,
                    selection: selection_opts.clone(),
                };
                budget_blocks(&ctx, blocks, selection.warnings, &budget, max_blocks).await?
            } else {
                let embedder = build_embedder(&cfg.embedding)?;
                let reranker = build_reranker(&cfg)?;
                let ctx = assembly::AssemblyContext {
                    repo_path: &project.repo_path,
                    db: &db,
                    embedder: Some(&embedder),
                    reranker: &reranker as &dyn RerankingProvider,
                    config: &cfg,
                    selection: selection_opts.clone(),
                };

                let progress_callback = progress_tx.as_ref().map(|tx| {
                    let tx = tx.clone();
                    std::sync::Arc::new(move |message: &'static str| {
                        let _ = tx.send(message);
                    }) as std::sync::Arc<dyn Fn(&'static str) + Send + Sync>
                });
                let pipeline_budget = assembly::pipeline::BudgetOptions {
                    max_tokens: budget.max_tokens,
                    reserved_output_tokens: budget.reserved_output_tokens,
                    tokenizer: budget.tokenizer.clone(),
                };
                let pipeline = assembly::pipeline::default_pipeline_with_progress(
                    &cfg,
                    pipeline_budget,
                    progress_callback,
                );
                let mut arena = assembly::Arena::new();
                let input = arena.insert(assembly::pipeline::QueryInput {
                    text: cmd.task.clone(),
                });
                let progress = progress_tx.clone();
                let handle = pipeline
                    .run_with_progress(&ctx, &mut arena, input, |message| {
                        if let Some(progress) = &progress {
                            let _ = progress.send(message);
                        }
                    })
                    .await?;
                let assembled = arena.get(handle);
                (assembled.blocks.clone(), assembled.warnings.clone())
            };

            let enriched =
                assembly::output::enrich_blocks(&project.repo_path, &db, &blocks).await?;
            let overview =
                assembly::output::build_repository_overview(&project.repo_path, &db).await;
            let payload = assembly::output::PromptPayload {
                overview,
                task: cmd.task.clone(),
                blocks: enriched,
                warnings,
            };
            let rendered = assembly::output::render_prompt(
                &payload,
                sections,
                prompt_theme.as_deref(),
            );
            if let Some(spinner) = spinner {
                spinner.finish_and_clear();
            }
            print!("{rendered}");
        }
        Command::Config(cmd) => match &cmd.command {
            config::ConfigCommand::Show => {
                let cfg = config::load_config(&cli)?;
                println!("{cfg:#?}");
            }
            config::ConfigCommand::Set(set) => {
                let result = config::set_config_value(set, cli.config_file.as_deref())?;
                if result.created {
                    println!(
                        "Created config at {} and set {}",
                        result.config_path.display(),
                        result.key
                    );
                } else {
                    println!("Updated {} in {}", result.key, result.config_path.display());
                }
            }
            config::ConfigCommand::Doctor => {
                let cfg = config::load_config(&cli)?;
                if cfg.embedding.api_key.is_none() {
                    return Err(eyre!("embedding api key missing"));
                }
                let _ = parse_tokenizer(&cfg.embedding.tokenizer)?;
                let _ = build_embedder(&cfg.embedding)?;
                println!("Config OK");
            }
        },
        Command::Topology(cmd) => {
            let cfg = config::load_config(&cli)?;
            let project = config::resolve_project(&cfg, project_override(&cli.command))?;
            let db = Db::open(&project.database_path, Some(cfg.embedding.embedding_dim)).await?;
            let snapshot = topology::TopologySnapshot::load(&db).await?;

            match &cmd.command {
                config::TopologyCommand::Stats(options) => {
                    let cycles_over_threshold = snapshot
                        .cycles
                        .iter()
                        .filter(|cycle| cycle.nodes.len() >= options.min_cycle_size)
                        .count();
                    println!("nodes: {}", snapshot.stats.node_count);
                    println!("edges: {}", snapshot.stats.edge_count);
                    println!("components: {}", snapshot.stats.component_count);
                    println!(
                        "sccs: {} (cyclic: {})",
                        snapshot.stats.scc_count, snapshot.stats.cyclic_scc_count
                    );
                    println!("cyclic nodes: {}", snapshot.stats.cyclic_node_count);
                    println!("cycles >= {}: {}", options.min_cycle_size, cycles_over_threshold);
                    println!("betti_0: {}", snapshot.stats.betti_0);
                    println!("betti_1: {}", snapshot.stats.betti_1);
                    println!("solid_score: {:.3}", snapshot.stats.solid_score);
                    println!("avg_out_degree: {:.2}", snapshot.stats.avg_out_degree);
                    println!("density: {:.4}", snapshot.stats.density);
                }
                config::TopologyCommand::Cycles(options) => {
                    let mut cycles = snapshot.cycles.clone();
                    cycles.sort_by(|a, b| {
                        b.nodes
                            .len()
                            .cmp(&a.nodes.len())
                            .then_with(|| b.cycle_rank.cmp(&a.cycle_rank))
                            .then_with(|| b.total_weight.partial_cmp(&a.total_weight).unwrap_or(std::cmp::Ordering::Equal))
                    });
                    for cycle in cycles.into_iter().take(options.limit) {
                        println!(
                            "cycle {} size={} rank={} weight={:.2} min={:.2} max={:.2}",
                            cycle.id,
                            cycle.nodes.len(),
                            cycle.cycle_rank,
                            cycle.total_weight,
                            cycle.min_weight,
                            cycle.max_weight
                        );
                        for edge in cycle.edges.iter().take(options.max_edges) {
                            println!(
                                "  {src} -> {dst} weight={weight:.2} cochange={cochange:.2} persistence={persistence:.2} cut={cut:.2}",
                                src = edge.src,
                                dst = edge.dst,
                                weight = edge.weight,
                                cochange = edge.cochange_weight,
                                persistence = edge.persistence,
                                cut = edge.cut_score
                            );
                        }
                    }
                }
                config::TopologyCommand::Star(options) => {
                    let mut neighbors = snapshot.star_neighborhood(&options.file, options.depth)?;
                    if neighbors.len() > options.limit {
                        neighbors.truncate(options.limit);
                    }
                    println!(
                        "star center={} depth={} count={}",
                        options.file,
                        options.depth,
                        neighbors.len()
                    );
                    for neighbor in neighbors {
                        println!(
                            "  {path} dist={dist} in={in_weight:.2} out={out_weight:.2} total={total:.2}",
                            path = neighbor.path,
                            dist = neighbor.distance,
                            in_weight = neighbor.in_weight,
                            out_weight = neighbor.out_weight,
                            total = neighbor.total_weight
                        );
                    }
                }
                config::TopologyCommand::Refactor(options) => {
                    let plan = snapshot.refactor_plan(RefactorOptions {
                        max_cuts_per_cycle: options.max_cuts_per_cycle,
                        max_total_cuts: options.max_total_cuts,
                        min_cut_score: options.min_cut_score,
                    });
                    let mut cuts = plan.cuts;
                    cuts.sort_by(|a, b| {
                        b.cut_score
                            .partial_cmp(&a.cut_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.weight.partial_cmp(&b.weight).unwrap_or(std::cmp::Ordering::Equal))
                    });
                    if cuts.len() > options.limit {
                        cuts.truncate(options.limit);
                    }
                    println!("cycles: {}", plan.total_cycles);
                    println!("cuts: {}", cuts.len());
                    for cut in cuts {
                        println!(
                            "  cycle={} {src} -> {dst} weight={weight:.2} persistence={persistence:.2} cut={cut_score:.2}",
                            cut.scc_id,
                            src = cut.src,
                            dst = cut.dst,
                            weight = cut.weight,
                            persistence = cut.persistence,
                            cut_score = cut.cut_score
                        );
                    }
                    for warning in plan.warnings {
                        println!("warning: {warning}");
                    }
                }
                config::TopologyCommand::Assemble(options) => {
                    let selection_opts = assembly::SelectionOptions::default();
                    let seed_sources = collect_seed_sources(
                        &project.repo_path,
                        &snapshot,
                        &selection_opts,
                        &options.files,
                        &[],
                    )?;
                    let selection = select_topology_files(
                        &snapshot,
                        &selection_opts,
                        seed_sources,
                        options.depth,
                        options.limit,
                    )?;
                    let blocks = build_topology_blocks(&db, &selection, options.per_file).await?;
                    let tokenizer = parse_tokenizer(&cfg.embedding.tokenizer)?;
                    let prompt_tokenizer_value = cfg.prompt.tokenizer.clone();
                    let prompt_tokenizer = if prompt_tokenizer_value.trim().is_empty() {
                        tokenizer
                    } else {
                        parse_tokenizer(&prompt_tokenizer_value)?
                    };
                    let budget = assembly::pipeline::BudgetOptions {
                        max_tokens: Some(cfg.embedding.context_length),
                        reserved_output_tokens: 0,
                        tokenizer: Some(prompt_tokenizer),
                    };
                    let max_blocks = options.limit.saturating_mul(options.per_file.max(1));
                    let noop = NoopReranker;
                    let ctx = assembly::AssemblyContext {
                        repo_path: &project.repo_path,
                        db: &db,
                        embedder: None,
                        reranker: &noop,
                        config: &cfg,
                        selection: selection_opts,
                    };
                    let (blocks, warnings) =
                        budget_blocks(&ctx, blocks, selection.warnings, &budget, max_blocks)
                            .await?;

                    let enriched =
                        assembly::output::enrich_blocks(&project.repo_path, &db, &blocks).await?;
                    let overview =
                        assembly::output::build_repository_overview(&project.repo_path, &db).await;
                    let payload = assembly::output::PromptPayload {
                        overview,
                        task: options.task.clone(),
                        blocks: enriched,
                        warnings,
                    };
                    let sections = resolve_sections(&options.sections);
                    let theme_value = resolve_theme_value(&options.theme, &cfg.prompt.theme);
                    let rendered = assembly::output::render_prompt(
                        &payload,
                        sections,
                        theme_value.as_deref(),
                    );
                    print!("{rendered}");
                }
            }
        }
    }

    Ok(())
}

fn project_override(command: &Command) -> Option<&str> {
    match command {
        Command::Init(_) => None,
        Command::Index(cmd) => cmd.project.project.as_deref(),
        Command::Search(cmd) => cmd.project.project.as_deref(),
        Command::Prompt(cmd) => cmd.project.project.as_deref(),
        Command::Topology(cmd) => cmd.project.project.as_deref(),
        Command::Config(_) => None,
    }
}

pub(crate) fn build_embedder(cfg: &config::Embedding) -> Result<EmbedClient> {
    let dialect = parse_dialect(cfg.dialect.as_str())?;
    let config = EmbedderConfig {
        api_key: cfg.api_key.clone(),
        base_url: cfg.url.clone(),
        timeout: Duration::from_secs(cfg.timeout_seconds),
        dialect,
        model: cfg.model.clone(),
        embedding_dim: cfg.embedding_dim,
        requests_per_minute: cfg.requests_per_minute,
        max_concurrent_requests: cfg.max_concurrent_requests,
        tokens_per_minute: cfg.tokens_per_minute,
    };

    EmbedClient::new(config).map_err(|err| eyre!(err))
}

pub(crate) fn build_reranker(cfg: &config::AppConfig) -> Result<RerankerClient> {
    let dialect = parse_dialect(cfg.reranker.dialect.as_str())?;
    let config = RerankerConfig {
        api_key: cfg
            .reranker
            .api_key
            .clone()
            .or_else(|| cfg.embedding.api_key.clone()),
        base_url: cfg.reranker.url.clone(),
        timeout: Duration::from_secs(cfg.reranker.timeout_seconds),
        dialect,
        model: cfg.reranker.model.clone(),
        instruction: cfg.reranker.instruction.clone(),
    };
    RerankerClient::new(config).map_err(|err| eyre!(err))
}

fn parse_dialect(value: &str) -> Result<ProviderDialect> {
    match value.to_ascii_lowercase().as_str() {
        "openai" => Ok(ProviderDialect::OpenAI),
        "deepinfra" => Ok(ProviderDialect::DeepInfra),
        other => Err(eyre!("unsupported embedder dialect: {}", other)),
    }
}

pub(crate) fn parse_tokenizer(value: &str) -> Result<Tokenizer> {
    let tokenizer = value
        .parse::<Tokenizer>()
        .map_err(|err| eyre!("invalid tokenizer: {err}"))?;
    tokenizer.preload().map_err(|err| eyre!(err))
}

fn split_history_path_specs(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(str::to_string)
        .collect()
}

fn normalize_seed_path(repo_root: &std::path::Path, value: &str) -> String {
    let raw = std::path::Path::new(value);
    let path = if raw.is_absolute() {
        raw.strip_prefix(repo_root).unwrap_or(raw).to_path_buf()
    } else {
        raw.to_path_buf()
    };
    let normalized = path.to_string_lossy().replace('\\', "/");
    normalized.trim_start_matches("./").to_string()
}

fn normalize_prefix(value: &str) -> String {
    value.trim().trim_start_matches("./").trim_end_matches('/').to_string()
}

fn selection_allows_path(selection: &assembly::SelectionOptions, path: &str) -> bool {
    if !selection.scope_paths.is_empty() {
        let in_scope = selection.scope_paths.iter().any(|prefix| {
            let normalized = normalize_prefix(prefix);
            !normalized.is_empty() && path.starts_with(&normalized)
        });
        if !in_scope {
            return false;
        }
    }
    for prefix in &selection.explicit_excludes {
        let normalized = normalize_prefix(prefix);
        if normalized.is_empty() {
            continue;
        }
        if path.starts_with(&normalized) {
            return false;
        }
    }
    true
}

struct SeedSources {
    sources: HashMap<String, assembly::pipeline::CandidateSource>,
    warnings: Vec<String>,
}

fn collect_seed_sources(
    repo_root: &std::path::Path,
    snapshot: &topology::TopologySnapshot,
    selection: &assembly::SelectionOptions,
    explicit: &[String],
    pinned: &[String],
) -> Result<SeedSources> {
    let mut sources = HashMap::new();
    let mut warnings = Vec::new();

    let mut insert_seed = |raw: &str, source: assembly::pipeline::CandidateSource| {
        let normalized = normalize_seed_path(repo_root, raw);
        if !snapshot.has_path(&normalized) {
            warnings.push(format!("seed not found in index: {raw}"));
            return;
        }
        if !selection_allows_path(selection, &normalized) {
            warnings.push(format!("seed outside scope/excludes: {raw}"));
            return;
        }
        sources.insert(normalized, source);
    };

    for raw in pinned {
        insert_seed(raw, assembly::pipeline::CandidateSource::Pinned);
    }
    for raw in explicit {
        insert_seed(raw, assembly::pipeline::CandidateSource::Explicit);
    }

    if sources.is_empty() {
        return Err(eyre!(
            "no valid seed files found (use --include/--pin or adjust scope)"
        ));
    }

    Ok(SeedSources { sources, warnings })
}

struct TopologySelection {
    file_paths: Vec<String>,
    weights: HashMap<String, f64>,
    seed_sources: HashMap<String, assembly::pipeline::CandidateSource>,
    warnings: Vec<String>,
}

fn select_topology_files(
    snapshot: &topology::TopologySnapshot,
    selection: &assembly::SelectionOptions,
    seeds: SeedSources,
    depth: usize,
    limit: usize,
) -> Result<TopologySelection> {
    if limit == 0 {
        return Err(eyre!("topology file limit must be > 0"));
    }

    let mut warnings = seeds.warnings;
    let seed_paths: Vec<String> = seeds.sources.keys().cloned().collect();
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
                if !selection_allows_path(selection, &neighbor.path) {
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
    for seed in &file_paths {
        if seed_set.contains(seed) {
            weights.entry(seed.clone()).and_modify(|v| {
                if *v < 1.0 {
                    *v = 1.0;
                }
            }).or_insert(1.0);
        }
    }

    Ok(TopologySelection {
        file_paths,
        weights,
        seed_sources: seeds.sources,
        warnings,
    })
}

async fn build_topology_blocks(
    db: &dyn Repository,
    selection: &TopologySelection,
    per_file: usize,
) -> Result<Vec<assembly::pipeline::ContextBlock>> {
    let chunks = db
        .chunks_for_files(&selection.file_paths, per_file)
        .await?;
    if chunks.is_empty() {
        return Err(eyre!("no chunks found for selected files"));
    }

    let mut blocks = Vec::new();
    for chunk in chunks {
        let source = selection
            .seed_sources
            .get(&chunk.file_path)
            .copied()
            .unwrap_or(assembly::pipeline::CandidateSource::Expanded);
        let score = selection.weights.get(&chunk.file_path).copied().unwrap_or(0.0);
        blocks.push(assembly::pipeline::ContextBlock {
            id: chunk.id,
            file_path: chunk.file_path,
            start_byte: chunk.start_byte,
            end_byte: chunk.end_byte,
            chunk_hash: chunk.chunk_hash,
            start_line: chunk.start_line,
            end_line: chunk.end_line,
            text: chunk.text,
            score,
            source,
        });
    }
    Ok(blocks)
}

async fn budget_blocks(
    ctx: &assembly::AssemblyContext<'_>,
    blocks: Vec<assembly::pipeline::ContextBlock>,
    warnings: Vec<String>,
    budget: &assembly::pipeline::BudgetOptions,
    max_blocks: usize,
) -> Result<(Vec<assembly::pipeline::ContextBlock>, Vec<String>)> {
    let mut arena = assembly::Arena::new();
    let input = arena.insert(assembly::pipeline::AstBlocks { blocks, warnings });
    let budget_stage = DefaultBudgetAndMerge {
        max_blocks,
        max_bytes: None,
        max_tokens: budget.max_tokens,
        reserved_output_tokens: budget.reserved_output_tokens,
        tokenizer: budget.tokenizer.clone(),
    };
    let budgeted = budget_stage.budget(ctx, &mut arena, input).await?;
    let assembled = DefaultAssembleContext
        .assemble(ctx, &mut arena, budgeted)
        .await?;
    let assembled = arena.get(assembled);
    Ok((assembled.blocks.clone(), assembled.warnings.clone()))
}

fn resolve_sections(sections: &[config::PromptSection]) -> assembly::output::PromptSections {
    if sections.is_empty() {
        assembly::output::PromptSections::all()
    } else {
        let mut selected = assembly::output::PromptSections::none();
        for section in sections {
            match section {
                config::PromptSection::Structure => selected.structure = true,
                config::PromptSection::Summary => selected.summary = true,
                config::PromptSection::Context => selected.context = true,
                config::PromptSection::Query => selected.query = true,
            }
        }
        selected
    }
}

fn resolve_theme_value(cli_theme: &str, default_theme: &str) -> Option<String> {
    let trimmed = cli_theme.trim();
    if !trimmed.is_empty() {
        return Some(trimmed.to_string());
    }
    let fallback = default_theme.trim();
    if fallback.is_empty() {
        None
    } else {
        Some(fallback.to_string())
    }
}

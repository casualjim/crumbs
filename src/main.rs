mod assembly;
mod config;
mod db;
mod embedding;
mod graph;
mod indexer;
mod issue_analysis;
mod issues;
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
use crate::issues::{IssueFilters, IssueSearchResult};
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
            let cwd = std::env::current_dir().unwrap_or_else(|_| project.repo_path.clone());
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
                let snapshot = topology::TopologySnapshot::load_with_workspace(
                    &db,
                    &project.repo_path,
                    &cwd,
                )
                .await?;
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
            let cwd = std::env::current_dir().unwrap_or_else(|_| project.repo_path.clone());
            let db = Db::open(&project.database_path, Some(cfg.embedding.embedding_dim)).await?;
            let snapshot = topology::TopologySnapshot::load_with_workspace(
                &db,
                &project.repo_path,
                &cwd,
            )
            .await?;

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
                    println!("betti_2: {}", snapshot.stats.betti_2);
                    println!("triangles: {}", snapshot.stats.triangle_count);
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
                config::TopologyCommand::Path(options) => {
                    let path = snapshot.shortest_path(&options.start, &options.end)?;
                    if path.is_empty() {
                        println!(
                            "No dependency path found between {} and {}",
                            options.start, options.end
                        );
                        return Ok(());
                    }
                    println!("path length={}", path.len().saturating_sub(1));
                    for (idx, node) in path.iter().enumerate() {
                        let package = snapshot
                            .package_for_path(node)
                            .map(|value| format!(" package={value}"))
                            .unwrap_or_default();
                        println!("  {}. {}{}", idx + 1, node, package);
                    }
                }
                config::TopologyCommand::Volumes(options) => {
                    let volumes = snapshot.feature_volumes(options.max_triangles, options.limit);
                    if volumes.is_empty() {
                        println!("No feature volumes detected.");
                        return Ok(());
                    }
                    for volume in volumes {
                        println!(
                            "volume {} nodes={} triangles={} cohesion={:.3}",
                            volume.id,
                            volume.nodes.len(),
                            volume.triangle_count,
                            volume.cohesion
                        );
                        for node in volume.nodes {
                            let package = snapshot
                                .package_for_path(&node)
                                .map(|value| format!(" package={value}"))
                                .unwrap_or_default();
                            println!("  - {}{}", node, package);
                        }
                    }
                }
                config::TopologyCommand::Layers(options) => {
                    let config = if let Some(path) = &options.config {
                        let content = std::fs::read_to_string(path)?;
                        if path.ends_with(".yaml") || path.ends_with(".yml") {
                            serde_yaml::from_str(&content)?
                        } else {
                            serde_json::from_str(&content)?
                        }
                    } else {
                        topology::layers::LayerConfig::default_config()
                    };
                    let result = topology::layers::check_layers(&snapshot, &config);
                    if result.is_valid {
                        println!("Layer check OK ({} layers).", config.layers.len());
                    } else {
                        println!(
                            "Layer violations: {} ({} orphaned nodes)",
                            result.violations.len(),
                            result.orphaned_nodes.len()
                        );
                        for violation in result.violations {
                            println!(
                                "  {} -> {} ({from} -> {to}) reason={reason}",
                                violation.from_node,
                                violation.to_node,
                                from = violation.from_layer,
                                to = violation.to_layer,
                                reason = violation.reason
                            );
                        }
                    }
                    if !result.orphaned_nodes.is_empty() {
                        println!("Orphaned nodes:");
                        for node in result.orphaned_nodes {
                            println!("  - {node}");
                        }
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
                config::TopologyCommand::Export(options) => {
                    let export = snapshot.export_graph(options.include_cochange);
                    if options.format == "json" {
                        let json = serde_json::to_string_pretty(&export)?;
                        std::fs::write(&options.output, json)?;
                    } else if options.format == "dot" {
                        let mut dot = String::from("digraph G {\n  rankdir=LR;\n");
                        for node in &export.nodes {
                            let label = if let Some(pkg) = &node.package {
                                format!("{}\\n({})", node.path, pkg)
                            } else {
                                node.path.clone()
                            };
                            dot.push_str(&format!("  \"{}\" [label=\"{}\"];\n", node.path, label));
                        }
                        for edge in &export.edges {
                            let color = match edge.kind {
                                topology::EdgeKind::Dependency => "black",
                                topology::EdgeKind::Cochange => "gray",
                            };
                            dot.push_str(&format!(
                                "  \"{}\" -> \"{}\" [label=\"{:.2}\", color=\"{}\"];\n",
                                edge.src, edge.dst, edge.weight, color
                            ));
                        }
                        dot.push_str("}\n");
                        std::fs::write(&options.output, dot)?;
                    } else {
                        return Err(eyre!("unsupported format: {}", options.format));
                    }
                    println!("Exported topology to {}", options.output);
                }
                config::TopologyCommand::Snapshot(options) => {
                    let export = snapshot.export_graph(false);
                    db.save_topology_snapshot(&options.name, &export).await?;
                    println!("Saved topology snapshot {}", options.name);
                }
                config::TopologyCommand::Diff(options) => {
                    let Some(baseline) = db.load_topology_snapshot(&options.name).await? else {
                        return Err(eyre!(
                            "snapshot '{}' not found (run topology snapshot --name <name>)",
                            options.name
                        ));
                    };
                    let current = snapshot.export_graph(false);
                    let mut baseline_edges = std::collections::HashMap::new();
                    for edge in baseline.edges {
                        baseline_edges.insert((edge.src, edge.dst, edge.kind), edge.weight);
                    }
                    let mut current_edges = std::collections::HashMap::new();
                    for edge in current.edges {
                        current_edges.insert((edge.src, edge.dst, edge.kind), edge.weight);
                    }

                    let mut added = Vec::new();
                    let mut removed = Vec::new();
                    for (edge, weight) in &current_edges {
                        if !baseline_edges.contains_key(edge) {
                            added.push((edge.clone(), *weight));
                        }
                    }
                    for (edge, weight) in &baseline_edges {
                        if !current_edges.contains_key(edge) {
                            removed.push((edge.clone(), *weight));
                        }
                    }

                    println!(
                        "Topology diff vs '{}' (added: {}, removed: {})",
                        options.name,
                        added.len(),
                        removed.len()
                    );
                    if !added.is_empty() {
                        println!("Added edges:");
                        for (edge, weight) in added.into_iter().take(options.limit) {
                            println!(
                                "  {} -> {} weight={:.2} kind={:?}",
                                edge.0, edge.1, weight, edge.2
                            );
                        }
                    }
                    if !removed.is_empty() {
                        println!("Removed edges:");
                        for (edge, weight) in removed.into_iter().take(options.limit) {
                            println!(
                                "  {} -> {} weight={:.2} kind={:?}",
                                edge.0, edge.1, weight, edge.2
                            );
                        }
                    }
                }
                config::TopologyCommand::Hotspots(options) => {
                    let hotspots =
                        snapshot.hotspots(options.limit, options.iterations, options.damping);
                    println!("Top {} hotspots:", hotspots.len());
                    for (idx, (path, score)) in hotspots.iter().enumerate() {
                        let package = snapshot
                            .package_for_path(path)
                            .map(|value| format!(" package={value}"))
                            .unwrap_or_default();
                        println!("  {}. {} score={:.4}{}", idx + 1, path, score, package);
                    }
                }
            }
        }
        Command::Issue(cmd) => {
            let cfg = config::load_config(&cli)?;
            let project = config::resolve_project(&cfg, project_override(&cli.command))?;
            let db = Db::open(&project.database_path, Some(cfg.embedding.embedding_dim)).await?;
            let jsonl_path = issues::issues_jsonl_path(&project.data_dir);

            issues::sync_from_jsonl(&db, &jsonl_path).await?;

            match &cmd.command {
                config::IssueCommand::Create(options) => {
                    let mut issue = issues::Issue::new(options.title.clone());
                    if let Some(description) = &options.description {
                        issue.description = description.clone();
                    }
                    if let Some(design) = &options.design {
                        issue.design = design.clone();
                    }
                    if let Some(acceptance) = &options.acceptance_criteria {
                        issue.acceptance_criteria = acceptance.clone();
                    }
                    if let Some(notes) = &options.notes {
                        issue.notes = notes.clone();
                    }
                    issue.status = options.status.clone();
                    issue.priority = options.priority;
                    issue.issue_type = options.issue_type.clone();
                    issue.assignee = options.assignee.clone();
                    issue.labels = issues::normalize_tags(&options.labels);
                    issue.dependencies = issues::normalize_dependencies(&options.dependencies);
                    issue.relates_to = issues::normalize_ids(&options.relates_to);
                    issue.affected_symbols = issues::normalize_symbols(&options.affected_symbols);
                    issue.external_refs = issues::normalize_external_refs(&options.external_refs);
                    issue.estimate_minutes = options.estimate_minutes;
                    issue.duplicate_of = options
                        .duplicate_of
                        .clone()
                        .filter(|value| !value.trim().is_empty());
                    issue.superseded_by = options
                        .superseded_by
                        .clone()
                        .filter(|value| !value.trim().is_empty());
                    issue.comments =
                        issues::build_comments(&options.comments, options.comment_author.as_deref());
                    db.upsert_issue(&issue).await?;
                    issues::export_to_jsonl(&db, &jsonl_path).await?;
                    println!("Created {}", issue.id);
                }
                config::IssueCommand::Update(options) => {
                    let mut issue = db
                        .get_issue(&options.id)
                        .await?
                        .ok_or_else(|| eyre!("issue not found: {}", options.id))?;
                    if let Some(title) = &options.title {
                        issue.title = title.clone();
                    }
                    if let Some(description) = &options.description {
                        issue.description = description.clone();
                    }
                    if let Some(design) = &options.design {
                        issue.design = design.clone();
                    }
                    if let Some(acceptance) = &options.acceptance_criteria {
                        issue.acceptance_criteria = acceptance.clone();
                    }
                    if let Some(notes) = &options.notes {
                        issue.notes = notes.clone();
                    }
                    if let Some(status) = &options.status {
                        issue.status = status.clone();
                    }
                    if let Some(priority) = options.priority {
                        issue.priority = priority;
                    }
                    if let Some(issue_type) = &options.issue_type {
                        issue.issue_type = issue_type.clone();
                    }
                    if let Some(assignee) = &options.assignee {
                        if assignee.trim().is_empty() {
                            issue.assignee = None;
                        } else {
                            issue.assignee = Some(assignee.clone());
                        }
                    }
                    issue.labels = issues::merge_tags(
                        &issue.labels,
                        &options.add_labels,
                        &options.remove_labels,
                    );
                    issue.dependencies = issues::merge_dependencies(
                        &issue.dependencies,
                        &options.add_dependencies,
                        &options.remove_dependencies,
                    );
                    issue.relates_to = issues::merge_ids(
                        &issue.relates_to,
                        &options.add_related,
                        &options.remove_related,
                    );
                    issue.affected_symbols = issues::merge_symbols(
                        &issue.affected_symbols,
                        &options.add_symbols,
                        &options.remove_symbols,
                    );
                    issue.external_refs = issues::merge_external_refs(
                        &issue.external_refs,
                        &options.add_external_refs,
                        &options.remove_external_refs,
                    );
                    if options.clear_estimate {
                        issue.estimate_minutes = None;
                    }
                    if let Some(value) = options.estimate_minutes {
                        issue.estimate_minutes = Some(value);
                    }
                    if options.clear_duplicate {
                        issue.duplicate_of = None;
                    }
                    if let Some(value) = &options.duplicate_of {
                        if value.trim().is_empty() {
                            issue.duplicate_of = None;
                        } else {
                            issue.duplicate_of = Some(value.clone());
                        }
                    }
                    if options.clear_superseded {
                        issue.superseded_by = None;
                    }
                    if let Some(value) = &options.superseded_by {
                        if value.trim().is_empty() {
                            issue.superseded_by = None;
                        } else {
                            issue.superseded_by = Some(value.clone());
                        }
                    }
                    let new_comments =
                        issues::build_comments(&options.add_comments, options.comment_author.as_deref());
                    if !new_comments.is_empty() {
                        issue.comments.extend(new_comments);
                    }
                    if options.restore {
                        issue.deleted_at = None;
                        issue.deleted_by = None;
                        issue.deleted_reason = None;
                    }
                    if options.mark_deleted {
                        issue.deleted_at = Some(jiff::Timestamp::now().to_string());
                    }
                    if let Some(value) = &options.deleted_by {
                        if value.trim().is_empty() {
                            issue.deleted_by = None;
                        } else {
                            issue.deleted_by = Some(value.clone());
                        }
                    }
                    if let Some(value) = &options.deleted_reason {
                        if value.trim().is_empty() {
                            issue.deleted_reason = None;
                        } else {
                            issue.deleted_reason = Some(value.clone());
                        }
                    }
                    issue.touch();
                    db.upsert_issue(&issue).await?;
                    issues::export_to_jsonl(&db, &jsonl_path).await?;
                    println!("Updated {}", issue.id);
                }
                config::IssueCommand::Close(options) => {
                    let mut issue = db
                        .get_issue(&options.id)
                        .await?
                        .ok_or_else(|| eyre!("issue not found: {}", options.id))?;
                    issue.close();
                    db.upsert_issue(&issue).await?;
                    issues::export_to_jsonl(&db, &jsonl_path).await?;
                    println!("Closed {}", issue.id);
                }
                config::IssueCommand::Get(options) => {
                    let issue = db
                        .get_issue(&options.id)
                        .await?
                        .ok_or_else(|| eyre!("issue not found: {}", options.id))?;
                    print_issue(&issue);
                }
                config::IssueCommand::List(options) => {
                    let filters = IssueFilters {
                        status: options.status.clone(),
                        assignee: options.assignee.clone(),
                        label: options.label.clone(),
                        issue_type: options.issue_type.clone(),
                        priority: options.priority,
                        limit: options.limit,
                    };
                    let issues = db.list_issues(filters).await?;
                    for issue in issues {
                        print_issue_summary(&issue);
                    }
                }
                config::IssueCommand::Search(options) => {
                    let results = db.search_issues(&options.query, options.limit).await?;
                    for IssueSearchResult { issue, score } in results {
                        println!(
                            "{id} [{status}] p{priority} score={score:.3} {title}",
                            id = issue.id,
                            status = issue.status,
                            priority = issue.priority,
                            score = score,
                            title = issue.title
                        );
                    }
                }
                config::IssueCommand::Context(options) => {
                    let issue = db
                        .get_issue(&options.id)
                        .await?
                        .ok_or_else(|| eyre!("issue not found: {}", options.id))?;
                    let selection_opts = assembly::SelectionOptions::default();
                    let theme_value = resolve_theme_value(&options.theme, &cfg.prompt.theme);
                    let sections = resolve_sections(&options.sections);

                    if issue.affected_symbols.is_empty() {
                        let tokenizer = parse_tokenizer(&cfg.embedding.tokenizer)?;
                        let embedder = build_embedder(&cfg.embedding)?;
                        let reranker = build_reranker(&cfg)?;
                        let ctx = assembly::AssemblyContext {
                            repo_path: &project.repo_path,
                            db: &db,
                            embedder: Some(&embedder),
                            reranker: &reranker as &dyn RerankingProvider,
                            config: &cfg,
                            selection: selection_opts,
                        };
                        let budget = assembly::pipeline::BudgetOptions {
                            max_tokens: Some(cfg.embedding.context_length),
                            reserved_output_tokens: 0,
                            tokenizer: Some(tokenizer.clone()),
                        };
                        let pipeline =
                            assembly::pipeline::default_pipeline(&cfg, budget);
                        let mut arena = assembly::Arena::new();
                        let input = arena.insert(assembly::pipeline::QueryInput {
                            text: issue.summary_query(),
                        });
                        let handle = pipeline.run(&ctx, &mut arena, input).await?;
                        let assembled = arena.get(handle);
                        let enriched =
                            assembly::output::enrich_blocks(&project.repo_path, &db, &assembled.blocks).await?;
                        let overview =
                            assembly::output::build_repository_overview(&project.repo_path, &db).await;
                        let payload = assembly::output::PromptPayload {
                            overview,
                            task: issue.title.clone(),
                            blocks: enriched,
                            warnings: assembled.warnings.clone(),
                        };
                        let rendered =
                            assembly::output::render_prompt(&payload, sections, theme_value.as_deref());
                        print!("{rendered}");
                    } else {
                        let cwd =
                            std::env::current_dir().unwrap_or_else(|_| project.repo_path.clone());
                        let snapshot = topology::TopologySnapshot::load_with_workspace(
                            &db,
                            &project.repo_path,
                            &cwd,
                        )
                        .await?;
                        let seed_sources = collect_seed_sources(
                            &project.repo_path,
                            &snapshot,
                            &selection_opts,
                            &issue.affected_symbols,
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
                            task: issue.title.clone(),
                            blocks: enriched,
                            warnings,
                        };
                        let rendered =
                            assembly::output::render_prompt(&payload, sections, theme_value.as_deref());
                        print!("{rendered}");
                    }
                }
                config::IssueCommand::Next(options) => {
                    let issues = db.list_all_issues().await?;
                    let suggestions = issue_analysis::suggest_next_tasks(
                        &issues,
                        options.assignee.as_deref(),
                        options.limit,
                    )?;
                    if suggestions.is_empty() {
                        println!("No ready issues found.");
                    } else {
                        for suggestion in suggestions {
                            let issue = suggestion.issue;
                            let blockers = if suggestion.blockers.is_empty() {
                                String::new()
                            } else {
                                format!(" blockers={}", suggestion.blockers.join(","))
                            };
                            println!(
                                "{id} [{status}] p{priority} {title} ({reason}){blockers}",
                                id = issue.id,
                                status = issue.status,
                                priority = issue.priority,
                                title = issue.title,
                                reason = suggestion.reason,
                                blockers = blockers
                            );
                        }
                    }
                }
                config::IssueCommand::Stale(options) => {
                    let issues = db.list_all_issues().await?;
                    let stale = issue_analysis::find_stale_issues(
                        &issues,
                        options.days,
                        options.assignee.as_deref(),
                        options.status.as_deref(),
                        options.limit,
                    )?;
                    if stale.is_empty() {
                        println!("No stale issues found.");
                    } else {
                        for entry in stale {
                            println!(
                                "{id} [{status}] p{priority} stale={days}d {title}",
                                id = entry.issue.id,
                                status = entry.issue.status,
                                priority = entry.issue.priority,
                                days = entry.days_inactive,
                                title = entry.issue.title
                            );
                        }
                    }
                }
                config::IssueCommand::Triage(options) => {
                    let mut updated = 0usize;
                    for id in &options.ids {
                        let mut issue = db
                            .get_issue(id)
                            .await?
                            .ok_or_else(|| eyre!("issue not found: {}", id))?;
                        if let Some(status) = &options.status {
                            issue.status = status.clone();
                        }
                        if let Some(priority) = options.priority {
                            issue.priority = priority;
                        }
                        if let Some(assignee) = &options.assignee {
                            if assignee.trim().is_empty() {
                                issue.assignee = None;
                            } else {
                                issue.assignee = Some(assignee.clone());
                            }
                        }
                        issue.labels = issues::merge_tags(
                            &issue.labels,
                            &options.add_labels,
                            &options.remove_labels,
                        );
                        issue.touch();
                        db.upsert_issue(&issue).await?;
                        updated += 1;
                    }
                    if updated > 0 {
                        issues::export_to_jsonl(&db, &jsonl_path).await?;
                    }
                    println!("Triaged {} issues.", updated);
                }
                config::IssueCommand::Duplicates(options) => {
                    let issues = db.list_all_issues().await?;
                    let duplicates = issue_analysis::find_duplicates(
                        &issues,
                        options.threshold,
                        options.limit,
                    );
                    if duplicates.is_empty() {
                        println!("No duplicate issues found.");
                    } else {
                        for pair in duplicates {
                            println!(
                                "{a} <-> {b} similarity={:.2} ({title_a} / {title_b})",
                                pair.similarity,
                                a = pair.issue_a.id,
                                b = pair.issue_b.id,
                                title_a = pair.issue_a.title,
                                title_b = pair.issue_b.title
                            );
                        }
                    }
                }
                config::IssueCommand::Related(options) => {
                    let issues = db.list_all_issues().await?;
                    if let Some(file) = &options.file {
                        let related =
                            issue_analysis::related_issues_for_file(&issues, file, options.limit);
                        if related.is_empty() {
                            println!("No related issues found for {}", file);
                        } else {
                            for entry in related {
                                println!(
                                    "{id} [{status}] score={score:.2} {title} reasons={reasons}",
                                    id = entry.issue.id,
                                    status = entry.issue.status,
                                    score = entry.score,
                                    title = entry.issue.title,
                                    reasons = entry.reasons.join(",")
                                );
                            }
                        }
                    } else if let Some(id) = &options.issue {
                        let issue = db
                            .get_issue(id)
                            .await?
                            .ok_or_else(|| eyre!("issue not found: {}", id))?;
                        let related = issue_analysis::related_issues_for_issue(
                            &issues,
                            &issue,
                            options.limit,
                        );
                        if related.is_empty() {
                            println!("No related issues found for {}", issue.id);
                        } else {
                            for entry in related {
                                println!(
                                    "{id} [{status}] score={score:.2} {title} reasons={reasons}",
                                    id = entry.issue.id,
                                    status = entry.issue.status,
                                    score = entry.score,
                                    title = entry.issue.title,
                                    reasons = entry.reasons.join(",")
                                );
                            }
                        }
                    } else {
                        return Err(eyre!("--file or --issue is required"));
                    }
                }
                config::IssueCommand::Infer(options) => match &options.command {
                    config::IssueInferCommand::Error(inner) => {
                        let draft = issue_analysis::infer_issue_from_error(&inner.message);
                        let issues = db.list_all_issues().await?;
                        println!("Suggested issue:");
                        println!("  title: {}", draft.title);
                        println!("  type: {}", draft.issue_type);
                        if !draft.labels.is_empty() {
                            println!("  labels: {}", draft.labels.join(", "));
                        }
                        if !draft.affected_symbols.is_empty() {
                            println!(
                                "  affected_symbols: {}",
                                draft.affected_symbols.join(", ")
                            );
                        }
                        println!("  description: {}", draft.description);

                        let results = db.search_issues(&inner.message, inner.limit).await?;
                        if !results.is_empty() {
                            println!("\nMatching issues:");
                            for result in results {
                                println!(
                                    "  {id} [{status}] score={score:.2} {title}",
                                    id = result.issue.id,
                                    status = result.issue.status,
                                    score = result.score,
                                    title = result.issue.title
                                );
                            }
                        }

                        let related =
                            issue_analysis::group_related_issues_by_file(&issues, &draft.affected_symbols, inner.limit);
                        for (file, entries) in related {
                            println!("\nRelated issues for {}:", file);
                            for entry in entries {
                                println!(
                                    "  {id} [{status}] score={score:.2} {title}",
                                    id = entry.issue.id,
                                    status = entry.issue.status,
                                    score = entry.score,
                                    title = entry.issue.title
                                );
                            }
                        }
                    }
                    config::IssueInferCommand::Diff(inner) => {
                        let diff_text = if let Some(path) = &inner.path {
                            std::fs::read_to_string(path)?
                        } else {
                            let mut args = vec!["diff", "--name-only"];
                            if let Some(range) = &inner.range {
                                args.push(range);
                            }
                            let output = std::process::Command::new("git")
                                .args(args)
                                .current_dir(&project.repo_path)
                                .output()?;
                            String::from_utf8_lossy(&output.stdout).to_string()
                        };
                        let draft = issue_analysis::infer_issue_from_diff(&diff_text);
                        println!("Suggested issue:");
                        println!("  title: {}", draft.title);
                        println!("  type: {}", draft.issue_type);
                        if !draft.labels.is_empty() {
                            println!("  labels: {}", draft.labels.join(", "));
                        }
                        if !draft.affected_symbols.is_empty() {
                            println!(
                                "  affected_symbols: {}",
                                draft.affected_symbols.join(", ")
                            );
                        }
                        println!("  description:\n{}", draft.description);

                        let issues = db.list_all_issues().await?;
                        let related =
                            issue_analysis::group_related_issues_by_file(&issues, &draft.affected_symbols, inner.limit);
                        for (file, entries) in related {
                            println!("\nRelated issues for {}:", file);
                            for entry in entries {
                                println!(
                                    "  {id} [{status}] score={score:.2} {title}",
                                    id = entry.issue.id,
                                    status = entry.issue.status,
                                    score = entry.score,
                                    title = entry.issue.title
                                );
                            }
                        }
                    }
                    config::IssueInferCommand::Todo(inner) => {
                        let content = std::fs::read_to_string(&inner.file)?;
                        let drafts = issue_analysis::infer_issues_from_todos(
                            &inner.file,
                            &content,
                            inner.limit,
                        );
                        if drafts.is_empty() {
                            println!("No TODOs found in {}", inner.file);
                        } else {
                            println!("Suggested issues:");
                            for draft in drafts {
                                println!("  title: {}", draft.title);
                                println!("  type: {}", draft.issue_type);
                                if !draft.labels.is_empty() {
                                    println!("  labels: {}", draft.labels.join(", "));
                                }
                                println!("  affected_symbols: {}", draft.affected_symbols.join(", "));
                                println!("  description: {}", draft.description);
                                println!("---");
                            }
                        }
                        let issues = db.list_all_issues().await?;
                        let related =
                            issue_analysis::related_issues_for_file(&issues, &inner.file, inner.limit);
                        if !related.is_empty() {
                            println!("\nRelated issues for {}:", inner.file);
                            for entry in related {
                                println!(
                                    "  {id} [{status}] score={score:.2} {title}",
                                    id = entry.issue.id,
                                    status = entry.issue.status,
                                    score = entry.score,
                                    title = entry.issue.title
                                );
                            }
                        }
                    }
                },
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
        Command::Issue(cmd) => cmd.project.project.as_deref(),
        Command::Config(_) => None,
    }
}

fn print_issue_summary(issue: &issues::Issue) {
    let labels = if issue.labels.is_empty() {
        String::new()
    } else {
        format!(" labels={}", issue.labels.join(","))
    };
    let assignee = issue
        .assignee
        .as_ref()
        .map(|value| format!(" assignee={value}"))
        .unwrap_or_default();
    println!(
        "{id} [{status}] p{priority} {title}{assignee}{labels}",
        id = issue.id,
        status = issue.status,
        priority = issue.priority,
        title = issue.title,
        assignee = assignee,
        labels = labels
    );
}

fn print_issue(issue: &issues::Issue) {
    println!("id: {}", issue.id);
    println!("title: {}", issue.title);
    println!("status: {}", issue.status);
    println!("priority: {}", issue.priority);
    println!("type: {}", issue.issue_type);
    if let Some(assignee) = &issue.assignee {
        println!("assignee: {assignee}");
    }
    if !issue.labels.is_empty() {
        println!("labels: {}", issue.labels.join(", "));
    }
    if !issue.dependencies.is_empty() {
        println!("dependencies: {}", format_dependencies(&issue.dependencies));
    }
    if !issue.relates_to.is_empty() {
        println!("relates_to: {}", issue.relates_to.join(", "));
    }
    if !issue.affected_symbols.is_empty() {
        println!("affected_symbols: {}", issue.affected_symbols.join(", "));
    }
    if !issue.external_refs.is_empty() {
        println!("external_refs: {}", format_external_refs(&issue.external_refs));
    }
    if let Some(estimate) = issue.estimate_minutes {
        println!("estimate_minutes: {}", estimate);
    }
    if let Some(duplicate_of) = &issue.duplicate_of {
        println!("duplicate_of: {}", duplicate_of);
    }
    if let Some(superseded_by) = &issue.superseded_by {
        println!("superseded_by: {}", superseded_by);
    }
    if let Some(deleted_at) = &issue.deleted_at {
        println!("deleted_at: {}", deleted_at);
    }
    if let Some(deleted_by) = &issue.deleted_by {
        println!("deleted_by: {}", deleted_by);
    }
    if let Some(deleted_reason) = &issue.deleted_reason {
        println!("deleted_reason: {}", deleted_reason);
    }
    if !issue.description.trim().is_empty() {
        println!("\ndescription:\n{}", issue.description);
    }
    if !issue.design.trim().is_empty() {
        println!("\ndesign:\n{}", issue.design);
    }
    if !issue.acceptance_criteria.trim().is_empty() {
        println!("\nacceptance:\n{}", issue.acceptance_criteria);
    }
    if !issue.notes.trim().is_empty() {
        println!("\nnotes:\n{}", issue.notes);
    }
    if !issue.comments.is_empty() {
        println!("\ncomments:");
        for comment in &issue.comments {
            let author = comment.author.as_deref().unwrap_or("unknown");
            println!("  {} {author}: {}", comment.created_at, comment.body);
        }
    }
    println!("created_at: {}", issue.created_at);
    println!("updated_at: {}", issue.updated_at);
    if let Some(closed_at) = &issue.closed_at {
        println!("closed_at: {}", closed_at);
    }
}

fn format_dependencies(values: &[issues::IssueDependency]) -> String {
    values
        .iter()
        .map(|dep| {
            if dep.kind.trim().is_empty() {
                dep.id.clone()
            } else {
                format!("{}:{}", dep.id, dep.kind)
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_external_refs(values: &[issues::IssueExternalRef]) -> String {
    values
        .iter()
        .map(|reference| {
            if reference.kind.trim().is_empty() {
                reference.value.clone()
            } else {
                format!("{}:{}", reference.kind, reference.value)
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
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

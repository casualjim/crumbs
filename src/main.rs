mod assembly;
mod config;
mod db;
mod embedding;
mod graph;
mod indexer;
mod logging;
mod reqwestx;
mod repository;
mod reranker;
mod search;
#[cfg(test)]
mod test_support;

use std::time::Duration;

use clap::Parser;
use eyre::{Result, eyre};
use text_chunking::Tokenizer;

use crate::config::{Cli, Command};
use crate::db::Db;
use crate::embedding::{Client as EmbedClient, EmbedderConfig, ProviderDialect};
use crate::indexer::{Indexer, IndexerConfig};
use crate::reranker::{Client as RerankerClient, RerankerConfig, RerankingProvider};

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
            let db = Db::open(&project.database_path, Some(cfg.embedding.embedding_dim))?;
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
            let db = Db::open(&project.database_path, Some(cfg.embedding.embedding_dim))?;
            let reranker = if cfg.search.rerank {
                Some(build_reranker(&cfg)?)
            } else {
                None
            };
            let reranker_ref = reranker
                .as_ref()
                .map(|client| client as &dyn RerankingProvider);
            let mut search_config =
                search::SearchConfig::new(cfg.search.limit, cfg.search.hybrid_weight);
            search_config.path_prefixes = cfg.search.path_prefixes.clone();
            search_config.file_exts = cfg.search.file_exts.clone();
            search_config.decompose = cfg.search.decompose;
            search_config.rerank = cfg.search.rerank;
            search_config.rerank_min_score = cfg.search.rerank_min_score;
            let results =
                search::search(&db, &embedder, reranker_ref, &tokenizer, &cmd.query, search_config)
                    .await?;
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
            let embedder = build_embedder(&cfg.embedding)?;
            let db = Db::open(&project.database_path, Some(cfg.embedding.embedding_dim))?;
            let reranker = if cfg.search.rerank {
                Some(build_reranker(&cfg)?)
            } else {
                None
            };

            let ctx = assembly::AssemblyContext {
                repo_path: &project.repo_path,
                db: &db,
                embedder: Some(&embedder),
                reranker: reranker
                    .as_ref()
                    .map(|client| client as &dyn RerankingProvider),
                config: &cfg,
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
            let budget = assembly::pipeline::BudgetOptions {
                max_tokens,
                reserved_output_tokens: cmd.reserved_output_tokens,
                tokenizer: Some(prompt_tokenizer),
            };
            let pipeline = assembly::pipeline::default_pipeline(&cfg, budget);
            let mut arena = assembly::Arena::new();
            let input = arena.insert(assembly::pipeline::QueryInput {
                text: cmd.task.clone(),
            });
            let handle = pipeline.run(&ctx, &mut arena, input).await?;
            let assembled = arena.get(handle);

            let enriched =
                assembly::output::enrich_blocks(&project.repo_path, &db, &assembled.blocks).await?;
            let overview = assembly::output::build_repository_overview(&project.repo_path, &db);
            let payload = assembly::output::PromptPayload {
                overview,
                task: cmd.task.clone(),
                blocks: enriched,
            };
            let format = match cmd.format {
                config::PromptFormat::Xml => assembly::output::PromptFormat::Xml,
                config::PromptFormat::Markdown => assembly::output::PromptFormat::Markdown,
            };
            let rendered = assembly::output::render_prompt(format, &payload);
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
    }

    Ok(())
}

fn project_override(command: &Command) -> Option<&str> {
    match command {
        Command::Init(_) => None,
        Command::Index(cmd) => cmd.project.project.as_deref(),
        Command::Search(cmd) => cmd.project.project.as_deref(),
        Command::Prompt(cmd) => cmd.project.project.as_deref(),
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

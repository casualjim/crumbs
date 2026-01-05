mod assembly;
mod config;
mod db;
mod embedding;
mod graph;
mod indexer;
mod repository;
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

#[tokio::main]
async fn main() -> Result<()> {
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

            let embedder = build_embedder(&cfg.embedding, tokenizer.clone())?;
            let db = Db::open(&project.database_path, Some(cfg.embedding.embedding_dim))?;
            let index_config = IndexerConfig {
                repo_path: project.repo_path.clone(),
                max_chunk_size: cfg.chunking.max_chunk_size,
                overlap_percentage: cfg.chunking.overlap,
                tokenizer: tokenizer.clone(),
                max_parallel: cfg.chunking.max_parallel,
                max_file_size: Some(cfg.chunking.max_file_size),
                large_file_threads: cfg.chunking.large_file_threads,
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
            let embedder = build_embedder(&cfg.embedding, tokenizer)?;
            let project = config::resolve_project(&cfg, project_override(&cli.command))?;
            let db = Db::open(&project.database_path, Some(cfg.embedding.embedding_dim))?;
            let results = search::search(
                &db,
                &embedder,
                &cmd.query,
                cfg.search.limit,
                cfg.search.hybrid_weight,
            )
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
            let embedder = build_embedder(&cfg.embedding, tokenizer)?;
            let db = Db::open(&project.database_path, Some(cfg.embedding.embedding_dim))?;

            let ctx = assembly::AssemblyContext {
                repo_path: &project.repo_path,
                db: &db,
                embedder: Some(&embedder),
                config: &cfg,
            };

            let pipeline = assembly::pipeline::default_pipeline(&cfg);
            let mut arena = assembly::Arena::new();
            let input = arena.insert(assembly::pipeline::QueryInput {
                text: cmd.task.clone(),
            });
            let handle = pipeline.run(&ctx, &mut arena, input).await?;
            let assembled = arena.get(handle);

            println!("<context>");
            for block in &assembled.blocks {
                let source = match block.source {
                    assembly::pipeline::CandidateSource::Primary => "primary",
                    assembly::pipeline::CandidateSource::Expanded => "expanded",
                };
                println!(
                    "<file path=\"{}\" start_byte=\"{}\" end_byte=\"{}\" source=\"{}\">",
                    block.file_path, block.start_byte, block.end_byte, source
                );
                println!("{}", block.text);
                println!("</file>");
            }
            println!("</context>");
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
                let tokenizer = parse_tokenizer(&cfg.embedding.tokenizer)?;
                let _ = build_embedder(&cfg.embedding, tokenizer)?;
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

pub(crate) fn build_embedder(cfg: &config::Embedding, tokenizer: Tokenizer) -> Result<EmbedClient> {
    let dialect = parse_dialect(cfg.dialect.as_str())?;
    let config = EmbedderConfig {
        api_key: cfg.api_key.clone(),
        base_url: cfg.url.clone(),
        timeout: Duration::from_secs(cfg.timeout_seconds),
        dialect,
        model: cfg.model.clone(),
        tokenizer,
        embedding_dim: cfg.embedding_dim,
        context_length: cfg.context_length,
        max_batch_size: cfg.max_batch_size,
        tokens_per_minute: cfg.tokens_per_minute,
    };

    EmbedClient::new(config).map_err(|err| eyre!(err))
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

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
        Command::Index(_) => {
            let cfg = config::load_config(&cli)?;
            let tokenizer = parse_tokenizer(&cfg.chunking.tokenizer)?;
            let embedder = build_embedder(&cfg.embedding)?;
            let project = config::resolve_project(&cfg, project_override(&cli.command))?;
            let db = Db::open(&project.database_path, Some(cfg.embedding.embedding_dim))?;
            let config = IndexerConfig {
                repo_path: project.repo_path,
                max_chunk_size: cfg.chunking.max_chunk_size,
                overlap_percentage: cfg.chunking.overlap,
                tokenizer,
                max_parallel: cfg.chunking.max_parallel,
                max_file_size: Some(cfg.chunking.max_file_size),
                large_file_threads: cfg.chunking.large_file_threads,
            };
            let indexer = Indexer::new(db, embedder, config);
            indexer.index().await?;
        }
        Command::Graph(_) => {
            let cfg = config::load_config(&cli)?;
            let tokenizer = parse_tokenizer(&cfg.chunking.tokenizer)?;
            let project = config::resolve_project(&cfg, project_override(&cli.command))?;
            let db = Db::open(&project.database_path, Some(cfg.embedding.embedding_dim))?;
            let config = graph::GraphConfig {
                repo_path: project.repo_path,
                max_chunk_size: cfg.chunking.max_chunk_size,
                overlap_percentage: cfg.chunking.overlap,
                tokenizer,
                max_parallel: cfg.chunking.max_parallel,
                max_file_size: Some(cfg.chunking.max_file_size),
                large_file_threads: cfg.chunking.large_file_threads,
                history_depth: cfg.history.depth,
                history_commit_size_limit_ratio: cfg.history.commit_size_limit_ratio,
                history_multi_parents: cfg.history.multi_parents,
                history_issue_regex: cfg.history.issue_regex.clone(),
                history_commit_exclude_regex: cfg.history.commit_exclude_regex.clone(),
                history_author_exclude_regex: cfg.history.author_exclude_regex.clone(),
                history_path_specs: split_history_path_specs(&cfg.history.path_specs),
            };
            let indexer = graph::GraphIndexer::new(db, config);
            indexer.index().await?;
        }
        Command::Search(cmd) => {
            let cfg = config::load_config(&cli)?;
            let embedder = build_embedder(&cfg.embedding)?;
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
        Command::Config(cmd) => match &cmd.command {
            config::ConfigCommand::Init(init) => {
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
            config::ConfigCommand::EnsureProject(ensure) => {
                let result = config::ensure_project(ensure, cli.config_file.as_deref())?;
                if result.created {
                    println!(
                        "Created config at {} and added project '{}'",
                        result.config_path.display(),
                        result.project_name
                    );
                } else if result.updated {
                    println!(
                        "Ensured project '{}' in {}",
                        result.project_name,
                        result.config_path.display()
                    );
                } else {
                    println!(
                        "Project '{}' already exists in {}",
                        result.project_name,
                        result.config_path.display()
                    );
                }
            }
        },
    }

    Ok(())
}

fn project_override(command: &Command) -> Option<&str> {
    match command {
        Command::Index(cmd) => cmd.project.project.as_deref(),
        Command::Graph(cmd) => cmd.project.project.as_deref(),
        Command::Search(cmd) => cmd.project.project.as_deref(),
        Command::Config(_) => None,
    }
}

fn build_embedder(cfg: &config::Embedding) -> Result<EmbedClient> {
    let dialect = parse_dialect(cfg.dialect.as_str())?;
    let config = EmbedderConfig {
        api_key: cfg.api_key.clone(),
        base_url: cfg.url.clone(),
        timeout: Duration::from_secs(cfg.timeout_seconds),
        dialect,
        model: cfg.model.clone(),
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

fn parse_tokenizer(value: &str) -> Result<Tokenizer> {
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

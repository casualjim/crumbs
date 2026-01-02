mod config;
mod db;
mod embedding;
mod graph;
mod indexer;
mod search;
#[cfg(test)]
mod tests;

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
  let cfg = config::load_config(&cli)?;

  match &cli.command {
    Command::Index(_) => {
      let tokenizer = parse_tokenizer(&cfg.chunking.tokenizer)?;
      let embedder = build_embedder(&cfg.embedding)?;
      let db = Db::open(&cfg.database.path, Some(cfg.database.embedding_dim))?;
      let config = IndexerConfig {
        repo_path: cfg.index.repo.clone(),
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
      let tokenizer = parse_tokenizer(&cfg.chunking.tokenizer)?;
      let db = Db::open(&cfg.database.path, Some(cfg.database.embedding_dim))?;
      let config = graph::GraphConfig {
        repo_path: cfg.graph.repo.clone(),
        max_chunk_size: cfg.chunking.max_chunk_size,
        overlap_percentage: cfg.chunking.overlap,
        tokenizer,
        max_parallel: cfg.chunking.max_parallel,
        max_file_size: Some(cfg.chunking.max_file_size),
        large_file_threads: cfg.chunking.large_file_threads,
      };
      let indexer = graph::GraphIndexer::new(db, config);
      indexer.index().await?;
    }
    Command::Search(cmd) => {
      let embedder = build_embedder(&cfg.embedding)?;
      let db = Db::open(&cfg.database.path, Some(cfg.database.embedding_dim))?;
      let results = search::search(&db, &embedder, &cmd.query, cfg.search.limit).await?;
      for (idx, result) in results.iter().enumerate() {
        let score = 1.0 - result.distance;
        println!(
          "{idx}. {path}:{start}-{end} score={score:.4}\n{text}\n",
          idx = idx + 1,
          path = result.file_path,
          start = result.start_byte,
          end = result.end_byte,
          score = score,
          text = result.text
        );
      }
    }
  }

  Ok(())
}

fn build_embedder(cfg: &config::Embedding) -> Result<EmbedClient> {
  let dialect = parse_dialect(cfg.dialect.as_str())?;
  let config = EmbedderConfig {
    api_key: cfg.api_key.clone(),
    base_url: cfg.base_url.clone(),
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

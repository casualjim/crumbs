use std::env;
use std::path::Path;
use std::time::Duration;

use eyre::Result;
use tempfile::TempDir;
use text_chunking::Tokenizer;

use crate::db::Db;
use crate::embedding::{Client as EmbedClient, EmbedderConfig, EmbeddingProvider, ProviderDialect};
use crate::graph::{GraphConfig, GraphIndexer};
use crate::indexer::{Indexer, IndexerConfig};
use crate::search;

struct EmbeddingEnv {
  base_url: String,
  api_key: Option<String>,
  model: String,
  dialect: ProviderDialect,
  timeout_seconds: u64,
  embedding_dim: usize,
  context_length: usize,
  max_batch_size: usize,
  tokens_per_minute: u32,
}

impl EmbeddingEnv {
  fn load() -> Self {
    let base_url = required_env("EMBEDDER_BASE_URL");
    let model = required_env("EMBEDDER_MODEL");
    let embedding_dim = required_env("EMBEDDING_DIM")
      .parse::<usize>()
      .unwrap_or_else(|_| panic!("EMBEDDING_DIM must be a usize"));

    let dialect = env::var("EMBEDDER_DIALECT").unwrap_or_else(|_| "openai".to_string());
    let dialect = match dialect.to_ascii_lowercase().as_str() {
      "openai" => ProviderDialect::OpenAI,
      "deepinfra" => ProviderDialect::DeepInfra,
      other => panic!("Unsupported EMBEDDER_DIALECT: {other}"),
    };

    let timeout_seconds = env::var("EMBEDDER_TIMEOUT_SECONDS")
      .ok()
      .and_then(|value| value.parse::<u64>().ok())
      .unwrap_or(10);

    let context_length = env::var("EMBEDDER_CONTEXT_LENGTH")
      .ok()
      .and_then(|value| value.parse::<usize>().ok())
      .unwrap_or(8192);

    let max_batch_size = env::var("EMBEDDER_MAX_BATCH_SIZE")
      .ok()
      .and_then(|value| value.parse::<usize>().ok())
      .unwrap_or(32);

    let tokens_per_minute = env::var("EMBEDDER_TOKENS_PER_MINUTE")
      .ok()
      .and_then(|value| value.parse::<u32>().ok())
      .unwrap_or(0);

    Self {
      base_url,
      api_key: env::var("EMBEDDER_API_KEY").ok(),
      model,
      dialect,
      timeout_seconds,
      embedding_dim,
      context_length,
      max_batch_size,
      tokens_per_minute,
    }
  }

  fn build_client(&self) -> Result<EmbedClient> {
    let config = EmbedderConfig {
      api_key: self.api_key.clone().map(secrecy::SecretString::from),
      base_url: self.base_url.clone(),
      timeout: Duration::from_secs(self.timeout_seconds),
      dialect: self.dialect.clone(),
      model: self.model.clone(),
      embedding_dim: self.embedding_dim,
      context_length: self.context_length,
      max_batch_size: self.max_batch_size,
      tokens_per_minute: self.tokens_per_minute,
    };
    EmbedClient::new(config)
  }
}

fn required_env(name: &str) -> String {
  env::var(name).unwrap_or_else(|_| panic!("Missing required env var: {name}"))
}

fn write_fixture_repo(root: &Path) -> Result<()> {
  std::fs::create_dir_all(root.join("src"))?;

  std::fs::write(
    root.join("src/lib.rs"),
    "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\
     pub fn run() -> i32 { add(1, 2) }\n",
  )?;

  std::fs::write(
    root.join("src/app.py"),
    "def add(a, b):\n    return a + b\n\n\
def run():\n    return add(1, 2)\n",
  )?;

  std::fs::write(
    root.join("src/app.go"),
    "package main\n\nfunc add(a int, b int) int { return a + b }\n\
func run() int { return add(1, 2) }\n",
  )?;

  std::fs::write(
    root.join("src/app.ts"),
    "export function add(a: number, b: number) { return a + b; }\n\
export function run() { return add(1, 2); }\n",
  )?;

  std::fs::write(
    root.join("src/app.js"),
    "function add(a, b) { return a + b; }\n\
function run() { return add(1, 2); }\n\
module.exports = { add, run };\n",
  )?;

  Ok(())
}

#[tokio::test]
async fn db_enforces_embedding_dim() -> Result<()> {
  let env = EmbeddingEnv::load();
  let dir = TempDir::new()?;
  let db_path = dir.path().join("context.duckdb");

  let _db = Db::open(&db_path, Some(env.embedding_dim))?;
  let mismatch = Db::open(&db_path, Some(env.embedding_dim + 1));
  assert!(mismatch.is_err(), "expected embedding_dim mismatch error");
  Ok(())
}

#[tokio::test]
async fn graph_build_populates_symbols_and_references() -> Result<()> {
  let env = EmbeddingEnv::load();
  let dir = TempDir::new()?;
  write_fixture_repo(dir.path())?;

  let db_path = dir.path().join("context.duckdb");
  let db = Db::open(&db_path, Some(env.embedding_dim))?;

  let config = GraphConfig {
    repo_path: dir.path().to_path_buf(),
    max_chunk_size: 1500,
    overlap_percentage: 0.2,
    tokenizer: Tokenizer::Characters,
    max_parallel: 4,
    max_file_size: Some(5 * 1024 * 1024),
    large_file_threads: 2,
  };
  let indexer = GraphIndexer::new(db, config);
  indexer.index().await?;

  let conn = duckdb::Connection::open(&db_path)?;
  let symbols: i64 = conn.query_row("SELECT COUNT(*) FROM symbols", [], |row| row.get(0))?;
  let references: i64 =
    conn.query_row("SELECT COUNT(*) FROM symbol_references", [], |row| row.get(0))?;

  assert!(symbols > 0, "expected symbols to be populated");
  assert!(references > 0, "expected references to be populated");
  Ok(())
}

#[tokio::test]
async fn end_to_end_index_and_search() -> Result<()> {
  let env = EmbeddingEnv::load();
  let dir = TempDir::new()?;
  write_fixture_repo(dir.path())?;

  let db_path = dir.path().join("context.duckdb");
  let embedder = env.build_client()?;
  let db = Db::open(&db_path, Some(env.embedding_dim))?;
  let config = IndexerConfig {
    repo_path: dir.path().to_path_buf(),
    max_chunk_size: 512,
    overlap_percentage: 0.1,
    tokenizer: Tokenizer::Characters,
    max_parallel: 2,
    max_file_size: Some(5 * 1024 * 1024),
    large_file_threads: 2,
  };
  let indexer = Indexer::new(db, embedder, config);
  indexer.index().await?;

  let embedder = env.build_client()?;
  let db = Db::open(&db_path, Some(env.embedding_dim))?;
  let results = search::search(&db, &embedder, "add numbers", 5).await?;

  assert!(!results.is_empty(), "expected search to return results");
  Ok(())
}

#[tokio::test]
async fn real_embedder_returns_expected_dimensions() -> Result<()> {
  let env = EmbeddingEnv::load();
  let embedder = env.build_client()?;
  let output = embedder
    .embed(&[crate::embedding::EmbeddingInput {
      text: "hello world".to_string(),
      token_count: None,
    }])
    .await?;

  assert_eq!(
    output.embeddings.first().map(|v| v.len()),
    Some(env.embedding_dim),
    "expected embedding dimension to match EMBEDDING_DIM"
  );
  Ok(())
}

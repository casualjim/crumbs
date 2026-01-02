use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use confique::Config as _;
use confique::Layer as _;
use secrecy::SecretString;

#[derive(confique::Config, Debug, Clone)]
pub struct AppConfig {
  #[config(nested)]
  pub embedding: Embedding,
  #[config(nested)]
  pub chunking: Chunking,
  #[config(nested)]
  pub database: Database,
  #[config(nested)]
  pub index: IndexPaths,
  #[config(nested)]
  pub graph: GraphPaths,
  #[config(nested)]
  pub search: SearchOptions,
}

#[derive(confique::Config, Debug, Clone)]
#[config(layer_attr(derive(clap::Args)))]
pub struct Embedding {
  #[config(env = "EMBEDDER_BASE_URL")]
  #[config(layer_attr(arg(long = "embedder-base-url")))]
  pub base_url: String,
  #[config(env = "EMBEDDER_API_KEY")]
  #[config(layer_attr(arg(long = "embedder-api-key")))]
  pub api_key: Option<SecretString>,
  #[config(env = "EMBEDDER_MODEL")]
  #[config(layer_attr(arg(long = "embedder-model")))]
  pub model: String,
  #[config(default = "openai", env = "EMBEDDER_DIALECT")]
  #[config(layer_attr(arg(long = "embedder-dialect")))]
  pub dialect: String,
  #[config(default = 10, env = "EMBEDDER_TIMEOUT_SECONDS")]
  #[config(layer_attr(arg(long = "embedder-timeout-seconds")))]
  pub timeout_seconds: u64,
  #[config(env = "EMBEDDING_DIM")]
  #[config(layer_attr(arg(long = "embedding-dim")))]
  pub embedding_dim: usize,
  #[config(default = 8192, env = "EMBEDDER_CONTEXT_LENGTH")]
  #[config(layer_attr(arg(long = "embedder-context-length")))]
  pub context_length: usize,
  #[config(default = 32, env = "EMBEDDER_MAX_BATCH_SIZE")]
  #[config(layer_attr(arg(long = "embedder-max-batch-size")))]
  pub max_batch_size: usize,
  #[config(default = 0, env = "EMBEDDER_TOKENS_PER_MINUTE")]
  #[config(layer_attr(arg(long = "embedder-tokens-per-minute")))]
  pub tokens_per_minute: u32,
}

#[derive(confique::Config, Debug, Clone)]
#[config(layer_attr(derive(clap::Args)))]
pub struct Chunking {
  #[config(default = 1500, env = "CONTEXT_MAX_CHUNK_SIZE")]
  #[config(layer_attr(arg(long = "max-chunk-size")))]
  pub max_chunk_size: usize,
  #[config(default = 0.2, env = "CONTEXT_CHUNK_OVERLAP")]
  #[config(layer_attr(arg(long = "overlap")))]
  pub overlap: f32,
  #[config(default = "characters", env = "CONTEXT_TOKENIZER")]
  #[config(layer_attr(arg(long = "tokenizer")))]
  pub tokenizer: String,
  #[config(default = 4, env = "CONTEXT_MAX_PARALLEL")]
  #[config(layer_attr(arg(long = "max-parallel")))]
  pub max_parallel: usize,
  #[config(default = 5 * 1024 * 1024, env = "CONTEXT_MAX_FILE_SIZE")]
  #[config(layer_attr(arg(long = "max-file-size")))]
  pub max_file_size: u64,
  #[config(default = 4, env = "CONTEXT_LARGE_FILE_THREADS")]
  #[config(layer_attr(arg(long = "large-file-threads")))]
  pub large_file_threads: usize,
}

#[derive(confique::Config, Debug, Clone)]
#[config(layer_attr(derive(clap::Args)))]
pub struct Database {
  #[config(default = "context.duckdb", env = "CONTEXT_DB_PATH")]
  #[config(layer_attr(arg(long = "db")))]
  pub path: PathBuf,
  #[config(env = "EMBEDDING_DIM")]
  #[config(layer_attr(arg(long = "embedding-dim")))]
  pub embedding_dim: usize,
}

#[derive(confique::Config, Debug, Clone)]
#[config(layer_attr(derive(clap::Args)))]
pub struct IndexPaths {
  #[config(default = ".", env = "CONTEXT_REPO_PATH")]
  #[config(layer_attr(arg(long = "repo")))]
  pub repo: PathBuf,
}

#[derive(confique::Config, Debug, Clone)]
#[config(layer_attr(derive(clap::Args)))]
pub struct GraphPaths {
  #[config(default = ".", env = "CONTEXT_REPO_PATH")]
  #[config(layer_attr(arg(long = "repo")))]
  pub repo: PathBuf,
}

#[derive(confique::Config, Debug, Clone)]
#[config(layer_attr(derive(clap::Args)))]
pub struct SearchOptions {
  #[config(default = 10, env = "CONTEXT_SEARCH_LIMIT")]
  #[config(layer_attr(arg(long = "limit")))]
  pub limit: usize,
}

#[derive(Parser)]
#[command(name = "context", version, about = "Codebase indexer and semantic search")]
pub struct Cli {
  /// Optional path to a config file to load in addition to the standard locations.
  #[arg(long = "config-file")]
  pub config_file: Option<PathBuf>,

  #[command(subcommand)]
  pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
  Index(IndexCli),
  Graph(GraphCli),
  Search(SearchCli),
}

#[derive(Args)]
pub struct IndexCli {
  #[command(flatten)]
  pub embedding: <Embedding as confique::Config>::Layer,
  #[command(flatten)]
  pub chunking: <Chunking as confique::Config>::Layer,
  #[command(flatten)]
  pub database: <Database as confique::Config>::Layer,
  #[command(flatten)]
  pub index: <IndexPaths as confique::Config>::Layer,
}

#[derive(Args)]
pub struct GraphCli {
  #[command(flatten)]
  pub chunking: <Chunking as confique::Config>::Layer,
  #[command(flatten)]
  pub database: <Database as confique::Config>::Layer,
  #[command(flatten)]
  pub graph: <GraphPaths as confique::Config>::Layer,
}

#[derive(Args)]
pub struct SearchCli {
  #[command(flatten)]
  pub embedding: <Embedding as confique::Config>::Layer,
  #[command(flatten)]
  pub database: <Database as confique::Config>::Layer,
  #[command(flatten)]
  pub search: <SearchOptions as confique::Config>::Layer,
  #[arg(value_name = "QUERY")]
  pub query: String,
}

pub fn load_config(cli: &Cli) -> eyre::Result<AppConfig> {
  let mut cli_layer = <AppConfig as confique::Config>::Layer::empty();
  match &cli.command {
    Command::Index(cmd) => {
      cli_layer.embedding = cmd.embedding.clone();
      cli_layer.chunking = cmd.chunking.clone();
      cli_layer.database = cmd.database.clone();
      cli_layer.index = cmd.index.clone();
    }
    Command::Graph(cmd) => {
      cli_layer.chunking = cmd.chunking.clone();
      cli_layer.database = cmd.database.clone();
      cli_layer.graph = cmd.graph.clone();
    }
    Command::Search(cmd) => {
      cli_layer.embedding = cmd.embedding.clone();
      cli_layer.database = cmd.database.clone();
      cli_layer.search = cmd.search.clone();
    }
  }

  let mut builder = AppConfig::builder().preloaded(cli_layer).env();
  if let Some(path) = &cli.config_file {
    builder = builder.file(path);
  }

  // Optional local config relative to cwd.
  if let Ok(cwd) = std::env::current_dir() {
    let local_root = cwd.join(".config");
    let local_root_config = local_root.join("context.toml");
    if local_root_config.exists() {
      builder = builder.file(local_root_config);
    }
    let local_root_secrets = local_root.join("context.secrets.toml");
    if local_root_secrets.exists() {
      builder = builder.file(local_root_secrets);
    }

    let local_dir = local_root.join("context");
    let local_dir_secrets = local_dir.join("secrets.toml");
    if local_dir_secrets.exists() {
      builder = builder.file(local_dir_secrets);
    }
    let local_dir_config = local_dir.join("config.toml");
    if local_dir_config.exists() {
      builder = builder.file(local_dir_config);
    }
  }

  // Optional XDG config (subdirectory only).
  if let Some(dir) = dirs::config_dir() {
    let xdg_dir = dir.join("context");
    let xdg_secrets = xdg_dir.join("secrets.toml");
    if xdg_secrets.exists() {
      builder = builder.file(xdg_secrets);
    }
    let xdg_config = xdg_dir.join("config.toml");
    if xdg_config.exists() {
      builder = builder.file(xdg_config);
    }
  }

  // macOS: also prefer ~/.config/context/* (non-standard but requested).
  #[cfg(target_os = "macos")]
  {
    if let Some(home) = dirs::home_dir() {
      let macos_dir = home.join(".config/context");
      let macos_secrets = macos_dir.join("secrets.toml");
      if macos_secrets.exists() {
        builder = builder.file(macos_secrets);
      }
      let macos_config = macos_dir.join("config.toml");
      if macos_config.exists() {
        builder = builder.file(macos_config);
      }
    }
  }

  builder
    .file("/etc/context/secrets.toml")
    .file("/etc/context/config.toml")
    .load()
    .map_err(|e| eyre::eyre!(e.to_string()))
}

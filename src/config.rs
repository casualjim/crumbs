use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};
use confique::Config as _;
use confique::Layer as _;
use secrecy::SecretString;
use std::fs;

#[derive(confique::Config, Debug, Clone)]
pub struct AppConfig {
    #[config(nested)]
    pub embedding: Embedding,
    #[config(nested)]
    pub chunking: Chunking,
    #[config(nested)]
    pub database: Database,
    #[config(nested)]
    pub paths: Paths,
    #[config(nested)]
    pub search: SearchOptions,
}

#[derive(confique::Config, Debug, Clone)]
#[config(layer_attr(derive(clap::Args, Clone)))]
pub struct Embedding {
    #[config(default = "https://api.deepinfra.com/v1/openai", env = "EMBEDDER_URL")]
    #[config(layer_attr(arg(long = "embedder-url")))]
    pub url: String,
    #[config(env = "EMBEDDER_API_KEY")]
    #[config(layer_attr(arg(long = "embedder-api-key")))]
    pub api_key: Option<SecretString>,
    #[config(default = "Qwen/Qwen3-Embedding-0.6B", env = "EMBEDDER_MODEL")]
    #[config(layer_attr(arg(long = "embedder-model")))]
    pub model: String,
    #[config(default = "deepinfra", env = "EMBEDDER_DIALECT")]
    #[config(layer_attr(arg(long = "embedder-dialect")))]
    pub dialect: String,
    #[config(default = 10, env = "EMBEDDER_TIMEOUT_SECONDS")]
    #[config(layer_attr(arg(long = "embedder-timeout-seconds")))]
    pub timeout_seconds: u64,
    #[config(default = 1024, env = "EMBEDDING_DIM")]
    #[config(layer_attr(arg(long = "embedding-dim")))]
    pub embedding_dim: usize,
    #[config(default = 32_768, env = "EMBEDDER_CONTEXT_LENGTH")]
    #[config(layer_attr(arg(long = "embedder-context-length")))]
    pub context_length: usize,
    #[config(default = 15, env = "EMBEDDER_MAX_BATCH_SIZE")]
    #[config(layer_attr(arg(long = "embedder-max-batch-size")))]
    pub max_batch_size: usize,
    #[config(default = 1_000_000, env = "EMBEDDER_TOKENS_PER_MINUTE")]
    #[config(layer_attr(arg(long = "embedder-tokens-per-minute")))]
    pub tokens_per_minute: u32,
}

#[derive(confique::Config, Debug, Clone)]
#[config(layer_attr(derive(clap::Args, Clone)))]
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
    #[config(default = 5_242_880, env = "CONTEXT_MAX_FILE_SIZE")]
    #[config(layer_attr(arg(long = "max-file-size")))]
    pub max_file_size: u64,
    #[config(default = 4, env = "CONTEXT_LARGE_FILE_THREADS")]
    #[config(layer_attr(arg(long = "large-file-threads")))]
    pub large_file_threads: usize,
}

#[derive(confique::Config, Debug, Clone)]
#[config(layer_attr(derive(clap::Args, Clone)))]
pub struct Database {
    #[config(default = "context.duckdb", env = "CONTEXT_DB_PATH")]
    #[config(layer_attr(arg(long = "db")))]
    pub path: PathBuf,
}

#[derive(confique::Config, Debug, Clone)]
#[config(layer_attr(derive(clap::Args, Clone)))]
pub struct Paths {
    #[config(env = "CONTEXT_REPO_PATH")]
    #[config(layer_attr(arg(long = "repo")))]
    pub repo: Option<PathBuf>,
}

#[derive(confique::Config, Debug, Clone)]
#[config(layer_attr(derive(clap::Args, Clone)))]
pub struct SearchOptions {
    #[config(default = 10, env = "CONTEXT_SEARCH_LIMIT")]
    #[config(layer_attr(arg(long = "limit")))]
    pub limit: usize,
    #[config(default = 0.6, env = "CONTEXT_HYBRID_WEIGHT")]
    #[config(layer_attr(arg(long = "hybrid-weight")))]
    pub hybrid_weight: f32,
}

#[derive(Parser)]
#[command(
    name = "context",
    version,
    about = "Codebase indexer and semantic search"
)]
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
    Init(InitCli),
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
    pub paths: <Paths as confique::Config>::Layer,
}

#[derive(Args)]
pub struct GraphCli {
    #[command(flatten)]
    pub chunking: <Chunking as confique::Config>::Layer,
    #[command(flatten)]
    pub database: <Database as confique::Config>::Layer,
    #[command(flatten)]
    pub paths: <Paths as confique::Config>::Layer,
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

#[derive(Args)]
pub struct InitCli {
    #[arg(value_name = "DIRECTORY")]
    pub directory: Option<PathBuf>,
    /// Overwrite existing config files if present.
    #[arg(long = "force")]
    pub force: bool,
}

pub fn load_config(cli: &Cli) -> eyre::Result<AppConfig> {
    let mut cli_layer = <AppConfig as confique::Config>::Layer::empty();
    match &cli.command {
        Command::Index(cmd) => {
            cli_layer.embedding = cmd.embedding.clone();
            cli_layer.chunking = cmd.chunking.clone();
            cli_layer.database = cmd.database.clone();
            cli_layer.paths = cmd.paths.clone();
        }
        Command::Graph(cmd) => {
            cli_layer.chunking = cmd.chunking.clone();
            cli_layer.database = cmd.database.clone();
            cli_layer.paths = cmd.paths.clone();
        }
        Command::Search(cmd) => {
            cli_layer.embedding = cmd.embedding.clone();
            cli_layer.database = cmd.database.clone();
            cli_layer.search = cmd.search.clone();
        }
        Command::Init(_) => {}
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

    let mut config = builder
        .file("/etc/context/secrets.toml")
        .file("/etc/context/config.toml")
        .load()
        .map_err(|e| eyre::eyre!(e.to_string()))?;

    if config.paths.repo.is_none() {
        config.paths.repo = Some(default_repo_path());
    }

    validate_config(&config)?;

    Ok(config)
}

pub struct InitResult {
    pub config_path: PathBuf,
    pub secrets_path: PathBuf,
    pub wrote_config: bool,
    pub wrote_secrets: bool,
}

const DEFAULT_CONFIG_TOML: &str = r#"# context configuration

[embedding]
# Default embedder uses DeepInfra's OpenAI-compatible endpoint.
url = "https://api.deepinfra.com/v1/openai"
model = "Qwen/Qwen3-Embedding-0.6B"
dialect = "deepinfra"
timeout_seconds = 10
embedding_dim = 1024
context_length = 32768
max_batch_size = 15
tokens_per_minute = 1000000

[chunking]
max_chunk_size = 1500
overlap = 0.2
tokenizer = "characters"
max_parallel = 4
max_file_size = 5242880
large_file_threads = 4

[database]
path = "context.duckdb"

[paths]
# repo = "/path/to/repo" # Optional; defaults to the git root of the current directory.

[search]
limit = 10
hybrid_weight = 0.6
"#;

const DEFAULT_SECRETS_TOML: &str = r#"# Secrets for context
# You can also set EMBEDDER_API_KEY in your environment instead.

[embedding]
# api_key = "sk-..."
"#;

pub fn init_config(init: &InitCli) -> eyre::Result<InitResult> {
    let root = match init.directory.as_ref() {
        Some(dir) => dir.join(".config").join("context"),
        None => default_config_root()?,
    };
    fs::create_dir_all(&root)?;

    let config_path = root.join("config.toml");
    let secrets_path = root.join("secrets.toml");

    let wrote_config = write_default_file(&config_path, DEFAULT_CONFIG_TOML, init.force)?;
    let wrote_secrets = write_default_file(&secrets_path, DEFAULT_SECRETS_TOML, init.force)?;

    Ok(InitResult {
        config_path,
        secrets_path,
        wrote_config,
        wrote_secrets,
    })
}

fn write_default_file(path: &Path, contents: &str, force: bool) -> eyre::Result<bool> {
    if path.exists() && !force {
        return Ok(false);
    }
    fs::write(path, contents)?;
    Ok(true)
}

fn default_config_root() -> eyre::Result<PathBuf> {
    let base =
        dirs::config_dir().ok_or_else(|| eyre::eyre!("unable to resolve user config directory"))?;
    Ok(base.join("context"))
}

fn validate_config(config: &AppConfig) -> eyre::Result<()> {
    if config.embedding.embedding_dim == 0 {
        return Err(eyre::eyre!("embedding_dim must be > 0"));
    }
    if config.embedding.context_length == 0 {
        return Err(eyre::eyre!("embedder context_length must be > 0"));
    }
    if config.embedding.max_batch_size == 0 {
        return Err(eyre::eyre!("embedder max_batch_size must be > 0"));
    }
    if config.chunking.max_chunk_size == 0 {
        return Err(eyre::eyre!("max_chunk_size must be > 0"));
    }
    if !(0.0..1.0).contains(&config.chunking.overlap) {
        return Err(eyre::eyre!("overlap must be in [0.0, 1.0)"));
    }
    if config.chunking.max_parallel == 0 {
        return Err(eyre::eyre!("max_parallel must be > 0"));
    }
    if config.chunking.max_file_size == 0 {
        return Err(eyre::eyre!("max_file_size must be > 0"));
    }
    if config.chunking.large_file_threads == 0 {
        return Err(eyre::eyre!("large_file_threads must be > 0"));
    }
    if config.search.limit == 0 {
        return Err(eyre::eyre!("search limit must be > 0"));
    }
    if !(0.0..=1.0).contains(&config.search.hybrid_weight) {
        return Err(eyre::eyre!("hybrid_weight must be in [0.0, 1.0]"));
    }
    Ok(())
}

fn default_repo_path() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    find_git_root(&cwd).unwrap_or(cwd)
}

fn find_git_root(start: &Path) -> Option<PathBuf> {
    let mut current = start.to_path_buf();
    loop {
        if current.join(".git").exists() {
            return Some(current);
        }
        if !current.pop() {
            break;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn validate_config_rejects_invalid_overlap() {
        let config = AppConfig {
            embedding: Embedding {
                url: "http://localhost".to_string(),
                api_key: None,
                model: "model".to_string(),
                dialect: "openai".to_string(),
                timeout_seconds: 10,
                embedding_dim: 2,
                context_length: 8,
                max_batch_size: 4,
                tokens_per_minute: 0,
            },
            chunking: Chunking {
                max_chunk_size: 10,
                overlap: 1.5,
                tokenizer: "characters".to_string(),
                max_parallel: 1,
                max_file_size: 1_024,
                large_file_threads: 1,
            },
            database: Database {
                path: PathBuf::from("context.duckdb"),
            },
            paths: Paths {
                repo: Some(PathBuf::from(".")),
            },
            search: SearchOptions {
                limit: 1,
                hybrid_weight: 0.6,
            },
        };

        assert!(
            validate_config(&config).is_err(),
            "expected invalid overlap to be rejected"
        );
    }

    #[test]
    fn init_config_creates_files() -> eyre::Result<()> {
        let dir = TempDir::new()?;
        let init = InitCli {
            directory: Some(dir.path().to_path_buf()),
            force: false,
        };

        let result = init_config(&init)?;
        assert!(result.config_path.exists(), "expected config.toml to exist");
        assert!(
            result.secrets_path.exists(),
            "expected secrets.toml to exist"
        );
        Ok(())
    }
}

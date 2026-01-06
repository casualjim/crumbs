use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand, ValueEnum};
use confique::Config as _;
use confique::Layer as _;
use secrecy::SecretString;
use serde::Deserialize;
use std::fs;
use toml_edit::{DocumentMut, Item, Table, value};

const DEFAULT_DATABASE_NAME: &str = "context.duckdb";

#[derive(confique::Config, Debug, Clone)]
pub struct AppConfig {
    #[config(nested)]
    pub embedding: Embedding,
    #[config(nested)]
    pub reranker: Reranker,
    #[config(nested)]
    pub chunking: Chunking,
    #[config(nested)]
    pub history: History,
    #[config(default = {})]
    pub projects: BTreeMap<String, Project>,
    #[config(nested)]
    pub search: SearchOptions,
    #[config(nested)]
    pub prompt: Prompting,
}

#[derive(confique::Config, Debug, Clone)]
#[config(layer_attr(derive(clap::Args, Clone)))]
pub struct Embedding {
    #[config(default = "https://api.deepinfra.com/v1/openai", env = "EMBEDDER_URL")]
    #[config(layer_attr(arg(
        id = "embedder_url",
        long = "embedder-url",
        help = "Embedding API base URL"
    )))]
    pub url: String,
    #[config(env = "EMBEDDER_API_KEY")]
    #[config(layer_attr(arg(
        id = "embedder_api_key",
        long = "embedder-api-key",
        help = "Embedding API key (or set in secrets)"
    )))]
    pub api_key: Option<SecretString>,
    #[config(default = "Qwen/Qwen3-Embedding-0.6B", env = "EMBEDDER_MODEL")]
    #[config(layer_attr(arg(
        id = "embedder_model",
        long = "embedder-model",
        help = "Embedding model name"
    )))]
    pub model: String,
    #[config(
        default = "hf:Qwen/Qwen3-Embedding-0.6B",
        env = "EMBEDDER_TOKENIZER"
    )]
    #[config(layer_attr(arg(
        id = "embedder_tokenizer",
        long = "embedder-tokenizer",
        help = "Tokenizer for chunking (characters|tiktoken:<name>|hf:<model>)"
    )))]
    pub tokenizer: String,
    #[config(default = "deepinfra", env = "EMBEDDER_DIALECT")]
    #[config(layer_attr(arg(
        id = "embedder_dialect",
        long = "embedder-dialect",
        help = "Provider dialect: openai|deepinfra"
    )))]
    pub dialect: String,
    #[config(default = 10, env = "EMBEDDER_TIMEOUT_SECONDS")]
    #[config(layer_attr(arg(
        id = "embedder_timeout_seconds",
        long = "embedder-timeout-seconds",
        help = "Request timeout in seconds"
    )))]
    pub timeout_seconds: u64,
    #[config(default = 1024, env = "EMBEDDING_DIM")]
    #[config(layer_attr(arg(
        id = "embedder_embedding_dim",
        long = "embedding-dim",
        help = "Embedding vector dimension"
    )))]
    pub embedding_dim: usize,
    #[config(default = 32_768, env = "EMBEDDER_CONTEXT_LENGTH")]
    #[config(layer_attr(arg(
        id = "embedder_context_length",
        long = "embedder-context-length",
        help = "Max tokens per embedding request"
    )))]
    pub context_length: usize,
    #[config(default = 256, env = "EMBEDDER_STREAM_BATCH_SIZE")]
    #[config(layer_attr(arg(
        id = "embedder_stream_batch_size",
        long = "embedder-stream-batch-size",
        help = "Max chunks per stream batch before token-aware splitting"
    )))]
    pub stream_batch_size: usize,
    #[config(default = 15, env = "EMBEDDER_MAX_BATCH_SIZE")]
    #[config(layer_attr(arg(
        id = "embedder_max_batch_size",
        long = "embedder-max-batch-size",
        help = "Max inputs per embedding batch"
    )))]
    pub max_batch_size: usize,
    #[config(default = 1000, env = "EMBEDDER_REQUESTS_PER_MINUTE")]
    #[config(layer_attr(arg(
        id = "embedder_requests_per_minute",
        long = "embedder-requests-per-minute",
        help = "Rate limit for embedding requests per minute"
    )))]
    pub requests_per_minute: usize,
    #[config(default = 300, env = "EMBEDDER_MAX_CONCURRENT_REQUESTS")]
    #[config(layer_attr(arg(
        id = "embedder_max_concurrent_requests",
        long = "embedder-max-concurrent-requests",
        help = "Max concurrent embedding requests"
    )))]
    pub max_concurrent_requests: usize,
    #[config(default = 1_000_000, env = "EMBEDDER_TOKENS_PER_MINUTE")]
    #[config(layer_attr(arg(
        id = "embedder_tokens_per_minute",
        long = "embedder-tokens-per-minute",
        help = "Rate limit for embedding tokens per minute"
    )))]
    pub tokens_per_minute: u32,
}

#[derive(confique::Config, Debug, Clone)]
#[config(layer_attr(derive(clap::Args, Clone)))]
pub struct Reranker {
    #[config(default = "https://api.deepinfra.com/v1", env = "RERANKER_URL")]
    #[config(layer_attr(arg(
        id = "reranker_url",
        long = "reranker-url",
        help = "Reranker API base URL"
    )))]
    pub url: String,
    #[config(env = "RERANKER_API_KEY")]
    #[config(layer_attr(arg(
        id = "reranker_api_key",
        long = "reranker-api-key",
        help = "Reranker API key (or set in secrets)"
    )))]
    pub api_key: Option<SecretString>,
    #[config(default = "Qwen/Qwen3-Reranker-0.6B", env = "RERANKER_MODEL")]
    #[config(layer_attr(arg(
        id = "reranker_model",
        long = "reranker-model",
        help = "Reranker model name"
    )))]
    pub model: String,
    #[config(default = "deepinfra", env = "RERANKER_DIALECT")]
    #[config(layer_attr(arg(
        id = "reranker_dialect",
        long = "reranker-dialect",
        help = "Provider dialect: openai|deepinfra"
    )))]
    pub dialect: String,
    #[config(default = 10, env = "RERANKER_TIMEOUT_SECONDS")]
    #[config(layer_attr(arg(
        id = "reranker_timeout_seconds",
        long = "reranker-timeout-seconds",
        help = "Reranker request timeout in seconds"
    )))]
    pub timeout_seconds: u64,
    #[config(env = "RERANKER_INSTRUCTION")]
    #[config(layer_attr(arg(
        id = "reranker_instruction",
        long = "reranker-instruction",
        help = "Optional reranker instruction/prompt"
    )))]
    pub instruction: Option<String>,
}

#[derive(confique::Config, Debug, Clone)]
#[config(layer_attr(derive(clap::Args, Clone)))]
pub struct Chunking {
    #[config(default = 1500, env = "CONTEXT_MAX_CHUNK_SIZE")]
    #[config(layer_attr(arg(long = "max-chunk-size", help = "Max characters/tokens per chunk")))]
    pub max_chunk_size: usize,
    #[config(default = 0.2, env = "CONTEXT_CHUNK_OVERLAP")]
    #[config(layer_attr(arg(long = "overlap", help = "Chunk overlap ratio (0.0-1.0)")))]
    pub overlap: f32,
    #[config(default = 4, env = "CONTEXT_MAX_PARALLEL")]
    #[config(layer_attr(arg(long = "max-parallel", help = "Max files to chunk in parallel")))]
    pub max_parallel: usize,
    #[config(default = 5_242_880, env = "CONTEXT_MAX_FILE_SIZE")]
    #[config(layer_attr(arg(long = "max-file-size", help = "Max file size (bytes) to index")))]
    pub max_file_size: u64,
    #[config(default = 4, env = "CONTEXT_LARGE_FILE_THREADS")]
    #[config(layer_attr(arg(
        long = "large-file-threads",
        help = "Threads to use for large file chunking"
    )))]
    pub large_file_threads: usize,
}

#[derive(confique::Config, Debug, Clone, Deserialize)]
pub struct Project {
    pub repo: PathBuf,
    pub data_dir: Option<PathBuf>,
    pub database: Option<PathBuf>,
}

#[derive(confique::Config, Debug, Clone)]
#[config(layer_attr(derive(clap::Args, Clone)))]
pub struct History {
    #[config(default = 10240, env = "CONTEXT_HISTORY_DEPTH")]
    #[config(layer_attr(arg(long = "history-depth", help = "Max commit history depth")))]
    pub depth: u32,
    #[config(default = 1.0, env = "CONTEXT_HISTORY_COMMIT_SIZE_LIMIT_RATIO")]
    #[config(layer_attr(arg(
        long = "history-commit-size-limit-ratio",
        help = "Ignore commits touching too many files (ratio)"
    )))]
    pub commit_size_limit_ratio: f32,
    #[config(default = false, env = "CONTEXT_HISTORY_MULTI_PARENTS")]
    #[config(layer_attr(arg(long = "history-multi-parents", help = "Include merge commits")))]
    pub multi_parents: bool,
    #[config(default = "(#\\d+)", env = "CONTEXT_HISTORY_ISSUE_REGEX")]
    #[config(layer_attr(arg(
        long = "history-issue-regex",
        help = "Issue/PR regex for commit messages"
    )))]
    pub issue_regex: String,
    #[config(env = "CONTEXT_HISTORY_COMMIT_EXCLUDE_REGEX")]
    #[config(layer_attr(arg(
        long = "history-commit-exclude-regex",
        help = "Exclude commits matching regex"
    )))]
    pub commit_exclude_regex: Option<String>,
    #[config(env = "CONTEXT_HISTORY_AUTHOR_EXCLUDE_REGEX")]
    #[config(layer_attr(arg(
        long = "history-author-exclude-regex",
        help = "Exclude authors matching regex"
    )))]
    pub author_exclude_regex: Option<String>,
    #[config(default = "", env = "CONTEXT_HISTORY_PATHS")]
    #[config(layer_attr(arg(
        long = "history-pathspec",
        value_delimiter = ',',
        help = "Comma-separated pathspecs to include"
    )))]
    pub path_specs: String,
}

#[derive(confique::Config, Debug, Clone)]
#[config(layer_attr(derive(clap::Args, Clone)))]
pub struct SearchOptions {
    #[config(default = 10, env = "CONTEXT_SEARCH_LIMIT")]
    #[config(layer_attr(arg(long = "limit", help = "Max results to return")))]
    pub limit: usize,
    #[config(default = 0.25, env = "CONTEXT_SEARCH_MIN_SCORE")]
    #[config(layer_attr(arg(
        long = "min-score",
        help = "Minimum score to include in results (0.0-1.0)"
    )))]
    pub min_score: f64,
    #[config(default = 0.6, env = "CONTEXT_HYBRID_WEIGHT")]
    #[config(layer_attr(arg(
        long = "hybrid-weight",
        help = "Weight for hybrid scoring (0=FTS, 1=vector)"
    )))]
    pub hybrid_weight: f32,
    #[config(default = [], env = "CONTEXT_SEARCH_PATH_PREFIX")]
    #[config(layer_attr(arg(
        long = "path-prefix",
        value_delimiter = ',',
        help = "Restrict results to file path prefixes (comma-separated)"
    )))]
    pub path_prefixes: Vec<String>,
    #[config(default = [], env = "CONTEXT_SEARCH_FILE_EXT")]
    #[config(layer_attr(arg(
        long = "file-ext",
        value_delimiter = ',',
        help = "Restrict results to file extensions (comma-separated)"
    )))]
    pub file_exts: Vec<String>,
    #[config(default = false, env = "CONTEXT_SEARCH_DECOMPOSE")]
    #[config(layer_attr(arg(
        long = "decompose",
        help = "Split multi-part queries on conjunctions"
    )))]
    pub decompose: bool,
    #[config(default = true, env = "CONTEXT_SEARCH_RERANK")]
    #[config(layer_attr(arg(
        long = "rerank",
        help = "Apply model-based reranking"
    )))]
    pub rerank: bool,
}

#[derive(confique::Config, Debug, Clone)]
#[config(layer_attr(derive(clap::Args, Clone)))]
pub struct Prompting {
    #[config(default = "", env = "CONTEXT_PROMPT_TOKENIZER")]
    #[config(layer_attr(arg(
        id = "prompt_tokenizer",
        long = "prompt-tokenizer",
        help = "Tokenizer for prompt budgeting (defaults to embedding tokenizer)"
    )))]
    pub tokenizer: String,
}

#[derive(Parser)]
#[command(
    name = "context",
    version,
    about = "Codebase indexer and context retrieval for LLM prompts",
    long_about = "Index a Git repo into a local database (chunks, embeddings, graphs, git history) \
and retrieve high-signal code context for LLM prompts."
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
    #[command(about = "Create default config and secrets files")]
    Init(InitCli),
    #[command(about = "Build or refresh the repo index")]
    Index(IndexCli),
    #[command(about = "Search the index with a natural language query")]
    Search(SearchCli),
    #[command(about = "Assemble prompt-ready context for a task")]
    Prompt(PromptCli),
    #[command(about = "Manage configuration files")]
    Config(ConfigCli),
}

#[derive(Args, Clone)]
pub struct ProjectCli {
    #[arg(long = "project", help = "Named project entry to use from config")]
    pub project: Option<String>,
}

#[derive(Args)]
pub struct IndexCli {
    #[command(flatten)]
    pub embedding: <Embedding as confique::Config>::Layer,
    #[command(flatten)]
    pub chunking: <Chunking as confique::Config>::Layer,
    #[command(flatten)]
    pub history: <History as confique::Config>::Layer,
    #[command(flatten)]
    pub project: ProjectCli,
}

#[derive(Args)]
pub struct SearchCli {
    #[command(flatten)]
    pub embedding: <Embedding as confique::Config>::Layer,
    #[command(flatten)]
    pub reranker: <Reranker as confique::Config>::Layer,
    #[command(flatten)]
    pub project: ProjectCli,
    #[command(flatten)]
    pub search: <SearchOptions as confique::Config>::Layer,
    #[arg(value_name = "QUERY", help = "Natural language search query")]
    pub query: String,
}

#[derive(Args)]
pub struct InitCli {
    #[arg(
        long = "local",
        help = "Write config under <repo>/.config/context instead of the user config dir"
    )]
    pub local: bool,
    #[arg(long = "force", help = "Overwrite existing config files")]
    pub force: bool,
}

#[derive(Args)]
pub struct PromptCli {
    #[command(flatten)]
    pub embedding: <Embedding as confique::Config>::Layer,
    #[command(flatten)]
    pub reranker: <Reranker as confique::Config>::Layer,
    #[command(flatten)]
    pub project: ProjectCli,
    #[command(flatten)]
    pub search: <SearchOptions as confique::Config>::Layer,
    #[command(flatten)]
    pub prompt: <Prompting as confique::Config>::Layer,
    #[arg(
        long = "max-tokens",
        default_value_t = 0,
        help = "Max input tokens for assembled context (0 uses embedder context length)"
    )]
    pub max_tokens: usize,
    #[arg(
        long = "reserved-output-tokens",
        default_value_t = 0,
        help = "Tokens reserved for model output"
    )]
    pub reserved_output_tokens: usize,
    #[arg(
        long = "format",
        value_enum,
        default_value = "xml",
        help = "Output format: xml|markdown"
    )]
    pub format: PromptFormat,
    #[arg(value_name = "TASK", help = "Task or question to build context for")]
    pub task: String,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum PromptFormat {
    #[value(alias = "md")]
    Markdown,
    Xml,
}

#[derive(Args)]
pub struct ConfigCli {
    #[command(subcommand)]
    pub command: ConfigCommand,
}

#[derive(Subcommand)]
pub enum ConfigCommand {
    #[command(about = "Show the resolved config")]
    Show,
    #[command(about = "Set a config value")]
    Set(ConfigSetCli),
    #[command(about = "Validate config and embedding setup")]
    Doctor,
}

#[derive(Args)]
pub struct ConfigSetCli {
    #[arg(value_name = "KEY", help = "Config key (e.g. embedding.model)")]
    pub key: String,
    #[arg(value_name = "VALUE", help = "Value to set")]
    pub value: String,
    #[arg(
        long = "local",
        help = "Write to <repo>/.config/context/config.toml instead of user config"
    )]
    pub local: bool,
}

pub fn load_config(cli: &Cli) -> eyre::Result<AppConfig> {
    let mut cli_layer = <AppConfig as confique::Config>::Layer::empty();
    match &cli.command {
        Command::Init(_) => {}
        Command::Index(cmd) => {
            cli_layer.embedding = cmd.embedding.clone();
            cli_layer.chunking = cmd.chunking.clone();
            cli_layer.history = cmd.history.clone();
        }
        Command::Search(cmd) => {
            cli_layer.embedding = cmd.embedding.clone();
            cli_layer.reranker = cmd.reranker.clone();
            cli_layer.search = cmd.search.clone();
        }
        Command::Prompt(cmd) => {
            cli_layer.embedding = cmd.embedding.clone();
            cli_layer.reranker = cmd.reranker.clone();
            cli_layer.search = cmd.search.clone();
            cli_layer.prompt = cmd.prompt.clone();
        }
        Command::Config(_) => {}
    }

    let mut builder = AppConfig::builder().preloaded(cli_layer).env();
    if let Some(path) = &cli.config_file {
        builder = builder.file(path);
    }

    // Optional local config relative to repo root or cwd.
    if let Ok(cwd) = std::env::current_dir() {
        let base = find_git_root(&cwd).unwrap_or(cwd);
        let local_root = base.join(".config");
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

    let config = builder
        .file("/etc/context/secrets.toml")
        .file("/etc/context/config.toml")
        .load()
        .map_err(|e| eyre::eyre!(e.to_string()))?;

    validate_config(&config)?;

    Ok(config)
}

pub struct ResolvedProject {
    pub repo_path: PathBuf,
    pub database_path: PathBuf,
}

pub fn resolve_project(
    config: &AppConfig,
    override_name: Option<&str>,
) -> eyre::Result<ResolvedProject> {
    let cwd = std::env::current_dir()?;
    resolve_project_for_cwd(config, &cwd, override_name)
}

fn resolve_project_for_cwd(
    config: &AppConfig,
    cwd: &Path,
    override_name: Option<&str>,
) -> eyre::Result<ResolvedProject> {
    if let Some(name) = override_name {
        let project = config.projects.get(name).ok_or_else(|| {
            eyre::eyre!("unknown project '{name}'; define it under [projects.{name}]")
        })?;
        return build_resolved_project(name, project, cwd);
    }

    let repo_root = find_git_root(cwd)
        .ok_or_else(|| eyre::eyre!("no git repository found in current directory"))?;

    if let Some((name, project)) = find_project_for_repo(&config.projects, &repo_root) {
        return build_resolved_project(name, project, cwd);
    }

    let implicit_name = repo_root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("repo")
        .to_string();
    let implicit_project = Project {
        repo: repo_root.clone(),
        data_dir: None,
        database: None,
    };
    build_resolved_project(&implicit_name, &implicit_project, cwd)
}

fn find_project_for_repo<'a>(
    projects: &'a BTreeMap<String, Project>,
    repo_root: &Path,
) -> Option<(&'a str, &'a Project)> {
    let repo_root = canonical_path(repo_root);
    projects
        .iter()
        .find(|(_, project)| canonical_path(&project.repo) == repo_root)
        .map(|(name, project)| (name.as_str(), project))
}

fn build_resolved_project(
    name: &str,
    project: &Project,
    cwd: &Path,
) -> eyre::Result<ResolvedProject> {
    let repo_path = resolve_path(&project.repo, cwd);
    let repo_path = canonical_or_existing(repo_path)?;

    let data_dir = if let Some(data_dir) = &project.data_dir {
        let resolved = resolve_path(data_dir, &repo_path);
        canonical_or_existing(resolved)?
    } else if prefer_os_data_root(&repo_path) {
        let base = dirs::data_dir()
            .ok_or_else(|| eyre::eyre!("unable to resolve XDG data directory"))?
            .join("context")
            .join(name);
        canonical_or_existing(base)?
    } else {
        repo_path.join(".config").join("context")
    };

    fs::create_dir_all(&data_dir)?;

    let database = project
        .database
        .as_ref()
        .cloned()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DATABASE_NAME));

    let database_path = if database.is_absolute() {
        database
    } else {
        data_dir.join(database)
    };

    ensure_repo_data_gitignore(&repo_path, &data_dir, &database_path)?;

    Ok(ResolvedProject {
        repo_path,
        database_path,
    })
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
tokenizer = "hf:Qwen/Qwen3-Embedding-0.6B"
dialect = "deepinfra"
timeout_seconds = 10
embedding_dim = 1024
context_length = 32768
stream_batch_size = 256
max_batch_size = 15
requests_per_minute = 1000
max_concurrent_requests = 300
tokens_per_minute = 1000000

[reranker]
url = "https://api.deepinfra.com/v1"
model = "Qwen/Qwen3-Reranker-0.6B"
dialect = "deepinfra"
timeout_seconds = 10
# instruction = ""

[chunking]
max_chunk_size = 1500
overlap = 0.2
max_parallel = 4
max_file_size = 5242880
large_file_threads = 4

[history]
depth = 10240
commit_size_limit_ratio = 1.0
multi_parents = false
issue_regex = "(#\\d+)"
# commit_exclude_regex = ""
# author_exclude_regex = ""
# path_specs = ""

# Project definitions live under [projects.<name>].
# Example:
# [projects.example]
# repo = "/path/to/repo"
# # data_dir = "/path/to/data"
# # database = "context.duckdb"

[search]
limit = 10
min_score = 0.25
hybrid_weight = 0.6
# path_prefixes = ["src/"]
# file_exts = ["rs"]
# decompose = false
rerank = true

[prompt]
# tokenizer = "" # defaults to embedding tokenizer
"#;

const DEFAULT_SECRETS_TOML: &str = r#"# Secrets for context
# You can also set EMBEDDER_API_KEY in your environment instead.

[embedding]
# api_key = "sk-..."

[reranker]
# api_key = "sk-..."
"#;

pub fn init_config(init: &InitCli) -> eyre::Result<InitResult> {
    let root = if init.local {
        let cwd = std::env::current_dir()?;
        let repo_root = find_git_root(&cwd).unwrap_or(cwd);
        repo_root.join(".config").join("context")
    } else {
        default_config_root()?
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
pub struct ConfigSetResult {
    pub config_path: PathBuf,
    pub key: String,
    pub created: bool,
}

pub fn set_config_value(
    cli: &ConfigSetCli,
    config_file: Option<&Path>,
) -> eyre::Result<ConfigSetResult> {
    let config_path = if cli.local {
        local_config_root()?.join("config.toml")
    } else {
        resolve_config_write_path(config_file)?
    };
    let existed = config_path.exists();
    let mut document = if existed {
        let contents = fs::read_to_string(&config_path)?;
        contents
            .parse::<DocumentMut>()
            .map_err(|err| eyre::eyre!("failed to parse {}: {}", config_path.display(), err))?
    } else {
        DocumentMut::new()
    };

    let value_item = parse_toml_value(&cli.value);
    set_document_value(&mut document, &cli.key, value_item)?;

    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config_path, document.to_string())?;

    Ok(ConfigSetResult {
        config_path,
        key: cli.key.clone(),
        created: !existed,
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

fn local_config_root() -> eyre::Result<PathBuf> {
    let cwd = std::env::current_dir()?;
    let repo_root = find_git_root(&cwd).unwrap_or(cwd);
    Ok(repo_root.join(".config").join("context"))
}

fn parse_toml_value(raw: &str) -> Item {
    if raw.eq_ignore_ascii_case("true") {
        return value(true);
    }
    if raw.eq_ignore_ascii_case("false") {
        return value(false);
    }
    if let Ok(parsed) = raw.parse::<i64>() {
        return value(parsed);
    }
    if let Ok(parsed) = raw.parse::<f64>() {
        return value(parsed);
    }
    value(raw)
}

fn set_document_value(document: &mut DocumentMut, key: &str, item: Item) -> eyre::Result<()> {
    let mut parts = key.split('.').collect::<Vec<_>>();
    let leaf = parts
        .pop()
        .ok_or_else(|| eyre::eyre!("config key cannot be empty"))?;

    let mut table = document.as_table_mut();
    for part in parts {
        let entry = table.entry(part).or_insert(Item::Table(Table::new()));
        table = entry
            .as_table_mut()
            .ok_or_else(|| eyre::eyre!("config path '{}' is not a table", part))?;
    }

    table[leaf] = item;
    Ok(())
}

fn resolve_config_write_path(config_file: Option<&Path>) -> eyre::Result<PathBuf> {
    if let Some(path) = config_file {
        return Ok(path.to_path_buf());
    }
    if let Some(path) = existing_user_config_path() {
        return Ok(path);
    }
    default_config_root().map(|root| root.join("config.toml"))
}

fn existing_user_config_path() -> Option<PathBuf> {
    let standard = dirs::config_dir().map(|dir| dir.join("context").join("config.toml"));
    if let Some(path) = standard
        && path.exists()
    {
        return Some(path);
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(home) = dirs::home_dir() {
            let path = home.join(".config").join("context").join("config.toml");
            if path.exists() {
                return Some(path);
            }
        }
    }

    None
}

fn validate_config(config: &AppConfig) -> eyre::Result<()> {
    if config.embedding.embedding_dim == 0 {
        return Err(eyre::eyre!("embedding_dim must be > 0"));
    }
    if config.embedding.tokenizer.trim().is_empty() {
        return Err(eyre::eyre!("embedding tokenizer must be set"));
    }
    if config.embedding.context_length == 0 {
        return Err(eyre::eyre!("embedder context_length must be > 0"));
    }
    if config.embedding.stream_batch_size == 0 {
        return Err(eyre::eyre!("embedder stream_batch_size must be > 0"));
    }
    if config.embedding.max_batch_size == 0 {
        return Err(eyre::eyre!("embedder max_batch_size must be > 0"));
    }
    if config.embedding.requests_per_minute == 0 {
        return Err(eyre::eyre!("embedder requests_per_minute must be > 0"));
    }
    if config.embedding.max_concurrent_requests == 0 {
        return Err(eyre::eyre!(
            "embedder max_concurrent_requests must be > 0"
        ));
    }
    if config.embedding.tokens_per_minute == 0 {
        return Err(eyre::eyre!("embedder tokens_per_minute must be > 0"));
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
    if !(0.0..=1.0).contains(&config.search.min_score) {
        return Err(eyre::eyre!("search min_score must be in [0.0, 1.0]"));
    }
    if !(0.0..=1.0).contains(&config.search.hybrid_weight) {
        return Err(eyre::eyre!("hybrid_weight must be in [0.0, 1.0]"));
    }
    if config.history.depth == 0 {
        return Err(eyre::eyre!("history depth must be > 0"));
    }
    if !(0.0..=1.0).contains(&config.history.commit_size_limit_ratio) {
        return Err(eyre::eyre!(
            "history commit_size_limit_ratio must be in [0.0, 1.0]"
        ));
    }
    Ok(())
}

fn prefer_os_data_root(repo_root: &Path) -> bool {
    if local_config_present(repo_root) {
        return false;
    }

    if etc_config_present() {
        return true;
    }

    user_config_present() || macos_alt_config_present()
}

fn local_config_present(repo_root: &Path) -> bool {
    let local_root = repo_root.join(".config");
    let candidates = [
        local_root.join("context.toml"),
        local_root.join("context.secrets.toml"),
        local_root.join("context").join("config.toml"),
        local_root.join("context").join("secrets.toml"),
    ];
    candidates.iter().any(|path| path.exists())
}

fn user_config_present() -> bool {
    if let Some(dir) = dirs::config_dir() {
        let base = dir.join("context");
        return base.join("config.toml").exists() || base.join("secrets.toml").exists();
    }
    false
}

fn macos_alt_config_present() -> bool {
    #[cfg(target_os = "macos")]
    {
        if let Some(home) = dirs::home_dir() {
            let base = home.join(".config").join("context");
            return base.join("config.toml").exists() || base.join("secrets.toml").exists();
        }
    }
    false
}

fn etc_config_present() -> bool {
    Path::new("/etc/context/config.toml").exists()
        || Path::new("/etc/context/secrets.toml").exists()
}

fn ensure_repo_data_gitignore(
    repo_root: &Path,
    data_root: &Path,
    database_path: &Path,
) -> eyre::Result<()> {
    if !repo_root.join(".git").exists() {
        return Ok(());
    }
    if !data_root.starts_with(repo_root) {
        return Ok(());
    }

    let ignore_path = data_root.join(".gitignore");
    let mut lines = Vec::new();
    if let Some(file_name) = database_path.file_name().and_then(|f| f.to_str()) {
        lines.push(file_name.to_string());
        lines.push(format!("{file_name}.wal"));
        lines.push(format!("{file_name}.shm"));
        lines.push(format!("{file_name}-wal"));
        lines.push(format!("{file_name}-shm"));
    }
    lines.push("secrets.toml".to_string());
    lines.push("!.gitignore".to_string());

    if ignore_path.exists() {
        let existing = fs::read_to_string(&ignore_path).unwrap_or_default();
        let mut additions = Vec::new();
        for line in &lines {
            if !existing
                .lines()
                .any(|existing_line| existing_line.trim() == line)
            {
                additions.push(line.as_str());
            }
        }
        if additions.is_empty() {
            return Ok(());
        }
        let mut updated = existing;
        if !updated.ends_with('\n') {
            updated.push('\n');
        }
        updated.push_str(&additions.join("\n"));
        updated.push('\n');
        fs::write(ignore_path, updated)?;
    } else {
        let mut contents = String::from("# context data\n");
        contents.push_str(&lines.join("\n"));
        contents.push('\n');
        fs::write(ignore_path, contents)?;
    }
    Ok(())
}

fn resolve_path(path: &Path, base: &Path) -> PathBuf {
    if let Some(expanded) = expand_tilde(path) {
        return expanded;
    }
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn expand_tilde(path: &Path) -> Option<PathBuf> {
    let raw = path.to_string_lossy();
    if !raw.starts_with("~/") {
        return None;
    }
    let home = dirs::home_dir()?;
    Some(home.join(raw.trim_start_matches("~/")))
}

fn canonical_or_existing(path: PathBuf) -> eyre::Result<PathBuf> {
    if path.exists() {
        Ok(path.canonicalize()?)
    } else {
        Ok(path)
    }
}

fn canonical_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
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
        let mut projects = BTreeMap::new();
        projects.insert(
            "example".to_string(),
            Project {
                repo: PathBuf::from("."),
                data_dir: None,
                database: None,
            },
        );
        let config = AppConfig {
            embedding: Embedding {
                url: "http://localhost".to_string(),
                api_key: None,
                model: "model".to_string(),
                tokenizer: "hf:Qwen/Qwen3-Embedding-0.6B".to_string(),
                dialect: "openai".to_string(),
                timeout_seconds: 10,
                embedding_dim: 2,
                context_length: 8,
                stream_batch_size: 4,
                max_batch_size: 4,
                requests_per_minute: 1000,
                max_concurrent_requests: 300,
                tokens_per_minute: 1,
            },
            reranker: Reranker {
                url: "http://localhost".to_string(),
                api_key: None,
                model: "model".to_string(),
                dialect: "openai".to_string(),
                timeout_seconds: 10,
                instruction: None,
            },
            chunking: Chunking {
                max_chunk_size: 10,
                overlap: 1.5,
                max_parallel: 1,
                max_file_size: 1_024,
                large_file_threads: 1,
            },
            history: History {
                depth: 1,
                commit_size_limit_ratio: 1.0,
                multi_parents: false,
                issue_regex: "(#\\d+)".to_string(),
                commit_exclude_regex: None,
                author_exclude_regex: None,
                path_specs: String::new(),
            },
            projects,
            search: SearchOptions {
                limit: 1,
                min_score: 0.25,
                hybrid_weight: 0.6,
                path_prefixes: Vec::new(),
                file_exts: Vec::new(),
                decompose: false,
                rerank: false,
            },
            prompt: Prompting {
                tokenizer: String::new(),
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
        let original = std::env::current_dir()?;
        std::env::set_current_dir(dir.path())?;
        let init = InitCli {
            local: true,
            force: false,
        };

        let result = init_config(&init)?;
        std::env::set_current_dir(original)?;
        assert!(result.config_path.exists(), "expected config.toml to exist");
        assert!(
            result.secrets_path.exists(),
            "expected secrets.toml to exist"
        );
        Ok(())
    }

    #[test]
    fn resolve_project_uses_repo_config_dir_when_local_config_exists() -> eyre::Result<()> {
        let dir = TempDir::new()?;
        let local_config_dir = dir.path().join(".config").join("context");
        fs::create_dir_all(&local_config_dir)?;
        fs::write(local_config_dir.join("config.toml"), "")?;
        fs::create_dir_all(dir.path().join(".git"))?;
        let mut projects = BTreeMap::new();
        projects.insert(
            "example".to_string(),
            Project {
                repo: dir.path().to_path_buf(),
                data_dir: None,
                database: None,
            },
        );

        let config = AppConfig {
            embedding: Embedding {
                url: "http://localhost".to_string(),
                api_key: None,
                model: "model".to_string(),
                tokenizer: "hf:Qwen/Qwen3-Embedding-0.6B".to_string(),
                dialect: "openai".to_string(),
                timeout_seconds: 10,
                embedding_dim: 2,
                context_length: 8,
                stream_batch_size: 4,
                max_batch_size: 4,
                requests_per_minute: 1000,
                max_concurrent_requests: 300,
                tokens_per_minute: 1,
            },
            reranker: Reranker {
                url: "http://localhost".to_string(),
                api_key: None,
                model: "model".to_string(),
                dialect: "openai".to_string(),
                timeout_seconds: 10,
                instruction: None,
            },
            chunking: Chunking {
                max_chunk_size: 10,
                overlap: 0.2,
                max_parallel: 1,
                max_file_size: 1_024,
                large_file_threads: 1,
            },
            history: History {
                depth: 1,
                commit_size_limit_ratio: 1.0,
                multi_parents: false,
                issue_regex: "(#\\d+)".to_string(),
                commit_exclude_regex: None,
                author_exclude_regex: None,
                path_specs: String::new(),
            },
            projects,
            search: SearchOptions {
                limit: 1,
                min_score: 0.25,
                hybrid_weight: 0.6,
                path_prefixes: Vec::new(),
                file_exts: Vec::new(),
                decompose: false,
                rerank: false,
            },
            prompt: Prompting {
                tokenizer: String::new(),
            },
        };

        let resolved = resolve_project_for_cwd(&config, dir.path(), Some("example"))?;
        assert!(
            resolved.database_path.starts_with(&local_config_dir),
            "expected db path to be under repo .config/context"
        );
        assert!(
            local_config_dir.join(".gitignore").exists(),
            "expected .gitignore to be created in repo data directory"
        );

        Ok(())
    }
}

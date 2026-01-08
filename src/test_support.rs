use std::path::Path;

use confique::Layer;
use eyre::{Result, eyre};

use crate::config;
pub(crate) fn write_fixture_repo(root: &Path) -> Result<()> {
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

pub(crate) fn load_test_embedder() -> Result<(crate::embedding::Client, usize)> {
    let cli = config::Cli {
        config_file: None,
        command: config::Command::Index(config::IndexCli {
            embedding: <config::Embedding as confique::Config>::Layer::empty(),
            chunking: <config::Chunking as confique::Config>::Layer::empty(),
            history: <config::History as confique::Config>::Layer::empty(),
            project: config::ProjectCli { project: None },
        }),
    };

    let cfg = config::load_config(&cli)?;
    if cfg.embedding.api_key.is_none() {
        return Err(eyre!(
            "embedding api key missing; configure EMBEDDER_API_KEY or secrets config"
        ));
    }

    let embedder = crate::build_embedder(&cfg.embedding)?;
    Ok((embedder, cfg.embedding.embedding_dim))
}

pub(crate) fn load_test_reranker() -> Result<crate::reranker::Client> {
    let cli = config::Cli {
        config_file: None,
        command: config::Command::Search(config::SearchCli {
            embedding: <config::Embedding as confique::Config>::Layer::empty(),
            reranker: <config::Reranker as confique::Config>::Layer::empty(),
            project: config::ProjectCli { project: None },
            search: <config::SearchOptions as confique::Config>::Layer::empty(),
            query: "test".to_string(),
        }),
    };

    let cfg = config::load_config(&cli)?;
    if cfg.reranker.api_key.is_none() && cfg.embedding.api_key.is_none() {
        return Err(eyre!(
            "reranker api key missing; configure RERANKER_API_KEY (or EMBEDDER_API_KEY) or secrets config"
        ));
    }

    crate::build_reranker(&cfg)
}

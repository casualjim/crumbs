use eyre::{Result, eyre};

use crate::db::{Db, SearchRow};
use crate::embedding::{Client as EmbedClient, EmbeddingInput, EmbeddingProvider};

pub async fn search(
  db: &Db,
  embedder: &EmbedClient,
  query: &str,
  limit: usize,
) -> Result<Vec<SearchRow>> {
  let input = EmbeddingInput {
    text: query.to_string(),
    token_count: None,
  };
  let output = EmbeddingProvider::embed(embedder, &[input]).await?;
    let mut embeddings = output.embeddings;
    if embeddings.is_empty() {
        return Err(eyre!("embedder returned no embeddings for query"));
    }
    db.search(&embeddings.remove(0), limit)
}

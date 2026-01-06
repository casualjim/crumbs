use std::collections::{HashMap, HashSet};

use eyre::{Result, eyre};

use crate::embedding::{EmbeddingInput, EmbeddingProvider};

use super::batcher::BatchItem;

pub(crate) async fn embed_batch(
    client: &dyn EmbeddingProvider,
    batch: Vec<BatchItem>,
    embeddings: &mut HashMap<String, Vec<f32>>,
) -> Result<HashSet<String>> {
    if batch.is_empty() {
        return Ok(HashSet::new());
    }

    let inputs: Vec<EmbeddingInput> = batch
        .iter()
        .map(|item| EmbeddingInput {
            text: item.text.clone(),
            token_count: item.token_count,
        })
        .collect();
    let output = client.embed(&inputs).await?;
    if output.embeddings.len() != batch.len() {
        return Err(eyre!(
            "embedder returned {} embeddings for {} inputs",
            output.embeddings.len(),
            batch.len()
        ));
    }

    let mut touched = HashSet::new();
    for (item, embedding) in batch.into_iter().zip(output.embeddings.into_iter()) {
        embeddings.insert(item.chunk_id, embedding);
        touched.insert(item.file_path);
    }

    Ok(touched)
}

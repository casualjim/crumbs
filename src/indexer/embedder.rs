use std::sync::Arc;

use eyre::{Result, eyre};
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use tokio::sync::{mpsc, Semaphore};

use crate::embedding::{EmbeddingInput, EmbeddingProvider};

use super::batcher::{BatchItem, BatchMeta, TokenAwareBatcher};

pub(crate) async fn embed_batch(
    client: &dyn EmbeddingProvider,
    batch: Vec<BatchItem>,
) -> Result<EmbeddingResult> {
    if batch.is_empty() {
        return Ok(EmbeddingResult {
            items: Vec::new(),
            embeddings: Vec::new(),
        });
    }

    let mut inputs = Vec::with_capacity(batch.len());
    let mut items = Vec::with_capacity(batch.len());
    for item in batch {
        inputs.push(EmbeddingInput {
            text: item.text,
            token_count: item.token_count,
        });
        items.push(BatchMeta {
            chunk_id: item.chunk_id,
            file_path: item.file_path,
        });
    }
    let output = client.embed(&inputs).await?;
    if output.embeddings.len() != items.len() {
        return Err(eyre!(
            "embedder returned {} embeddings for {} inputs",
            output.embeddings.len(),
            items.len()
        ));
    }
    Ok(EmbeddingResult {
        items,
        embeddings: output.embeddings,
    })
}

pub(crate) struct EmbedderService {
    batcher: TokenAwareBatcher,
    batch_tx: Option<mpsc::Sender<Vec<BatchItem>>>,
}

pub(crate) struct EmbeddingResult {
    pub(crate) items: Vec<BatchMeta>,
    pub(crate) embeddings: Vec<Vec<f32>>,
}

impl EmbedderService {
    pub(crate) fn new(
        embedder: Arc<dyn EmbeddingProvider>,
        max_tokens: usize,
        max_batch_size: usize,
        workers: usize,
    ) -> (Self, mpsc::Receiver<Result<EmbeddingResult>>) {
        let worker_count = workers.max(1);
        let (batch_tx, mut batch_rx) = mpsc::channel::<Vec<BatchItem>>(worker_count * 2);
        let (result_tx, result_rx) = mpsc::channel::<Result<EmbeddingResult>>(worker_count * 2);
        let semaphore = Arc::new(Semaphore::new(worker_count));
        let dispatcher_embedder = Arc::clone(&embedder);
        tokio::spawn(async move {
            let mut in_flight: FuturesUnordered<_> = FuturesUnordered::new();
            let mut rx_closed = false;
            loop {
                tokio::select! {
                    batch = batch_rx.recv(), if !rx_closed && in_flight.len() < worker_count => {
                        match batch {
                            Some(batch) => {
                                let embedder = Arc::clone(&dispatcher_embedder);
                                let semaphore = Arc::clone(&semaphore);
                                in_flight.push(async move {
                                    let _permit = semaphore
                                        .acquire_owned()
                                        .await
                                        .map_err(|_| eyre!("embedder semaphore closed"))?;
                                    embed_batch(embedder.as_ref(), batch).await
                                });
                            }
                            None => {
                                rx_closed = true;
                            }
                        }
                    }
                    Some(result) = in_flight.next(), if !in_flight.is_empty() => {
                        let _ = result_tx.send(result).await;
                    }
                    else => {
                        if rx_closed {
                            break;
                        }
                    }
                }
            }
        });

        (
            Self {
            batcher: TokenAwareBatcher::new(max_tokens, max_batch_size),
            batch_tx: Some(batch_tx),
        },
            result_rx,
        )
    }

    pub(crate) async fn enqueue(&mut self, item: BatchItem) -> Result<bool> {
        if let Some(batch) = self.batcher.add(item) {
            if let Some(tx) = self.batch_tx.as_ref() {
                tx.send(batch).await?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) async fn flush(&mut self) -> Result<bool> {
        if self.batch_tx.is_none() {
            return Ok(false);
        }

        let mut sent = false;
        if let Some(batch) = self.batcher.flush() {
            if let Some(tx) = self.batch_tx.as_ref() {
                tx.send(batch).await?;
                sent = true;
            }
        }
        drop(self.batch_tx.take());
        Ok(sent)
    }

    pub(crate) fn remove_file(&mut self, file_path: &str) {
        self.batcher.remove_file(file_path);
    }
}

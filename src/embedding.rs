use std::time::Duration;

use async_trait::async_trait;
use eyre::Result;
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::debug;

use crate::reqwestx::api_client::{ApiClient, ApiClientConfig};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum ProviderDialect {
    #[default]
    OpenAI,
    DeepInfra,
}

impl std::fmt::Display for ProviderDialect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderDialect::OpenAI => write!(f, "openai"),
            ProviderDialect::DeepInfra => write!(f, "deepinfra"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    pub api_key: Option<SecretString>,
    pub base_url: String,
    pub timeout: Duration,
    pub dialect: ProviderDialect,
    pub model: String,
    pub embedding_dim: usize,
    pub requests_per_minute: usize,
    pub max_concurrent_requests: usize,
    pub tokens_per_minute: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingInput {
    pub text: String,
    pub token_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbedOutput {
    pub embeddings: Vec<Vec<f32>>,
}

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, input: &[EmbeddingInput]) -> Result<EmbedOutput>;
}

#[derive(Clone)]
pub struct Client {
    client: ApiClient,
    model: String,
    dimension: usize,
    dialect: ProviderDialect,
}

impl Client {
    pub fn new(config: EmbedderConfig) -> Result<Self> {
        let api_config = ApiClientConfig {
            base_url: config.base_url.clone(),
            api_key: config.api_key.clone(),
            max_concurrent_requests: config.max_concurrent_requests,
            max_requests_per_minute: config.requests_per_minute,
            max_tokens_per_minute: config.tokens_per_minute as usize,
            max_retries: 3,
            timeout: config.timeout,
        };

        let client = ApiClient::new(api_config)?;

        Ok(Self {
            client,
            model: config.model,
            dimension: config.embedding_dim,
            dialect: config.dialect,
        })
    }

    fn estimate_token_count(&self, input: &[EmbeddingInput]) -> u32 {
        let mut tokens: u32 = 0;
        for inp in input {
            tokens = tokens.saturating_add(inp.token_count as u32);
        }
        tokens
    }

    fn prepare_inputs(&self, input: &[EmbeddingInput]) -> Result<Vec<String>> {
        let mut batch_texts = Vec::with_capacity(input.len());
        for inp in input {
            batch_texts.push(inp.text.clone());
        }
        Ok(batch_texts)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbeddingObject {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbedApiResponse {
    data: Vec<EmbeddingObject>,
}

#[async_trait]
impl EmbeddingProvider for Client {
    async fn embed(&self, input: &[EmbeddingInput]) -> Result<EmbedOutput> {
        debug!("Embedding input batch_size: {}", input.len());
        let batch_texts = self.prepare_inputs(input)?;
        let estimated_tokens = self.estimate_token_count(input);

        let payload = match self.dialect {
            ProviderDialect::OpenAI | ProviderDialect::DeepInfra => {
                json!({
                  "input": &batch_texts,
                  "model": self.model,
                  "encoding_format": "float",
                  "dimensions": self.dimension
                })
            }
        };
        let response: EmbedApiResponse = self
            .client
            .post_json("/embeddings", &payload, estimated_tokens)
            .await?;

        let data = response.data;
        let mut embeddings = vec![Vec::new(); data.len()];
        for item in data {
            if item.index < embeddings.len() {
                embeddings[item.index] = item.embedding;
            }
        }

        Ok(EmbedOutput { embeddings })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic;
    use std::time::Duration;

    use secrecy::SecretString;

    #[test]
    fn embedder_new_should_not_panic_on_invalid_api_key() {
        let result = panic::catch_unwind(|| {
            let _ = Client::new(EmbedderConfig {
                api_key: Some(SecretString::from("bad\nkey")),
                base_url: "http://127.0.0.1:1".to_string(),
                timeout: Duration::from_secs(1),
                dialect: ProviderDialect::OpenAI,
                model: "test-model".to_string(),
                embedding_dim: 2,
                requests_per_minute: 1000,
                max_concurrent_requests: 300,
                tokens_per_minute: 1,
            });
        });

        assert!(
            result.is_ok(),
            "Client::new should return Err, not panic, for invalid API keys"
        );
    }
}

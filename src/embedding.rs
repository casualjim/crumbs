use std::time::Duration;

use async_trait::async_trait;
use eyre::Result;
use http::{
  HeaderValue,
  header::{AUTHORIZATION, CONTENT_TYPE},
};
use reqwest::Client as HttpClient;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{debug, error};

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
  pub context_length: usize,
  pub max_batch_size: usize,
  pub tokens_per_minute: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingInput {
  pub text: String,
  pub token_count: Option<usize>,
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
  client: HttpClient,
  api_key: Option<SecretString>,
  base_url: String,
  model: String,
  dimension: usize,
  dialect: ProviderDialect,
}

impl Client {
  pub fn new(config: EmbedderConfig) -> Result<Self> {
    if config.context_length == 0 {
      return Err(eyre::eyre!("context_length must be > 0"));
    }
    if config.max_batch_size == 0 {
      return Err(eyre::eyre!("max_batch_size must be > 0"));
    }
    let _ = config.tokens_per_minute;

    let client = HttpClient::builder()
      .default_headers({
        let mut headers = http::HeaderMap::new();
        if let Some(api_key) = &config.api_key {
          headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key.expose_secret())).unwrap(),
          );
        }
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers
      })
      .user_agent(format!(
        "Context/Embeddings; dialect={}",
        config.dialect
      ))
      .timeout(config.timeout)
      .build()?;

    Ok(Self {
      client,
      api_key: config.api_key,
      base_url: config.base_url,
      model: config.model,
      dimension: config.embedding_dim,
      dialect: config.dialect,
    })
  }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbeddingObject {
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
    let mut batch_texts = Vec::with_capacity(input.len());
    for inp in input {
      if inp.text.trim().is_empty() {
        return Err(eyre::eyre!("Input text cannot be empty"));
      }
      batch_texts.push(inp.text.trim());
    }

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

    let req = self.client.post(format!("{}/embeddings", self.base_url));
    let req = if let Some(api_key) = &self.api_key {
      req.bearer_auth(api_key.expose_secret())
    } else {
      req
    };

    let response = req
      .json(&payload)
      .send()
      .await
      .map_err(|e| {
        error!("Failed to send embedding request: {e}");
        e
      })?
      .error_for_status()
      .map_err(|e| {
        error!("Embedding request returned error status: {e}");
        e
      })?
      .json::<EmbedApiResponse>()
      .await
      .map_err(|e| {
        error!("Failed to parse embedding response: {e}");
        e
      })?;

    Ok(EmbedOutput {
      embeddings: response.data.into_iter().map(|embedding| embedding.embedding).collect(),
    })
  }
}

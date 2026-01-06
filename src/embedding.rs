use std::sync::Arc;
use std::time::{Duration, Instant};

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
use text_chunking::Tokenizer as ChunkTokenizer;
use tokio::sync::Mutex;
use tracing::warn;
use tracing::{debug, error};

use tiktoken_rs::{CoreBPE, cl100k_base, o200k_base, p50k_base, p50k_edit, r50k_base};
use tokenizers::Tokenizer as HfTokenizer;

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
    pub tokenizer: ChunkTokenizer,
    pub embedding_dim: usize,
    pub context_length: usize,
    pub max_batch_size: usize,
    pub tokens_per_minute: u32,
}

#[derive(Clone)]
enum TokenCounter {
    Characters,
    Tiktoken(CoreBPE),
    HuggingFace(Arc<HfTokenizer>),
}

impl TokenCounter {
    fn count_tokens(&self, text: &str) -> Result<usize> {
        match self {
            TokenCounter::Characters => Ok(text.chars().count()),
            TokenCounter::Tiktoken(encoder) => Ok(encoder.encode_ordinary(text).len()),
            TokenCounter::HuggingFace(tokenizer) => tokenizer
                .encode(text, false)
                .map(|encoding| encoding.len())
                .map_err(|err| eyre::eyre!("tokenizer encode failed: {err}")),
        }
    }
}

fn token_counter_from(tokenizer: &ChunkTokenizer) -> Result<TokenCounter> {
    match tokenizer {
        ChunkTokenizer::Characters => Ok(TokenCounter::Characters),
        ChunkTokenizer::Tiktoken(encoding) => {
            let encoder = match encoding.as_str() {
                "cl100k_base" => cl100k_base(),
                "p50k_base" => p50k_base(),
                "p50k_edit" => p50k_edit(),
                "r50k_base" => r50k_base(),
                "o200k_base" => o200k_base(),
                other => {
                    return Err(eyre::eyre!("Unknown tiktoken encoding: {other}"));
                }
            }
            .map_err(|err| eyre::eyre!("Failed to create tiktoken: {err}"))?;
            Ok(TokenCounter::Tiktoken(encoder))
        }
        ChunkTokenizer::PreloadedTiktoken(encoder) => Ok(TokenCounter::Tiktoken(
            Arc::try_unwrap(encoder.clone()).unwrap_or_else(|arc| (*arc).clone()),
        )),
        ChunkTokenizer::HuggingFace(model_id) => {
            let tokenizer = HfTokenizer::from_pretrained(model_id, None)
                .map_err(|err| eyre::eyre!("Failed to load HF tokenizer {model_id}: {err}"))?;
            Ok(TokenCounter::HuggingFace(Arc::new(tokenizer)))
        }
        ChunkTokenizer::PreloadedHuggingFace(tokenizer) => {
            Ok(TokenCounter::HuggingFace(tokenizer.clone()))
        }
    }
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
    token_counter: TokenCounter,
    dimension: usize,
    context_length: usize,
    max_batch_size: usize,
    dialect: ProviderDialect,
    rate_limit: Option<Arc<Mutex<RateLimiter>>>,
}

#[derive(Debug)]
struct TokenBucket {
    capacity: f64,
    tokens: f64,
    refill_rate: f64,
    last_refill: Instant,
}

impl TokenBucket {
    fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            capacity,
            tokens: capacity,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    fn try_consume(&mut self, tokens_needed: f64) -> Result<(), Duration> {
        self.refill();
        if self.tokens >= tokens_needed {
            self.tokens -= tokens_needed;
            Ok(())
        } else {
            let tokens_short = tokens_needed - self.tokens;
            let wait_seconds = tokens_short / self.refill_rate;
            Err(Duration::from_secs_f64(wait_seconds))
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let new_tokens = elapsed * self.refill_rate;
        self.tokens = (self.tokens + new_tokens).min(self.capacity);
        self.last_refill = now;
    }
}

#[derive(Debug)]
struct RateLimiter {
    token_bucket: TokenBucket,
    max_tokens_per_minute: u32,
}

impl Client {
    pub fn new(config: EmbedderConfig) -> Result<Self> {
        if config.embedding_dim == 0 {
            return Err(eyre::eyre!("embedding_dim must be > 0"));
        }
        if config.context_length == 0 {
            return Err(eyre::eyre!("context_length must be > 0"));
        }
        if config.max_batch_size == 0 {
            return Err(eyre::eyre!("max_batch_size must be > 0"));
        }
        let token_counter = token_counter_from(&config.tokenizer)?;
        let rate_limit = if config.tokens_per_minute == 0 {
            None
        } else {
            let token_rate = config.tokens_per_minute as f64 / 60.0;
            let token_capacity = (token_rate * 10.0).min(config.tokens_per_minute as f64);
            Some(Arc::new(Mutex::new(RateLimiter {
                token_bucket: TokenBucket::new(token_capacity, token_rate),
                max_tokens_per_minute: config.tokens_per_minute,
            })))
        };

        let mut headers = http::HeaderMap::new();
        if let Some(api_key) = &config.api_key {
            let value = HeaderValue::from_str(&format!("Bearer {}", api_key.expose_secret()))
                .map_err(|err| eyre::eyre!("invalid embedder api key: {err}"))?;
            headers.insert(AUTHORIZATION, value);
        }
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let client = HttpClient::builder()
            .default_headers(headers)
            .user_agent(format!("Context/Embeddings; dialect={}", config.dialect))
            .timeout(config.timeout)
            .build()?;

        Ok(Self {
            client,
            api_key: config.api_key,
            base_url: config.base_url,
            model: config.model,
            token_counter,
            dimension: config.embedding_dim,
            context_length: config.context_length,
            max_batch_size: config.max_batch_size,
            dialect: config.dialect,
            rate_limit,
        })
    }

    fn estimate_token_count(&self, input: &[EmbeddingInput]) -> u32 {
        let mut tokens: u32 = 0;
        for inp in input {
            let trimmed = inp.text.trim();
            if trimmed.is_empty() {
                continue;
            }
            let count = if let Some(count) = inp.token_count {
                count
            } else {
                self.token_counter
                    .count_tokens(trimmed)
                    .unwrap_or_else(|err| {
                        warn!("token count failed, falling back to char estimate: {err}");
                        (trimmed.len() / 4).max(1)
                    })
            };
            tokens = tokens.saturating_add(count as u32);
        }
        tokens
    }

    fn prepare_inputs(&self, input: &[EmbeddingInput]) -> Result<Vec<String>> {
        if input.is_empty() {
            return Err(eyre::eyre!("input batch cannot be empty"));
        }
        if input.len() > self.max_batch_size {
            return Err(eyre::eyre!(
                "input batch size {} exceeds max_batch_size {}",
                input.len(),
                self.max_batch_size
            ));
        }

        let mut batch_texts = Vec::with_capacity(input.len());
        for inp in input {
            let trimmed = inp.text.trim();
            if trimmed.is_empty() {
                return Err(eyre::eyre!("Input text cannot be empty"));
            }
            let token_count = if let Some(count) = inp.token_count {
                count
            } else {
                self.token_counter
                    .count_tokens(trimmed)
                    .unwrap_or_else(|err| {
                        warn!("token count failed, falling back to char estimate: {err}");
                        trimmed.chars().count()
                    })
            };
            if token_count > self.context_length {
                return Err(eyre::eyre!(
                    "input length {} exceeds context_length {}",
                    token_count,
                    self.context_length
                ));
            }
            batch_texts.push(trimmed.to_string());
        }
        Ok(batch_texts)
    }

    fn validate_embeddings(&self, input_len: usize, embeddings: &[Vec<f32>]) -> Result<()> {
        if embeddings.is_empty() {
            return Err(eyre::eyre!("embedder returned no embeddings"));
        }
        if embeddings.len() != input_len {
            return Err(eyre::eyre!(
                "embedding count mismatch: expected {}, got {}",
                input_len,
                embeddings.len()
            ));
        }
        for (idx, embedding) in embeddings.iter().enumerate() {
            if embedding.len() != self.dimension {
                return Err(eyre::eyre!(
                    "embedding dimension mismatch at index {}: expected {}, got {}",
                    idx,
                    self.dimension,
                    embedding.len()
                ));
            }
        }
        Ok(())
    }

    async fn enforce_rate_limit(&self, input: &[EmbeddingInput]) -> Result<()> {
        let Some(state) = &self.rate_limit else {
            return Ok(());
        };
        let tokens = self.estimate_token_count(input);

        if tokens == 0 {
            return Ok(());
        }

        loop {
            let wait_duration = {
                let mut guard = state.lock().await;
                if tokens > guard.max_tokens_per_minute {
                    return Err(eyre::eyre!(
                        "input tokens {} exceed tokens_per_minute {}",
                        tokens,
                        guard.max_tokens_per_minute
                    ));
                }
                match guard.token_bucket.try_consume(tokens as f64) {
                    Ok(()) => return Ok(()),
                    Err(wait) => wait,
                }
            };
            let wait_with_buffer = wait_duration + Duration::from_millis(10);
            if wait_with_buffer > Duration::from_millis(100) {
                debug!("Rate limit: waiting {:?}", wait_with_buffer);
            }
            tokio::time::sleep(wait_with_buffer).await;
        }
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
        let batch_texts = self.prepare_inputs(input)?;
        self.enforce_rate_limit(input).await?;

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

        let embeddings: Vec<Vec<f32>> = response
            .data
            .into_iter()
            .map(|embedding| embedding.embedding)
            .collect();
        self.validate_embeddings(input.len(), &embeddings)?;

        Ok(EmbedOutput { embeddings })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic;
    use std::time::Duration;

    use secrecy::SecretString;

    fn build_test_embedder(
        embedding_dim: usize,
        context_length: usize,
        max_batch_size: usize,
    ) -> Client {
        Client::new(EmbedderConfig {
            api_key: None,
            base_url: "http://127.0.0.1:1".to_string(),
            timeout: Duration::from_secs(5),
            dialect: ProviderDialect::OpenAI,
            model: "test-model".to_string(),
            tokenizer: ChunkTokenizer::Characters,
            embedding_dim,
            context_length,
            max_batch_size,
            tokens_per_minute: 1_000,
        })
        .expect("build test embedder")
    }

    #[test]
    fn embedder_new_should_not_panic_on_invalid_api_key() {
        let result = panic::catch_unwind(|| {
            let _ = Client::new(EmbedderConfig {
                api_key: Some(SecretString::from("bad\nkey")),
                base_url: "http://127.0.0.1:1".to_string(),
                timeout: Duration::from_secs(1),
                dialect: ProviderDialect::OpenAI,
                model: "test-model".to_string(),
                tokenizer: ChunkTokenizer::Characters,
                embedding_dim: 2,
                context_length: 8,
                max_batch_size: 1,
                tokens_per_minute: 1,
            });
        });

        assert!(
            result.is_ok(),
            "Client::new should return Err, not panic, for invalid API keys"
        );
    }

    #[test]
    fn embedder_new_rejects_zero_dimension() {
        let result = Client::new(EmbedderConfig {
            api_key: None,
            base_url: "http://127.0.0.1:1".to_string(),
            timeout: Duration::from_secs(1),
            dialect: ProviderDialect::OpenAI,
            model: "test-model".to_string(),
            tokenizer: ChunkTokenizer::Characters,
            embedding_dim: 0,
            context_length: 8,
            max_batch_size: 1,
            tokens_per_minute: 1,
        });

        assert!(
            result.is_err(),
            "expected Client::new to reject embedding_dim = 0"
        );
    }

    #[test]
    fn embedder_rejects_empty_input_batch() {
        let client = build_test_embedder(2, 8, 4);
        let result = client.prepare_inputs(&[]);

        assert!(
            result.is_err(),
            "expected embedder to reject empty input batches"
        );
    }

    #[test]
    fn embedder_rejects_mismatched_embedding_count() {
        let client = build_test_embedder(2, 8, 4);
        let result = client.validate_embeddings(1, &[vec![0.1, 0.2], vec![0.3, 0.4]]);

        assert!(
            result.is_err(),
            "expected error when embedding count does not match inputs"
        );
    }

    #[test]
    fn embedder_rejects_dimension_mismatch() {
        let client = build_test_embedder(2, 8, 4);
        let result = client.validate_embeddings(1, &[vec![0.1]]);

        assert!(
            result.is_err(),
            "expected error when embedding dimension does not match config"
        );
    }

    #[test]
    fn embedder_rejects_empty_response() {
        let client = build_test_embedder(2, 8, 4);
        let result = client.validate_embeddings(1, &[]);

        assert!(
            result.is_err(),
            "expected error when embedder returns no embeddings"
        );
    }

    #[test]
    fn embedder_enforces_max_batch_size() {
        let client = build_test_embedder(2, 8, 1);
        let inputs = vec![
            EmbeddingInput {
                text: "first".to_string(),
                token_count: None,
            },
            EmbeddingInput {
                text: "second".to_string(),
                token_count: None,
            },
        ];
        let result = client.prepare_inputs(&inputs);

        assert!(
            result.is_err(),
            "expected error when batch exceeds max_batch_size"
        );
    }

    #[test]
    fn embedder_enforces_context_length() {
        let client = build_test_embedder(2, 1, 4);
        let inputs = vec![EmbeddingInput {
            text: "hello".to_string(),
            token_count: None,
        }];
        let result = client.prepare_inputs(&inputs);

        assert!(
            result.is_err(),
            "expected error when input exceeds context_length"
        );
    }
}

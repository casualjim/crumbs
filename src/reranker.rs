use std::time::Duration;

use async_trait::async_trait;
use eyre::Result;
use http::{
    HeaderValue,
    header::{AUTHORIZATION, CONTENT_TYPE},
};
use reqwest::Client as HttpClient;
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use serde_json::json;
use tracing::{debug, error};

use crate::embedding::ProviderDialect;

#[derive(Debug, Clone)]
pub struct RerankerConfig {
    pub api_key: Option<SecretString>,
    pub base_url: String,
    pub timeout: Duration,
    pub dialect: ProviderDialect,
    pub model: String,
    pub instruction: Option<String>,
}

#[async_trait]
pub trait RerankingProvider: Send + Sync {
    async fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f64>>;
}

#[derive(Clone)]
pub struct Client {
    client: HttpClient,
    api_key: Option<SecretString>,
    base_url: String,
    model: String,
    instruction: Option<String>,
    dialect: ProviderDialect,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RerankApiResponse {
    scores: Vec<f64>,
    #[serde(default)]
    input_tokens: Option<i64>,
}

impl Client {
    pub fn new(config: RerankerConfig) -> Result<Self> {
        let mut headers = http::HeaderMap::new();
        if let Some(api_key) = &config.api_key {
            let value = HeaderValue::from_str(&format!("Bearer {}", api_key.expose_secret()))
                .map_err(|err| eyre::eyre!("invalid reranker api key: {err}"))?;
            headers.insert(AUTHORIZATION, value);
        }
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let client = HttpClient::builder()
            .default_headers(headers)
            .user_agent(format!("Context/Reranker; dialect={}", config.dialect))
            .timeout(config.timeout)
            .build()?;

        Ok(Self {
            client,
            api_key: config.api_key,
            base_url: config.base_url,
            model: config.model,
            instruction: config.instruction,
            dialect: config.dialect,
        })
    }

    async fn rerank_deepinfra(&self, query: &str, documents: &[String]) -> Result<Vec<f64>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        if query.trim().is_empty() {
            return Err(eyre::eyre!("rerank query cannot be empty"));
        }

        let queries = vec![query; documents.len()];
        let mut payload = json!({
            "queries": queries,
            "documents": documents,
        });

        if let Some(instruction) = &self.instruction {
            payload["instruction"] = json!(instruction);
        }

        let endpoint = format!("{}/inference/{}", self.base_url, self.model);
        let mut req = self.client.post(&endpoint).json(&payload);
        if let Some(api_key) = &self.api_key {
            req = req.bearer_auth(api_key.expose_secret());
        }

        let response = req
            .send()
            .await
            .map_err(|e| {
                error!("Failed to send reranking request: {e}");
                e
            })?
            .error_for_status()
            .map_err(|e| {
                error!("Reranking request returned error status: {e}");
                e
            })?
            .json::<RerankApiResponse>()
            .await
            .map_err(|e| {
                error!("Failed to parse reranking response: {e}");
                e
            })?;

        if let Some(input_tokens) = response.input_tokens {
            debug!("Reranking used {} input tokens", input_tokens);
        }

        Ok(response
            .scores
            .into_iter()
            .map(|s| s.clamp(0.0, 1.0))
            .collect())
    }
}

#[async_trait]
impl RerankingProvider for Client {
    async fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f64>> {
        match self.dialect {
            ProviderDialect::DeepInfra => self.rerank_deepinfra(query, documents).await,
            ProviderDialect::OpenAI => Err(eyre::eyre!(
                "openai reranking is not configured; use deepinfra or add openai support"
            )),
        }
    }
}

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use bytes::Bytes;
use eyre::Result;
use futures::{Stream, StreamExt};
use http::header::{AUTHORIZATION, CONTENT_TYPE};
use http::{HeaderMap, HeaderValue};
use pin_project_lite::pin_project;
use reqwest::{Client, Response};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};
use tokio::time::sleep;
use tracing::{debug, trace};

#[derive(Debug, Clone)]
pub struct ApiClientConfig {
    pub base_url: String,
    pub api_key: Option<SecretString>,
    pub max_concurrent_requests: usize,
    pub max_requests_per_minute: usize,
    pub max_tokens_per_minute: usize,
    pub max_retries: usize,
    pub timeout: Duration,
}

impl Default for ApiClientConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            api_key: None,
            max_concurrent_requests: 300,
            max_requests_per_minute: 1000,
            max_tokens_per_minute: 1_000_000,
            max_retries: 3,
            timeout: Duration::from_secs(90),
        }
    }
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
    request_bucket: TokenBucket,
    token_bucket: TokenBucket,
}

#[derive(Clone)]
pub struct ApiClient {
    client: Client,
    config: ApiClientConfig,
    concurrent_semaphore: Arc<Semaphore>,
    rate_limiter: Arc<Mutex<RateLimiter>>,
}

impl ApiClient {
    pub fn new(config: ApiClientConfig) -> Result<Self> {
        let client = Client::builder()
            .default_headers({
                let mut headers = HeaderMap::new();
                if let Some(api_key) = &config.api_key {
                    let value =
                        HeaderValue::from_str(&format!("Bearer {}", api_key.expose_secret()))
                            .map_err(|err| eyre::eyre!("invalid api key header: {err}"))?;
                    headers.insert(AUTHORIZATION, value);
                }
                headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
                headers
            })
            .user_agent("Breeze/ApiClient")
            .timeout(config.timeout)
            .build()?;

        let concurrent_semaphore = Arc::new(Semaphore::new(config.max_concurrent_requests));

        let request_rate = config.max_requests_per_minute as f64 / 60.0;
        let token_rate = config.max_tokens_per_minute as f64 / 60.0;

        let request_capacity = (request_rate * 10.0).min(config.max_requests_per_minute as f64);
        let token_capacity = (token_rate * 10.0).min(config.max_tokens_per_minute as f64);

        let rate_limiter = Arc::new(Mutex::new(RateLimiter {
            request_bucket: TokenBucket::new(request_capacity, request_rate),
            token_bucket: TokenBucket::new(token_capacity, token_rate),
        }));

        Ok(Self {
            client,
            config,
            concurrent_semaphore,
            rate_limiter,
        })
    }

    pub async fn post_json<Req, Res>(
        &self,
        path: &str,
        payload: &Req,
        token_count: u32,
    ) -> Result<Res>
    where
        Req: Serialize,
        Res: for<'de> Deserialize<'de>,
    {
        let url = format!("{}{}", self.config.base_url, path);
        let body_bytes = serde_json::to_vec(payload)?;

        let mut retries = 0;
        loop {
            debug!("Attempting request to {} (attempt {})", url, retries + 1);

            self.wait_for_rate_limit(token_count).await?;

            let request_permit = self.concurrent_semaphore.clone().acquire_owned().await?;

            let mut request = self
                .client
                .post(&url)
                .header("Content-Type", "application/json")
                .body(body_bytes.clone());

            if let Some(api_key) = &self.config.api_key {
                request = request.header(
                    "Authorization",
                    format!("Bearer {}", api_key.expose_secret()),
                );
            }

            let result = request.send().await;

            match result {
                Ok(response) => {
                    let status = response.status();

                    if status.is_success() {
                        let body_bytes = read_body_with_permit(response, request_permit).await?;
                        let result = serde_json::from_slice(&body_bytes)?;
                        debug!("Request to {} succeeded with status {}", url, status);
                        return Ok(result);
                    } else if should_retry(status.as_u16()) && retries < self.config.max_retries {
                        drop(request_permit);

                        retries += 1;
                        let backoff = calculate_backoff(retries);
                        debug!(
                            "Retrying after {} seconds due to status {}",
                            backoff.as_secs(),
                            status
                        );
                        sleep(backoff).await;
                        continue;
                    } else {
                        let error_text = response.text().await.unwrap_or_default();
                        return Err(eyre::Report::msg(format!(
                            "API request failed with status {}: {}",
                            status, error_text
                        )));
                    }
                }
                Err(e) => {
                    drop(request_permit);

                    if is_retryable_error(&e) && retries < self.config.max_retries {
                        retries += 1;
                        let backoff = calculate_backoff(retries);
                        debug!(
                            "Retrying after {} seconds due to error: {}",
                            backoff.as_secs(),
                            e
                        );
                        sleep(backoff).await;
                        continue;
                    } else {
                        return Err(e.into());
                    }
                }
            }
        }
    }

    async fn wait_for_rate_limit(&self, token_count: u32) -> Result<()> {
        loop {
            let wait_duration = {
                let mut limiter = self.rate_limiter.lock().await;

                let request_result = limiter.request_bucket.try_consume(1.0);
                let token_result = limiter.token_bucket.try_consume(token_count as f64);

                match (request_result, token_result) {
                    (Ok(()), Ok(())) => {
                        return Ok(());
                    }
                    (Ok(()), Err(token_wait)) => {
                        limiter.request_bucket.tokens += 1.0;
                        token_wait
                    }
                    (Err(request_wait), Ok(())) => {
                        limiter.token_bucket.tokens += token_count as f64;
                        request_wait
                    }
                    (Err(request_wait), Err(token_wait)) => request_wait.max(token_wait),
                }
            };

            let wait_with_buffer = wait_duration + Duration::from_millis(10);

            if wait_with_buffer > Duration::from_millis(100) {
                debug!(
                    "Rate limit: waiting {:?} before next request",
                    wait_with_buffer
                );
            }

            sleep(wait_with_buffer).await;
        }
    }
}

async fn read_body_with_permit(
    response: Response,
    request_permit: OwnedSemaphorePermit,
) -> Result<Bytes> {
    let stream = response.bytes_stream();
    let guarded_stream = GuardedStream::new(stream, request_permit);

    let chunks: Vec<_> = guarded_stream.collect::<Vec<_>>().await;

    let mut combined = Vec::new();
    for chunk in chunks {
        let chunk = chunk?;
        combined.extend_from_slice(chunk.as_ref());
    }

    Ok(Bytes::from(combined))
}

pin_project! {
    struct GuardedStream<S> {
        #[pin]
        inner: S,
        _request_permit: Option<OwnedSemaphorePermit>,
    }
}

impl<S> GuardedStream<S> {
    fn new(inner: S, request_permit: OwnedSemaphorePermit) -> Self {
        Self {
            inner,
            _request_permit: Some(request_permit),
        }
    }
}

impl<S, E> Stream for GuardedStream<S>
where
    S: Stream<Item = Result<Bytes, E>>,
{
    type Item = Result<Bytes, E>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();

        match this.inner.poll_next(cx) {
            Poll::Ready(None) => {
                trace!("Response body fully consumed, releasing permit");
                *this._request_permit = None;
                Poll::Ready(None)
            }
            other => other,
        }
    }
}

fn should_retry(status: u16) -> bool {
    matches!(status, 429 | 500 | 502 | 503 | 504)
}

fn is_retryable_error(error: &reqwest::Error) -> bool {
    error.is_timeout() || error.is_connect()
}

fn calculate_backoff(retry_count: usize) -> Duration {
    #[cfg(test)]
    {
        let base = 2u64;
        let millis = base.pow(retry_count as u32).min(60) * 10;
        Duration::from_millis(millis)
    }

    #[cfg(not(test))]
    {
        let base = 2u64;
        let seconds = base.pow(retry_count as u32).min(60);
        Duration::from_secs(seconds)
    }
}

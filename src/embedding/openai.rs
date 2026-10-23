//! OpenAI-compatible embedding provider.
//!
//! Calls any OpenAI-compatible `/v1/embeddings` endpoint.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::EmbeddingConfig;
use crate::error::{McpMemoryError, Result};

use super::EmbeddingProvider;

pub struct OpenAiEmbedder {
    client: reqwest::Client,
    endpoint: String,
    model: String,
    dimensions: usize,
    api_key: Option<String>,
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
    dimensions: usize,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl OpenAiEmbedder {
    pub fn new(config: &EmbeddingConfig) -> Result<Self> {
        let api_key = match &config.api_key_env {
            Some(env_var) => {
                let key = std::env::var(env_var).map_err(|_| McpMemoryError::Config {
                    message: format!("Environment variable '{}' not set for API key", env_var),
                })?;
                Some(key)
            }
            None => None,
        };

        Ok(Self {
            // Generous timeout: embedding an 8B model batch can take seconds,
            // but a hung endpoint must not block recall/remember forever.
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            endpoint: config.endpoint.clone(),
            model: config.model.clone(),
            dimensions: config.dimensions,
            api_key,
        })
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAiEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut results = self.embed_batch(&[text]).await?;
        results.pop().ok_or_else(|| McpMemoryError::Embedding {
            message: "empty response from embedding API".into(),
        })
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let body = EmbeddingRequest {
            model: &self.model,
            input: texts.to_vec(),
            dimensions: self.dimensions,
        };

        let mut req = self.client.post(&self.endpoint);
        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }
        let resp = req
            .json(&body)
            .send()
            .await
            .map_err(|e| McpMemoryError::Embedding {
                message: format!("HTTP request failed: {e}"),
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(McpMemoryError::Embedding {
                message: format!("API returned {status}: {body}"),
            });
        }

        let parsed: EmbeddingResponse =
            resp.json().await.map_err(|e| McpMemoryError::Embedding {
                message: format!("Failed to parse response: {e}"),
            })?;

        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

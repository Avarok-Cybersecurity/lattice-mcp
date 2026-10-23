//! HTTP-based reranker — supports OpenAI-compatible and Jina/BAAI API formats.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::{RerankerApiFormat, RerankerConfig};
use crate::error::{McpMemoryError, Result};

use super::{RerankCandidate, RerankResult, RerankerProvider};

pub struct HttpReranker {
    client: reqwest::Client,
    endpoint: String,
    model: String,
    api_format: RerankerApiFormat,
    api_key: Option<String>,
}

#[derive(Serialize)]
struct OpenAiRerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: Vec<&'a str>,
}

#[derive(Serialize)]
struct JinaRerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: Vec<&'a str>,
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<RerankResponseItem>,
}

#[derive(Deserialize)]
struct RerankResponseItem {
    index: usize,
    relevance_score: f32,
}

impl HttpReranker {
    pub fn new(config: &RerankerConfig) -> Result<Self> {
        let api_key = match &config.api_key_env {
            Some(env_var) => {
                let key = std::env::var(env_var).map_err(|_| McpMemoryError::Config {
                    message: format!(
                        "Environment variable '{}' not set for reranker API key",
                        env_var
                    ),
                })?;
                Some(key)
            }
            None => None,
        };

        Ok(Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            endpoint: config.endpoint.clone(),
            model: config.model.clone(),
            api_format: config.api_format.clone(),
            api_key,
        })
    }
}

#[async_trait]
impl RerankerProvider for HttpReranker {
    async fn rerank(
        &self,
        query: &str,
        candidates: &[RerankCandidate],
    ) -> Result<Vec<RerankResult>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let docs: Vec<&str> = candidates.iter().map(|c| c.document.as_str()).collect();

        let mut req = self.client.post(&self.endpoint);
        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }

        let req = match self.api_format {
            RerankerApiFormat::Openai => req.json(&OpenAiRerankRequest {
                model: &self.model,
                query,
                documents: docs,
            }),
            RerankerApiFormat::Jina => req.json(&JinaRerankRequest {
                model: &self.model,
                query,
                documents: docs,
            }),
        };

        let resp = req.send().await.map_err(|e| McpMemoryError::Reranker {
            message: format!("HTTP request failed: {e}"),
        })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(McpMemoryError::Reranker {
                message: format!("API returned {status}: {body}"),
            });
        }

        let parsed: RerankResponse = resp.json().await.map_err(|e| McpMemoryError::Reranker {
            message: format!("Failed to parse response: {e}"),
        })?;

        Ok(parsed
            .results
            .into_iter()
            .map(|r| RerankResult {
                index: r.index,
                relevance_score: r.relevance_score,
            })
            .collect())
    }
}

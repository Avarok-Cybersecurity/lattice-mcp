//! Chat provider abstraction + OpenAI implementation (SBIO boundary).
//!
//! Used by the `reflect` tool for LLM-based memory consolidation.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::ConsolidationConfig;
use crate::embedding::chat_types::{DistilledKnowledge, EntailmentVerdict, QueryTransform};
use crate::error::{McpMemoryError, Result};
use crate::memory::types::Memory;

/// SBIO boundary: abstracts LLM chat completions.
///
/// Beyond `summarize`, three optional methods support LLM-augmented features
/// (Steps 7/9/10). Their default impls return `Ok(None)` so a provider that
/// does not implement them transparently falls back to the deterministic
/// algorithmic path (PCND — no implicit behavior, explicit decline signal).
#[async_trait]
pub trait ChatProvider: Send + Sync {
    /// Summarize a cluster of memories into a single concise knowledge statement.
    async fn summarize(&self, memories: &[Memory]) -> Result<String>;

    /// Judge whether `candidate` contradicts/entails `existing` (Step 9).
    async fn entailment(
        &self,
        _existing: &str,
        _candidate: &str,
    ) -> Result<Option<EntailmentVerdict>> {
        Ok(None)
    }

    /// Produce a structured distillation of a memory cluster (Step 7).
    async fn distill(&self, _memories: &[Memory]) -> Result<Option<DistilledKnowledge>> {
        Ok(None)
    }

    /// Rewrite / expand a retrieval query before embedding (Step 10).
    async fn transform_query(&self, _query: &str) -> Result<Option<QueryTransform>> {
        Ok(None)
    }
}

pub struct OpenAiChat {
    client: reqwest::Client,
    endpoint: String,
    model: String,
    api_key: String,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    max_tokens: u32,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: String,
}

impl OpenAiChat {
    pub fn new(config: &ConsolidationConfig, api_key_env: Option<&str>) -> Result<Option<Self>> {
        let (endpoint, model) = match (&config.chat_endpoint, &config.chat_model) {
            (Some(ep), Some(m)) => (ep.clone(), m.clone()),
            _ => return Ok(None),
        };

        let api_key = match api_key_env {
            Some(env_var) => std::env::var(env_var).map_err(|_| McpMemoryError::Config {
                message: format!("Environment variable '{env_var}' not set for chat API key"),
            })?,
            None => String::new(),
        };

        Ok(Some(Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            endpoint,
            model,
            api_key,
        }))
    }
}

#[async_trait]
impl ChatProvider for OpenAiChat {
    async fn summarize(&self, memories: &[Memory]) -> Result<String> {
        let memories_text: Vec<String> = memories
            .iter()
            .enumerate()
            .map(|(i, m)| {
                format!(
                    "{}. [{}] {}",
                    i + 1,
                    m.category.collection_name(),
                    m.content
                )
            })
            .collect();
        let joined = memories_text.join("\n");

        let system_prompt = "You are a memory consolidation system. Given a cluster of related \
            episodic memories from a coding agent, synthesize them into a single concise knowledge \
            statement (1-3 sentences) that captures the key insight, pattern, or fact. Focus on \
            what was learned, not the individual events. Output ONLY the knowledge statement.";

        let user_prompt = format!("Consolidate these related memories:\n\n{joined}");

        let body = ChatRequest {
            model: &self.model,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: system_prompt,
                },
                ChatMessage {
                    role: "user",
                    content: &user_prompt,
                },
            ],
            max_tokens: 256,
        };

        let resp = self
            .client
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| McpMemoryError::Chat {
                message: format!("HTTP request failed: {e}"),
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(McpMemoryError::Chat {
                message: format!("API returned {status}: {body}"),
            });
        }

        let parsed: ChatResponse = resp.json().await.map_err(|e| McpMemoryError::Chat {
            message: format!("Failed to parse response: {e}"),
        })?;

        parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content.trim().to_string())
            .ok_or_else(|| McpMemoryError::Chat {
                message: "empty response from chat API".into(),
            })
    }
}

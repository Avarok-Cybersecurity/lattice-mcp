use crate::error::{McpMemoryError, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

mod importance;
mod types;
mod validate;
pub use importance::*;
pub use types::*;

#[cfg(test)]
mod tests;

/// Top-level configuration — all fields required (PCND).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub database: DatabaseConfig,
    pub embedding: EmbeddingConfig,
    pub memory: MemoryConfig,
    pub retrieval: RetrievalConfig,
    pub consolidation: ConsolidationConfig,
    /// Optional cross-encoder reranker for retrieval refinement.
    pub reranker: Option<RerankerConfig>,
    /// Optional adaptive importance scoring (all 7 algorithms).
    /// Omit to use legacy `access_adjusted_importance`.
    pub importance: Option<ImportanceConfig>,
    /// Optional write-policy: novelty tagging and near-duplicate handling.
    /// Omit (or `enabled = false`) to keep today's unconditional write path.
    #[serde(default)]
    pub write_policy: Option<WritePolicyConfig>,
    /// Optional adaptive fusion of the four HybridRAG signals.
    /// Omit (or `enabled = false`) to keep fixed-weight `hybrid_score`.
    #[serde(default)]
    pub fusion: Option<FusionConfig>,
    /// Optional utility-driven lifecycle (budgeted demotion/eviction).
    /// Omit (or `enabled = false`) to never demote/evict (today's behavior).
    #[serde(default)]
    pub lifecycle: Option<LifecycleConfig>,
    /// Optional MMR result diversification.
    /// Omit (or `enabled = false`) to keep pure relevance ranking.
    #[serde(default)]
    pub diversify: Option<DiversifyConfig>,
    /// Optional write-time contradiction detection (experimental).
    /// Omit (or `enabled = false`) to keep blind upserts (today's behavior).
    #[serde(default)]
    pub contradiction: Option<ContradictionConfig>,
    /// Optional query transformation before embedding.
    /// Omit (or `enabled = false`) to embed the raw query (today's behavior).
    #[serde(default)]
    pub query_transform: Option<QueryTransformConfig>,
    /// Optional persistent user-avatar subsystem (strict local-first).
    /// Omit (or `enabled = false`) to keep the avatar tools inert.
    #[serde(default)]
    pub avatar: Option<AvatarConfig>,
    /// Optional multi-signal retrieval (blend a lexical signal into scoring).
    /// Omit (or `enabled = false`) to keep pure semantic+graph scoring.
    #[serde(default)]
    pub multi_signal: Option<MultiSignalConfig>,
}

impl Config {
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| McpMemoryError::Config {
            message: format!("Failed to read config file {}: {}", path.display(), e),
        })?;
        let config: Self = toml::from_str(&content).map_err(|e| McpMemoryError::Config {
            message: format!("Failed to parse config: {}", e),
        })?;
        config.validate()?;
        Ok(config)
    }
}

/// Expand ~ to the user's home directory.
pub fn expand_path(path: &str) -> PathBuf {
    if path.starts_with('~') {
        if let Some(home) = dirs_home() {
            return home.join(&path[2..]);
        }
    }
    PathBuf::from(path)
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
        .map(PathBuf::from)
}

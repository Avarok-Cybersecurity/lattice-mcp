use crate::error::{McpMemoryError, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Top-level configuration — all fields required (PCND).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub database: DatabaseConfig,
    pub embedding: EmbeddingConfig,
    pub memory: MemoryConfig,
    pub retrieval: RetrievalConfig,
    pub consolidation: ConsolidationConfig,
    pub hooks: HooksConfig,
    /// Optional cross-encoder reranker for retrieval refinement.
    pub reranker: Option<RerankerConfig>,
    /// Optional adaptive importance scoring (all 7 algorithms).
    /// Omit to use legacy `access_adjusted_importance`.
    pub importance: Option<ImportanceConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// "embedded" or "remote"
    pub mode: DatabaseMode,
    /// Data directory for embedded mode (e.g. ~/.lattice-mcp/data)
    pub data_dir: PathBuf,
    /// URL for remote mode (e.g. http://192.168.68.61:6334)
    pub remote_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatabaseMode {
    Embedded,
    Remote,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// OpenAI-compatible embeddings endpoint
    pub endpoint: String,
    /// Model name (e.g. "text-embedding-3-small")
    pub model: String,
    /// Vector dimensionality (e.g. 1536)
    pub dimensions: usize,
    /// Environment variable name holding the API key (omit for local endpoints)
    pub api_key_env: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Ebbinghaus half-life for episodic memory decay (days)
    pub episodic_decay_half_life_days: f64,
    /// Ebbinghaus half-life for semantic memory decay (days)
    pub semantic_decay_half_life_days: f64,
    /// Ebbinghaus half-life for procedural memory decay (days)
    pub procedural_decay_half_life_days: f64,
    /// Auto-consolidate after N related episodic memories
    pub consolidation_threshold: usize,
    /// Max results returned by recall/get_context
    pub max_retrieval_results: usize,
    /// Max number of pinned memories allowed
    pub max_pinned_memories: usize,
    /// Max content length (chars) for a single pinned memory
    pub max_pinned_content_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Weight for vector similarity in hybrid score (α)
    pub vector_weight: f32,
    /// Weight for graph relevance in hybrid score (β)
    pub graph_weight: f32,
    /// Weight for temporal recency in hybrid score (γ)
    pub recency_weight: f32,
    /// Weight for importance in hybrid score (δ)
    pub importance_weight: f32,
    /// Max graph traversal depth for context expansion
    pub graph_hop_depth: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationConfig {
    /// "llm", "algorithmic", or "auto"
    pub mode: ConsolidationMode,
    /// Chat completions endpoint (required for LLM mode)
    pub chat_endpoint: Option<String>,
    /// Chat model name (required for LLM mode)
    pub chat_model: Option<String>,
    /// Minimum importance for a memory to be included in clustering.
    /// Memories below this threshold are skipped during reflect.
    /// Default: 0.4 (excludes auto-captured 0.3 events like session-starts).
    #[serde(default = "default_min_importance")]
    pub min_importance: f32,
    /// Temporal half-life (days) for clustering proximity weighting.
    /// Memories this many days apart have half the effective term overlap.
    /// Default: 7.0 days.
    #[serde(default = "default_temporal_half_life")]
    pub temporal_half_life_days: f64,
}

fn default_min_importance() -> f32 {
    0.4
}

fn default_temporal_half_life() -> f64 {
    7.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsolidationMode {
    Llm,
    Algorithmic,
    Auto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HooksConfig {
    /// "hybrid", "auto_only", or "manual_only"
    pub mode: HooksMode,
    /// Tool patterns to auto-capture (e.g. ["Edit", "Write", "Bash"])
    pub auto_capture_patterns: Vec<String>,
    /// Default importance for auto-captured memories
    pub auto_importance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HooksMode {
    Hybrid,
    AutoOnly,
    ManualOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankerConfig {
    /// Reranker HTTP endpoint (e.g. "http://192.168.68.61:8081/v1/rerank")
    pub endpoint: String,
    /// Model name sent in the rerank request
    pub model: String,
    /// API format: "openai" or "jina"
    pub api_format: RerankerApiFormat,
    /// Environment variable holding the API key (omit for local endpoints)
    pub api_key_env: Option<String>,
    /// Blend weight: final = λ·reranker + (1-λ)·hybrid. Range [0.0, 1.0].
    pub reranker_weight: f32,
    /// Over-fetch multiplier: vector search retrieves limit × this value
    pub candidate_multiplier: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RerankerApiFormat {
    Openai,
    Jina,
}

/// Adaptive importance scoring configuration.
/// When present and `enabled = true`, replaces legacy `access_adjusted_importance`
/// with a composite of Bayesian, FSRS, SM-2, PageRank, TF-IDF, and salience signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceConfig {
    /// Master enable flag. `false` = use legacy scoring.
    pub enabled: bool,
    /// Sub-signal weights within the importance composite.
    /// Auto-normalized at runtime; need not sum to 1.0.
    /// Set any weight to 0.0 to disable that algorithm.
    #[serde(default = "ImportanceWeights::default")]
    pub weights: ImportanceWeights,
    /// FSRS per-category initial stability tuning.
    #[serde(default)]
    pub fsrs: FsrsConfig,
    /// PageRank damping factor (typically 0.85).
    #[serde(default = "default_pagerank_damping")]
    pub pagerank_damping: f32,
    /// PageRank iteration count (typically 20).
    #[serde(default = "default_pagerank_iterations")]
    pub pagerank_iterations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceWeights {
    pub bayesian: f32,
    pub fsrs: f32,
    pub fsrs_difficulty: f32,
    pub pagerank: f32,
    pub tfidf: f32,
    pub salience: f32,
    pub sm2: f32,
}

impl Default for ImportanceWeights {
    fn default() -> Self {
        Self {
            bayesian: 0.35,
            fsrs: 0.15,
            fsrs_difficulty: 0.15,
            pagerank: 0.10,
            tfidf: 0.10,
            salience: 0.10,
            sm2: 0.05,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsrsConfig {
    /// Initial stability for episodic memories (days).
    #[serde(default = "default_episodic_stability")]
    pub episodic_initial_stability: f32,
    /// Initial stability for semantic memories (days).
    #[serde(default = "default_semantic_stability")]
    pub semantic_initial_stability: f32,
    /// Initial stability for procedural memories (days).
    #[serde(default = "default_procedural_stability")]
    pub procedural_initial_stability: f32,
    /// Initial difficulty for all new memories [0.0, 1.0].
    #[serde(default = "default_fsrs_difficulty")]
    pub initial_difficulty: f32,
}

impl Default for FsrsConfig {
    fn default() -> Self {
        Self {
            episodic_initial_stability: default_episodic_stability(),
            semantic_initial_stability: default_semantic_stability(),
            procedural_initial_stability: default_procedural_stability(),
            initial_difficulty: default_fsrs_difficulty(),
        }
    }
}

fn default_episodic_stability() -> f32 { 1.0 }
fn default_semantic_stability() -> f32 { 7.0 }
fn default_procedural_stability() -> f32 { 3.0 }
fn default_fsrs_difficulty() -> f32 { 0.3 }
fn default_pagerank_damping() -> f32 { 0.85 }
fn default_pagerank_iterations() -> usize { 20 }

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

    pub fn validate(&self) -> Result<()> {
        if self.embedding.dimensions == 0 {
            return Err(McpMemoryError::Config {
                message: "embedding.dimensions must be > 0".into(),
            });
        }
        if self.embedding.endpoint.is_empty() {
            return Err(McpMemoryError::Config {
                message: "embedding.endpoint must not be empty".into(),
            });
        }
        if let Some(ref key_env) = self.embedding.api_key_env {
            if key_env.is_empty() {
                return Err(McpMemoryError::Config {
                    message: "embedding.api_key_env must not be empty when specified".into(),
                });
            }
        }

        let w = &self.retrieval;
        let sum = w.vector_weight + w.graph_weight + w.recency_weight + w.importance_weight;
        if (sum - 1.0).abs() > 0.01 {
            return Err(McpMemoryError::Config {
                message: format!("retrieval weights must sum to 1.0, got {sum}"),
            });
        }
        if w.graph_hop_depth == 0 {
            return Err(McpMemoryError::Config {
                message: "retrieval.graph_hop_depth must be > 0".into(),
            });
        }

        if self.memory.consolidation_threshold == 0 {
            return Err(McpMemoryError::Config {
                message: "memory.consolidation_threshold must be > 0".into(),
            });
        }

        if self.memory.max_pinned_memories == 0 {
            return Err(McpMemoryError::Config {
                message: "memory.max_pinned_memories must be > 0".into(),
            });
        }

        if self.memory.max_pinned_content_length == 0 {
            return Err(McpMemoryError::Config {
                message: "memory.max_pinned_content_length must be > 0".into(),
            });
        }

        if matches!(self.database.mode, DatabaseMode::Remote)
            && self.database.remote_url.is_none()
        {
            return Err(McpMemoryError::Config {
                message: "database.remote_url required when mode is 'remote'".into(),
            });
        }

        if matches!(
            self.consolidation.mode,
            ConsolidationMode::Llm | ConsolidationMode::Auto
        ) && self.consolidation.chat_endpoint.is_none()
            && matches!(self.consolidation.mode, ConsolidationMode::Llm)
        {
            return Err(McpMemoryError::Config {
                message: "consolidation.chat_endpoint required when mode is 'llm'".into(),
            });
        }

        if self.consolidation.min_importance < 0.0 || self.consolidation.min_importance > 1.0 {
            return Err(McpMemoryError::Config {
                message: "consolidation.min_importance must be in [0.0, 1.0]".into(),
            });
        }
        if self.consolidation.temporal_half_life_days <= 0.0 {
            return Err(McpMemoryError::Config {
                message: "consolidation.temporal_half_life_days must be > 0".into(),
            });
        }

        if self.hooks.auto_importance < 0.0 || self.hooks.auto_importance > 1.0 {
            return Err(McpMemoryError::Config {
                message: "hooks.auto_importance must be in [0.0, 1.0]".into(),
            });
        }

        if let Some(ref imp) = self.importance {
            if imp.pagerank_damping < 0.0 || imp.pagerank_damping > 1.0 {
                return Err(McpMemoryError::Config {
                    message: "importance.pagerank_damping must be in [0.0, 1.0]".into(),
                });
            }
            if imp.pagerank_iterations == 0 {
                return Err(McpMemoryError::Config {
                    message: "importance.pagerank_iterations must be > 0".into(),
                });
            }
            if imp.fsrs.initial_difficulty < 0.0 || imp.fsrs.initial_difficulty > 1.0 {
                return Err(McpMemoryError::Config {
                    message: "importance.fsrs.initial_difficulty must be in [0.0, 1.0]".into(),
                });
            }
        }

        if let Some(ref rr) = self.reranker {
            if rr.endpoint.is_empty() {
                return Err(McpMemoryError::Config {
                    message: "reranker.endpoint must not be empty".into(),
                });
            }
            if rr.model.is_empty() {
                return Err(McpMemoryError::Config {
                    message: "reranker.model must not be empty".into(),
                });
            }
            if rr.reranker_weight < 0.0 || rr.reranker_weight > 1.0 {
                return Err(McpMemoryError::Config {
                    message: "reranker.reranker_weight must be in [0.0, 1.0]".into(),
                });
            }
            if rr.candidate_multiplier == 0 {
                return Err(McpMemoryError::Config {
                    message: "reranker.candidate_multiplier must be > 0".into(),
                });
            }
            if let Some(ref key_env) = rr.api_key_env {
                if key_env.is_empty() {
                    return Err(McpMemoryError::Config {
                        message: "reranker.api_key_env must not be empty when specified".into(),
                    });
                }
            }
        }

        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        Config {
            database: DatabaseConfig {
                mode: DatabaseMode::Embedded,
                data_dir: PathBuf::from("/tmp/lattice-mcp-test"),
                remote_url: None,
            },
            embedding: EmbeddingConfig {
                endpoint: "https://api.openai.com/v1/embeddings".into(),
                model: "text-embedding-3-small".into(),
                dimensions: 1536,
                api_key_env: Some("OPENAI_API_KEY".into()),
            },
            memory: MemoryConfig {
                episodic_decay_half_life_days: 30.0,
                semantic_decay_half_life_days: 365.0,
                procedural_decay_half_life_days: 180.0,
                consolidation_threshold: 10,
                max_retrieval_results: 20,
                max_pinned_memories: 10,
                max_pinned_content_length: 500,
            },
            retrieval: RetrievalConfig {
                vector_weight: 0.4,
                graph_weight: 0.25,
                recency_weight: 0.2,
                importance_weight: 0.15,
                graph_hop_depth: 2,
            },
            consolidation: ConsolidationConfig {
                mode: ConsolidationMode::Auto,
                chat_endpoint: None,
                chat_model: None,
                min_importance: 0.4,
                temporal_half_life_days: 7.0,
            },
            hooks: HooksConfig {
                mode: HooksMode::Hybrid,
                auto_capture_patterns: vec![
                    "Edit".into(),
                    "Write".into(),
                    "Bash".into(),
                ],
                auto_importance: 0.3,
            },
            reranker: None,
            importance: None,
        }
    }

    #[test]
    fn valid_config_passes() {
        assert!(test_config().validate().is_ok());
    }

    #[test]
    fn zero_dimensions_fails() {
        let mut cfg = test_config();
        cfg.embedding.dimensions = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn bad_weights_fails() {
        let mut cfg = test_config();
        cfg.retrieval.vector_weight = 0.5;
        // sum now 0.5 + 0.25 + 0.2 + 0.15 = 1.1
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn remote_without_url_fails() {
        let mut cfg = test_config();
        cfg.database.mode = DatabaseMode::Remote;
        cfg.database.remote_url = None;
        assert!(cfg.validate().is_err());
    }

    fn test_reranker_config() -> RerankerConfig {
        RerankerConfig {
            endpoint: "http://localhost:8081/v1/rerank".into(),
            model: "test-model".into(),
            api_format: RerankerApiFormat::Openai,
            api_key_env: None,
            reranker_weight: 0.6,
            candidate_multiplier: 3,
        }
    }

    #[test]
    fn reranker_config_validates() {
        let mut cfg = test_config();
        cfg.reranker = Some(test_reranker_config());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn reranker_bad_weight_fails() {
        let mut cfg = test_config();
        let mut rr = test_reranker_config();
        rr.reranker_weight = 1.5;
        cfg.reranker = Some(rr);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn reranker_zero_multiplier_fails() {
        let mut cfg = test_config();
        let mut rr = test_reranker_config();
        rr.candidate_multiplier = 0;
        cfg.reranker = Some(rr);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn bad_min_importance_fails() {
        let mut cfg = test_config();
        cfg.consolidation.min_importance = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn zero_temporal_half_life_fails() {
        let mut cfg = test_config();
        cfg.consolidation.temporal_half_life_days = 0.0;
        assert!(cfg.validate().is_err());
    }
}

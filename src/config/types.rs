//! Configuration sub-types (PCND — all fields explicit unless documented).

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Adaptive signal-fusion policy (Step 4). All fields explicit (PCND).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Master enable flag.
    pub enabled: bool,
    /// Fusion strategy.
    pub mode: FusionMode,
    /// Reciprocal Rank Fusion constant (typical 60.0). Only used by `Rrf`.
    pub rrf_k: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FusionMode {
    /// Per-memory weights that favor feedback-calibrated importance as
    /// Bayesian evidence accumulates (shipped default mode).
    ConfidenceWeighted,
    /// Rank-only Reciprocal Rank Fusion (experimental, benchmark-gated).
    Rrf,
}

/// Write-time contradiction policy (Step 9, experimental). All fields
/// explicit (PCND). The algorithmic path only flags (`CONTRADICTS`); an
/// irreversible `SUPERSEDES` requires LLM confirmation AND `auto_supersede`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContradictionConfig {
    pub enabled: bool,
    /// Only neighbors at/above this cosine similarity are checked [0.0, 1.0].
    pub min_similarity: f32,
    /// LLM-confirmed contradictions also add a `SUPERSEDES` edge.
    pub auto_supersede: bool,
}

/// Persistent user-avatar policy. All fields explicit (PCND).
///
/// Privacy: strict local-first. Validation FAILS FAST if `enabled` while the
/// database is remote — a persistent model of a real person stays on-device.
/// `redact_patterns` are user-declared literal substrings to mask (built-in
/// email/number redaction always applies on top).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvatarConfig {
    pub enabled: bool,
    /// Permit slots outside the curated vocabulary.
    pub allow_freeform: bool,
    /// Extra literal PII patterns to redact pre-write.
    #[serde(default)]
    pub redact_patterns: Vec<String>,
    /// Run explicit/LLM avatar slot derivation at session_end.
    pub auto_derive: bool,
    /// Escape hatch for the strict local-first guard. When `false` (default),
    /// enabling the avatar with a remote database fails validation — the user
    /// model stays on-device. Set `true` to deliberately persist the avatar to
    /// a trusted remote LatticeDB (e.g. a shared LAN server), accepting that
    /// the user model then leaves the device.
    #[serde(default)]
    pub allow_remote: bool,
}

/// Query-transformation policy (Step 10). All fields explicit (PCND).
/// `enabled` turns on the pre-embedding rewrite (LLM via
/// `ChatProvider::transform_query`, else conservative keyword stripping).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryTransformConfig {
    pub enabled: bool,
}

/// Multi-signal retrieval: blend a lexical (keyword-overlap) signal into the
/// retrieval score alongside the semantic/graph/recency/importance signals
/// (Mem0-style). All fields explicit (PCND). `lexical_weight` ∈ [0,1] is the
/// blend: `final = (1 - w) * hybrid + w * lexical_overlap`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSignalConfig {
    pub enabled: bool,
    pub lexical_weight: f32,
}

/// Result diversification policy (Step 6, MMR). All fields explicit (PCND).
/// `lambda` ∈ [0,1]: 1.0 = pure relevance (no diversification), lower values
/// trade relevance for novelty among the top results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversifyConfig {
    pub enabled: bool,
    pub lambda: f32,
}

/// Utility-driven memory lifecycle policy (Step 5). All fields explicit (PCND).
///
/// Keeps the top `budget_per_category` memories by blended utility; overflow
/// is demoted (importance halved — recoverable). `allow_delete` additionally
/// hard-deletes overflow below `delete_below` (destructive — default-safe
/// deployments keep it false; `content_id` is a hash so a re-`remember`
/// restores deleted content).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleConfig {
    /// Master enable flag.
    pub enabled: bool,
    /// Max memories kept per category before demotion/eviction kicks in.
    pub budget_per_category: usize,
    /// Hard-delete (not just demote) low-utility overflow.
    pub allow_delete: bool,
    /// Utility threshold under which overflow is evicted (only if `allow_delete`).
    pub delete_below: f32,
    /// Run a prune pass for every category at `session_end`.
    pub prune_on_session_end: bool,
}

/// Write-time novelty / near-duplicate policy (Step 3).
///
/// All fields explicit (PCND). When `enabled`, every `remember` is tagged
/// with a novelty score in `importance_state.novelty`. When additionally
/// `skip_near_dups`, a near-duplicate write is reversed and the existing
/// similar memory is returned instead (never silently dropped).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WritePolicyConfig {
    /// Master enable flag.
    pub enabled: bool,
    /// Cosine similarity at/above which a write is a near-duplicate [0.0, 1.0].
    pub near_dup_threshold: f32,
    /// If true, reverse near-duplicate writes and return the existing memory.
    /// Default-style behavior is tag-only; set true to actively dedup.
    pub skip_near_dups: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// "embedded" or "remote"
    pub mode: DatabaseMode,
    /// Data directory for embedded mode (e.g. ~/.lattice-mcp/data)
    pub data_dir: PathBuf,
    /// URL for remote mode (e.g. http://192.0.2.10:6334)
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
    /// Step 7: structured 4-field distillation + provenance edges. `Some(true)`
    /// enables it (LLM via `ChatProvider::distill`, else algorithmic). Absent
    /// or `Some(false)` keeps the plain single-string summary (today's path).
    #[serde(default)]
    pub distill_enabled: Option<bool>,
    /// Step 8: `Some(true)` runs `reflect` automatically at `session_end`
    /// once `memory.consolidation_threshold` new memories have accumulated.
    /// Absent or `Some(false)` keeps consolidation fully manual (today's path).
    #[serde(default)]
    pub auto_consolidate: Option<bool>,
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
pub struct RerankerConfig {
    /// Reranker HTTP endpoint (e.g. "http://192.0.2.10:8081/v1/rerank")
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

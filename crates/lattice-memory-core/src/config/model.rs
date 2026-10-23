//! Configuration for the pure memory model (PCND — every field explicit).
//! The host's TOML `Config` embeds these; they are defined here so the
//! model and its config have one source.

use serde::{Deserialize, Serialize};

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

/// Retrieval-time retraction filter. All fields explicit (PCND). When
/// `enabled`, recall drops memories marked superseded, and a confirmed
/// `SUPERSEDES` also marks the superseded memory (`superseded_by`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalFilterConfig {
    pub enabled: bool,
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

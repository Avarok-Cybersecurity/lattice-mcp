//! Adaptive-importance configuration sub-types.

use serde::{Deserialize, Serialize};

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

fn default_episodic_stability() -> f32 {
    1.0
}
fn default_semantic_stability() -> f32 {
    7.0
}
fn default_procedural_stability() -> f32 {
    3.0
}
fn default_fsrs_difficulty() -> f32 {
    0.3
}
fn default_pagerank_damping() -> f32 {
    0.85
}
fn default_pagerank_iterations() -> usize {
    20
}

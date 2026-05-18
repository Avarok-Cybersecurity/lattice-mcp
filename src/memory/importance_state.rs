//! Per-memory adaptive importance state (SSOT for all importance algorithms).
//!
//! Stored as a single payload field on each Memory. Old memories without
//! this field deserialize to `Default::default()` via `#[serde(default)]`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Adaptive importance state tracked per memory.
///
/// Each field maps to one of the 7 importance algorithms:
/// - Bayesian (alpha/beta): evidence-based confidence
/// - FSRS (stability/difficulty): calibrated forgetting curve
/// - SM-2 (easiness/interval/repetitions): spaced repetition scheduling
/// - TF-IDF (uniqueness): corpus rarity at write time
/// - Salience: prediction error / novelty at write time
/// - PageRank: graph-structural importance (periodic)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ImportanceState {
    /// Bayesian Beta-Binomial: success count (relevant recalls).
    pub alpha: f32,
    /// Bayesian Beta-Binomial: failure count (irrelevant recalls).
    pub beta: f32,

    /// FSRS: per-memory stability in days. Higher = slower forgetting.
    pub fsrs_stability: f32,
    /// FSRS: per-memory difficulty in [0.0, 1.0]. Higher = harder to retain.
    pub fsrs_difficulty: f32,

    /// SM-2: easiness factor. Starts at 2.5, floor at 1.3.
    pub sm2_easiness: f32,
    /// SM-2: current review interval in days.
    pub sm2_interval: f32,
    /// SM-2: count of consecutive successful recalls.
    pub sm2_repetitions: u32,

    /// TF-IDF uniqueness: corpus rarity computed at write time [0.0, 1.0].
    pub tfidf_uniqueness: f32,
    /// Prediction error / salience: novelty computed at write time [0.0, 1.0].
    pub salience: f32,
    /// Cached PageRank score from periodic graph analysis [0.0, 1.0].
    pub pagerank: f32,

    /// Timestamp of last feedback event (for FSRS/SM-2 interval calculations).
    pub last_review: Option<DateTime<Utc>>,
}

impl Default for ImportanceState {
    fn default() -> Self {
        Self {
            alpha: 1.0,           // Beta(1,1) = uniform prior ("we know nothing")
            beta: 1.0,            // Beta(1,1) = uniform prior
            fsrs_stability: 1.0,  // 1 day initial stability
            fsrs_difficulty: 0.3, // moderate difficulty
            sm2_easiness: 2.5,    // SM-2 standard initial EF
            sm2_interval: 1.0,    // 1 day initial interval
            sm2_repetitions: 0,   // no successful recalls yet
            tfidf_uniqueness: 0.5, // neutral until computed
            salience: 0.5,         // neutral until computed
            pagerank: 0.0,         // unranked until first PageRank pass
            last_review: None,     // no feedback yet
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let state = ImportanceState::default();
        assert_eq!(state.alpha, 1.0);
        assert_eq!(state.beta, 1.0);
        assert_eq!(state.fsrs_stability, 1.0);
        assert_eq!(state.fsrs_difficulty, 0.3);
        assert_eq!(state.sm2_easiness, 2.5);
        assert_eq!(state.sm2_interval, 1.0);
        assert_eq!(state.sm2_repetitions, 0);
        assert_eq!(state.tfidf_uniqueness, 0.5);
        assert_eq!(state.salience, 0.5);
        assert_eq!(state.pagerank, 0.0);
        assert!(state.last_review.is_none());
    }

    #[test]
    fn serde_roundtrip() {
        let state = ImportanceState {
            alpha: 5.0,
            beta: 2.0,
            last_review: Some(Utc::now()),
            ..Default::default()
        };
        let json = serde_json::to_vec(&state).unwrap();
        let restored: ImportanceState = serde_json::from_slice(&json).unwrap();
        assert_eq!(restored.alpha, 5.0);
        assert_eq!(restored.beta, 2.0);
        assert!(restored.last_review.is_some());
    }

    #[test]
    fn deserialize_empty_object_uses_defaults() {
        // Simulates old memories that have no importance_state at all.
        // The #[serde(default)] on Memory will call ImportanceState::default().
        let state: ImportanceState = serde_json::from_str("{}").unwrap();
        assert_eq!(state.alpha, 1.0);
        assert_eq!(state.beta, 1.0);
    }
}

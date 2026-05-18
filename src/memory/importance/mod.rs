//! Composite importance scoring — aggregates all 7 importance algorithms.
//!
//! Pure functions, no I/O (SBIO). Each sub-algorithm lives in its own module.

pub mod bayesian;
pub mod fsrs;
pub mod pagerank;
pub mod sm2;
pub mod write_time;

use chrono::{DateTime, Utc};

use super::importance_state::ImportanceState;
use crate::config::ImportanceWeights;

/// Compose all importance signals into a single [0.0, 1.0] score.
///
/// Uses confidence-weighted blending: as feedback evidence accumulates,
/// feedback-responsive signals (Bayesian, FSRS stability, FSRS difficulty, SM-2)
/// dominate over static write-time signals (PageRank, TF-IDF, Salience).
///
/// The Bayesian signal includes an exploration bonus (Feel-Good Thompson Sampling)
/// that surfaces uncertain memories. The bonus fades as evidence accumulates.
///
/// At 0 reviews confidence=0.5 (equal blend); at 20 reviews confidence≈0.98.
/// Weights control relative strength *within* each group.
pub fn composite_importance(
    state: &ImportanceState,
    _now: DateTime<Utc>,
    weights: &ImportanceWeights,
) -> f32 {
    // Pure posterior mean for importance scoring. Exploration bonus
    // (bayesian::posterior_with_exploration) is applied at retrieval time,
    // not here — importance measures "how valuable", not "what to explore".
    let bayesian = bayesian::posterior_mean(state.alpha, state.beta);

    let fsrs_signal = fsrs_normalized_stability(state.fsrs_stability);

    // Inverted difficulty: low difficulty (easy to retain) = high importance.
    // Good memories → difficulty≈0.0 → signal≈1.0.
    // Bad memories → difficulty≈0.9 → signal≈0.1.
    let fsrs_ease = 1.0 - state.fsrs_difficulty;

    let sm2_mastery = sm2::mastery(state.sm2_interval, state.sm2_easiness);

    let feedback_weight =
        weights.bayesian + weights.fsrs + weights.fsrs_difficulty + weights.sm2;
    let static_weight = weights.pagerank + weights.tfidf + weights.salience;

    let feedback_score = if feedback_weight > 0.0 {
        (weights.bayesian * bayesian
            + weights.fsrs * fsrs_signal
            + weights.fsrs_difficulty * fsrs_ease
            + weights.sm2 * sm2_mastery)
            / feedback_weight
    } else {
        0.0
    };

    let static_score = if static_weight > 0.0 {
        (weights.pagerank * state.pagerank
            + weights.tfidf * state.tfidf_uniqueness
            + weights.salience * state.salience)
            / static_weight
    } else {
        0.0
    };

    let confidence = evidence_confidence(state.alpha, state.beta);

    let composite = if feedback_weight > 0.0 && static_weight > 0.0 {
        confidence * feedback_score + (1.0 - confidence) * static_score
    } else if feedback_weight > 0.0 {
        feedback_score
    } else {
        static_score
    };

    composite.clamp(0.0, 1.0)
}

/// Map accumulated Bayesian evidence to a [0.5, 1.0) confidence in feedback signals.
///
/// `review_count = alpha + beta - 2` strips the Beta(1,1) uniform prior.
/// At 0 reviews → 0.5 (equal trust in feedback vs static).
/// At 20 reviews → ~0.976 (feedback dominates).
/// Half-saturation at ~1 review keeps the curve responsive from the first feedback.
fn evidence_confidence(alpha: f32, beta: f32) -> f32 {
    let review_count = (alpha + beta - 2.0).max(0.0);
    0.5 + 0.5 * (1.0 - 1.0 / (1.0 + review_count))
}

/// Normalize FSRS stability into a [0, 1) importance signal.
///
/// Uses a half-saturation curve: `stability / (stability + k)`.
/// With k=3.0: stability=7 → 0.700, stability=5 → 0.625, stability=0.1 → 0.032.
/// K=3 chosen to widen the gap between good (stability≈7) and bad (stability≈0.1)
/// compared to K=5 (which capped good at 0.583).
fn fsrs_normalized_stability(stability: f32) -> f32 {
    const K: f32 = 3.0;
    stability.max(0.0) / (stability.max(0.0) + K)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ImportanceWeights;

    fn default_weights() -> ImportanceWeights {
        ImportanceWeights::default()
    }

    #[test]
    fn default_state_gives_moderate_score() {
        let state = ImportanceState::default();
        let now = Utc::now();
        let score = composite_importance(&state, now, &default_weights());
        // 0 reviews → confidence=0.5, blend of feedback (~0.33) and static (~0.33)
        assert!(score > 0.2 && score < 0.8, "score={score}");
    }

    #[test]
    fn zero_weights_gives_zero() {
        let state = ImportanceState::default();
        let weights = ImportanceWeights {
            bayesian: 0.0,
            fsrs: 0.0,
            fsrs_difficulty: 0.0,
            pagerank: 0.0,
            tfidf: 0.0,
            salience: 0.0,
            sm2: 0.0,
        };
        let score = composite_importance(&state, Utc::now(), &weights);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn high_bayesian_raises_score() {
        let mut state = ImportanceState::default();
        state.alpha = 10.0; // strong positive evidence → high confidence + high feedback
        state.beta = 1.0;
        let score = composite_importance(&state, Utc::now(), &default_weights());
        assert!(score > 0.5, "high bayesian should raise score, got {score}");
    }

    #[test]
    fn score_clamped_to_unit() {
        let mut state = ImportanceState::default();
        state.alpha = 100.0;
        state.pagerank = 1.0;
        state.tfidf_uniqueness = 1.0;
        state.salience = 1.0;
        let score = composite_importance(&state, Utc::now(), &default_weights());
        assert!(score <= 1.0);
    }

    #[test]
    fn evidence_confidence_at_zero_reviews() {
        // Beta(1,1) prior → 0 reviews → confidence = 0.5
        let c = evidence_confidence(1.0, 1.0);
        assert!((c - 0.5).abs() < 0.001, "0 reviews → confidence=0.5, got {c}");
    }

    #[test]
    fn evidence_confidence_grows_with_feedback() {
        let c0 = evidence_confidence(1.0, 1.0);   // 0 reviews
        let c5 = evidence_confidence(4.0, 3.0);    // 5 reviews
        let c20 = evidence_confidence(15.0, 7.0);  // 20 reviews
        assert!(c0 < c5, "confidence should grow: {c0} vs {c5}");
        assert!(c5 < c20, "confidence should grow: {c5} vs {c20}");
        assert!(c20 > 0.95, "20 reviews → high confidence, got {c20}");
    }

    #[test]
    fn feedback_dominates_after_many_reviews() {
        // After many positive reviews, static signals should barely matter
        let mut state = ImportanceState::default();
        state.alpha = 20.0;
        state.beta = 2.0;
        state.fsrs_stability = 7.0;
        state.pagerank = 0.0;
        state.tfidf_uniqueness = 0.0;
        state.salience = 0.0;

        let score = composite_importance(&state, Utc::now(), &default_weights());
        // confidence ≈ 0.976, feedback_score high, static_score ≈ 0
        // Score should still be high despite zero static signals
        assert!(score > 0.5, "feedback should dominate after 20 reviews, got {score}");
    }

    #[test]
    fn feedback_only_weights_work() {
        let state = ImportanceState::default();
        let weights = ImportanceWeights {
            bayesian: 0.5,
            fsrs: 0.2,
            fsrs_difficulty: 0.1,
            sm2: 0.2,
            pagerank: 0.0,
            tfidf: 0.0,
            salience: 0.0,
        };
        let score = composite_importance(&state, Utc::now(), &weights);
        assert!(score > 0.0, "feedback-only weights should produce score, got {score}");
    }

    #[test]
    fn static_only_weights_work() {
        let mut state = ImportanceState::default();
        state.pagerank = 0.8;
        let weights = ImportanceWeights {
            bayesian: 0.0,
            fsrs: 0.0,
            fsrs_difficulty: 0.0,
            sm2: 0.0,
            pagerank: 0.5,
            tfidf: 0.3,
            salience: 0.2,
        };
        let score = composite_importance(&state, Utc::now(), &weights);
        assert!(score > 0.0, "static-only weights should produce score, got {score}");
    }

    #[test]
    fn fsrs_difficulty_improves_separation() {
        let mut good = ImportanceState::default();
        good.fsrs_difficulty = 0.05; // easy to retain
        let mut bad = ImportanceState::default();
        bad.fsrs_difficulty = 0.90; // hard to retain

        let good_score = composite_importance(&good, Utc::now(), &default_weights());
        let bad_score = composite_importance(&bad, Utc::now(), &default_weights());
        assert!(
            good_score > bad_score,
            "low difficulty should score higher: {good_score} vs {bad_score}"
        );
    }
}

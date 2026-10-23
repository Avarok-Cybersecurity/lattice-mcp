//! Adaptive fusion of the four HybridRAG signals (pure, no I/O — SBIO).
//!
//! The default pipeline uses fixed config weights (`hybrid_score`). When
//! `fusion` is enabled this re-derives `final_score`:
//!
//! - **ConfidenceWeighted** (shipped): per-memory weights that reallocate
//!   vector-similarity mass onto the feedback-calibrated importance signal
//!   as Bayesian evidence accumulates. With no `rate_recall` feedback a
//!   memory keeps the exact base weights, so behavior is continuous.
//! - **Rrf** (experimental, benchmark-gated): rank-only Reciprocal Rank
//!   Fusion across the four signals. Discards magnitudes; degenerate on the
//!   small per-category candidate sets typical here — kept opt-in for
//!   evaluation, not recommended as a default.

use crate::config::{FusionConfig, FusionMode, RetrievalConfig};
use crate::memory::importance::evidence_confidence;
use crate::memory::types::ScoredMemory;

/// Per-memory signal weights `[vector, graph, recency, importance]`.
///
/// `confidence` ∈ [0.5, 1.0) is the Bayesian evidence confidence. At 0.5
/// (no feedback) the base config weights are returned unchanged. As it
/// rises, up to half of the vector weight is reallocated onto importance
/// (feedback-calibrated importance is more trustworthy than raw similarity
/// once evidence exists). Mass-neutral: the sum is preserved.
pub fn confidence_weights(confidence: f32, cfg: &RetrievalConfig) -> [f32; 4] {
    let base = [
        cfg.vector_weight,
        cfg.graph_weight,
        cfg.recency_weight,
        cfg.importance_weight,
    ];
    let shift = ((confidence - 0.5) * 2.0).clamp(0.0, 1.0);
    let realloc = 0.5 * base[0] * shift;
    [base[0] - realloc, base[1], base[2], base[3] + realloc]
}

/// Confidence-weighted fused score for one memory.
fn confidence_fused(state_conf: f32, cfg: &RetrievalConfig, s: &ScoredMemory) -> f32 {
    let w = confidence_weights(state_conf, cfg);
    w[0] * s.vector_similarity
        + w[1] * s.graph_relevance
        + w[2] * s.temporal_recency
        + w[3] * s.importance_score
}

/// Rank-only Reciprocal Rank Fusion across the four signals.
/// Returns fused scores aligned to `items` input order.
pub fn reciprocal_rank_fusion(items: &[[f32; 4]], k: f32) -> Vec<f32> {
    let n = items.len();
    let mut scores = vec![0.0_f32; n];
    let extractors: [fn(&[f32; 4]) -> f32; 4] = [|a| a[0], |a| a[1], |a| a[2], |a| a[3]];
    for get in extractors {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            get(&items[b])
                .partial_cmp(&get(&items[a]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (rank, &idx) in order.iter().enumerate() {
            scores[idx] += 1.0 / (k + rank as f32 + 1.0);
        }
    }
    scores
}

/// Re-derive `final_score` for every candidate per the fusion config.
/// Operates in place on the already-built candidate set (no I/O).
pub fn apply(fc: &FusionConfig, cfg: &RetrievalConfig, scored: &mut [ScoredMemory]) {
    match fc.mode {
        FusionMode::ConfidenceWeighted => {
            for s in scored.iter_mut() {
                let c = evidence_confidence(
                    s.memory.importance_state.alpha,
                    s.memory.importance_state.beta,
                );
                s.final_score = confidence_fused(c, cfg, s);
            }
        }
        FusionMode::Rrf => {
            let items: Vec<[f32; 4]> = scored
                .iter()
                .map(|s| {
                    [
                        s.vector_similarity,
                        s.graph_relevance,
                        s.temporal_recency,
                        s.importance_score,
                    ]
                })
                .collect();
            let fused = reciprocal_rank_fusion(&items, fc.rrf_k);
            for (s, f) in scored.iter_mut().zip(fused) {
                s.final_score = f;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RetrievalConfig;

    fn cfg() -> RetrievalConfig {
        RetrievalConfig {
            vector_weight: 0.4,
            graph_weight: 0.25,
            recency_weight: 0.2,
            importance_weight: 0.15,
            graph_hop_depth: 2,
        }
    }

    #[test]
    fn no_feedback_preserves_base_weights() {
        let w = confidence_weights(0.5, &cfg());
        assert!((w[0] - 0.4).abs() < 1e-6);
        assert!((w[3] - 0.15).abs() < 1e-6);
    }

    #[test]
    fn weights_are_mass_neutral() {
        for c in [0.5, 0.7, 0.9, 0.99] {
            let w = confidence_weights(c, &cfg());
            let sum: f32 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "sum={sum} at c={c}");
        }
    }

    #[test]
    fn evidence_shifts_mass_vector_to_importance() {
        let w = confidence_weights(0.95, &cfg());
        assert!(w[0] < 0.4, "vector weight should drop, got {}", w[0]);
        assert!(w[3] > 0.15, "importance weight should rise, got {}", w[3]);
        // graph & recency untouched
        assert!((w[1] - 0.25).abs() < 1e-6);
        assert!((w[2] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn rrf_rewards_consistently_top_ranked_item() {
        // item 0 leads every signal ⇒ highest RRF score.
        let items = [
            [0.9, 0.9, 0.9, 0.9],
            [0.5, 0.4, 0.3, 0.2],
            [0.1, 0.1, 0.1, 0.1],
        ];
        let s = reciprocal_rank_fusion(&items, 60.0);
        assert!(s[0] > s[1] && s[1] > s[2], "scores={s:?}");
    }
}

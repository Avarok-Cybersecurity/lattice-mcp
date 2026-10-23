//! Write-time surprise / novelty signals (pure, no I/O — SBIO).
//!
//! Reuses the neighbor similarity set already produced by `find_similar`
//! (zero extra I/O). Inspired by reward-prediction-error / novelty-gated
//! memory writing: a write that closely matches an existing memory carries
//! little new information and can be tagged or skipped.

/// Novelty of a candidate memory given its neighbor similarities.
///
/// `1.0` = no similar neighbor (fully novel); `0.0` = an identical neighbor
/// already exists. With no neighbors the memory is maximally novel.
/// Similarities are cosine-derived in `[0, 1]`; result clamped to `[0, 1]`.
pub fn novelty(neighbor_sims: &[f32]) -> f32 {
    let max_sim = neighbor_sims.iter().copied().fold(0.0_f32, f32::max);
    (1.0 - max_sim).clamp(0.0, 1.0)
}

/// Whether a write is a near-duplicate: some neighbor's similarity is at or
/// above `near_dup_threshold` (equivalently, novelty ≤ 1 − threshold).
pub fn is_near_dup(novelty: f32, near_dup_threshold: f32) -> bool {
    novelty <= 1.0 - near_dup_threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_neighbors_is_maximally_novel() {
        assert_eq!(novelty(&[]), 1.0);
    }

    #[test]
    fn identical_neighbor_is_zero_novelty() {
        assert!((novelty(&[0.2, 1.0, 0.5]) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn novelty_uses_max_similarity() {
        // max sim 0.7 ⇒ novelty 0.3
        assert!((novelty(&[0.1, 0.7, 0.4]) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn novelty_is_clamped() {
        // Defensive: a >1 similarity must not produce negative novelty.
        assert_eq!(novelty(&[1.5]), 0.0);
        assert_eq!(novelty(&[-0.3]), 1.0);
    }

    #[test]
    fn near_dup_detection_respects_threshold() {
        // threshold 0.9 ⇒ near-dup when novelty ≤ 0.1 (sim ≥ 0.9)
        assert!(is_near_dup(novelty(&[0.95]), 0.9));
        assert!(!is_near_dup(novelty(&[0.8]), 0.9));
        // exact boundary counts as near-dup
        assert!(is_near_dup(0.1, 0.9));
    }
}

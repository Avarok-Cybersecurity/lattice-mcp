//! Write-time importance signals: TF-IDF uniqueness and salience.
//!
//! Computed once when a memory is stored. Both use the vector search
//! results that `find_similar()` already produces, so no extra
//! I/O is needed (SBIO).
//!
//! - **TF-IDF uniqueness**: inverse of max similarity to existing memories.
//!   High = novel content, low = redundant/duplicate.
//! - **Salience (prediction error)**: how surprising the new memory is
//!   relative to the nearest existing memories. High = unexpected content.

/// Compute corpus uniqueness from nearest-neighbor similarity scores.
///
/// `nearest_scores`: cosine similarities of the k-nearest existing memories.
/// Returns [0.0, 1.0] where 1.0 = maximally unique (nothing similar exists).
///
/// Formula: `1.0 - max(nearest_scores)`.
/// Empty corpus → 1.0 (everything is novel when there's nothing to compare).
pub fn tfidf_uniqueness(nearest_scores: &[f32]) -> f32 {
    if nearest_scores.is_empty() {
        return 1.0;
    }
    let max_sim = nearest_scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    (1.0 - max_sim).clamp(0.0, 1.0)
}

/// Compute salience (prediction error / novelty) from nearest distances.
///
/// `nearest_scores`: cosine similarities of the k-nearest existing memories.
/// Returns [0.0, 1.0] where 1.0 = maximally surprising.
///
/// Uses mean distance (1 - similarity) of the top-k neighbors.
/// Higher mean distance = content is far from everything known = high salience.
/// Empty corpus → 1.0 (first memory is maximally novel).
pub fn salience(nearest_scores: &[f32]) -> f32 {
    if nearest_scores.is_empty() {
        return 1.0;
    }
    let sum_dist: f32 = nearest_scores.iter().map(|&s| 1.0 - s).sum();
    let mean_dist = sum_dist / nearest_scores.len() as f32;
    mean_dist.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_corpus_max_uniqueness() {
        assert_eq!(tfidf_uniqueness(&[]), 1.0);
    }

    #[test]
    fn duplicate_content_low_uniqueness() {
        // Very high similarity to existing memory
        let scores = vec![0.98, 0.85, 0.72];
        let u = tfidf_uniqueness(&scores);
        assert!(
            u < 0.1,
            "near-duplicate should have low uniqueness, got {u}"
        );
    }

    #[test]
    fn novel_content_high_uniqueness() {
        // Low similarity to all existing memories
        let scores = vec![0.15, 0.10, 0.08];
        let u = tfidf_uniqueness(&scores);
        assert!(
            u > 0.8,
            "novel content should have high uniqueness, got {u}"
        );
    }

    #[test]
    fn uniqueness_clamped() {
        // Edge case: negative similarity (shouldn't happen but safety)
        let scores = vec![-0.1];
        let u = tfidf_uniqueness(&scores);
        assert!((0.0..=1.0).contains(&u));
    }

    #[test]
    fn empty_corpus_max_salience() {
        assert_eq!(salience(&[]), 1.0);
    }

    #[test]
    fn similar_neighbors_low_salience() {
        // Very close neighbors → content is expected
        let scores = vec![0.95, 0.92, 0.90];
        let s = salience(&scores);
        assert!(
            s < 0.15,
            "similar neighbors should give low salience, got {s}"
        );
    }

    #[test]
    fn distant_neighbors_high_salience() {
        // Far neighbors → content is surprising
        let scores = vec![0.1, 0.15, 0.2];
        let s = salience(&scores);
        assert!(
            s > 0.8,
            "distant neighbors should give high salience, got {s}"
        );
    }

    #[test]
    fn salience_uses_mean_not_max() {
        // One close, one far — salience should be moderate
        let scores = vec![0.9, 0.1];
        let s = salience(&scores);
        // mean distance = (0.1 + 0.9) / 2 = 0.5
        assert!((s - 0.5).abs() < 0.01, "salience should be ~0.5, got {s}");
    }

    #[test]
    fn salience_clamped() {
        let scores = vec![1.5]; // shouldn't happen but safety
        let s = salience(&scores);
        assert!((0.0..=1.0).contains(&s));
    }
}

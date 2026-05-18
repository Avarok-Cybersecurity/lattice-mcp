//! Pure retrieval functions — HybridRAG scoring, ranking, deduplication.
//! No I/O, no side effects.

use crate::config::RetrievalConfig;
use crate::memory::types::ScoredMemory;

/// HybridRAG scoring: combine vector similarity, graph relevance,
/// recency, and importance into a single ranked score.
///
/// `final = α·vector_sim + β·graph_rel + γ·recency + δ·importance`
pub fn hybrid_score(
    vector_similarity: f32,
    graph_relevance: f32,
    temporal_recency: f32,
    importance: f32,
    weights: &RetrievalConfig,
) -> f32 {
    weights.vector_weight * vector_similarity
        + weights.graph_weight * graph_relevance
        + weights.recency_weight * temporal_recency
        + weights.importance_weight * importance
}

/// Compute graph relevance from traversal results.
///
/// Closer neighbors contribute more via inverse-depth weighting: `1/(depth+1)`.
/// Returns [0.0, 1.0].
pub fn graph_relevance_from_traversal(
    neighbor_ids: &[(u64, usize)], // (target_id, depth)
    result_ids: &[u64],            // IDs from vector search hits
) -> f32 {
    if neighbor_ids.is_empty() || result_ids.is_empty() {
        return 0.0;
    }

    let mut total = 0.0_f32;
    let mut count = 0_u32;

    for (id, depth) in neighbor_ids {
        if result_ids.contains(id) {
            total += 1.0 / (*depth as f32 + 1.0);
            count += 1;
        }
    }

    if count > 0 {
        (total / count as f32).min(1.0)
    } else {
        0.0
    }
}

/// Blend hybrid score with reranker score.
///
/// `blended = λ·reranker_score + (1-λ)·hybrid_score`
pub fn blend_reranker_score(hybrid_score: f32, reranker_score: f32, reranker_weight: f32) -> f32 {
    reranker_weight * reranker_score + (1.0 - reranker_weight) * hybrid_score
}

/// Rank scored memories by final_score descending, deduplicate by ID, truncate to limit.
pub fn rank_and_dedup(mut scored: Vec<ScoredMemory>, limit: usize) -> Vec<ScoredMemory> {
    // Sort descending by score
    scored.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Deduplicate by memory ID (keep highest score)
    let mut seen = std::collections::HashSet::new();
    scored.retain(|m| seen.insert(m.memory.id));

    scored.truncate(limit);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_weights() -> RetrievalConfig {
        RetrievalConfig {
            vector_weight: 0.4,
            graph_weight: 0.25,
            recency_weight: 0.2,
            importance_weight: 0.15,
            graph_hop_depth: 2,
        }
    }

    #[test]
    fn hybrid_score_perfect() {
        let score = hybrid_score(1.0, 1.0, 1.0, 1.0, &test_weights());
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn hybrid_score_zero() {
        let score = hybrid_score(0.0, 0.0, 0.0, 0.0, &test_weights());
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn hybrid_score_vector_dominant() {
        let score = hybrid_score(1.0, 0.0, 0.0, 0.0, &test_weights());
        assert!((score - 0.4).abs() < 0.001);
    }

    #[test]
    fn graph_relevance_empty() {
        let score = graph_relevance_from_traversal(&[], &[1, 2, 3]);
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn graph_relevance_direct_neighbor() {
        // ID 2 is at depth 1, and 2 is in result_ids
        let neighbors = vec![(2, 1)];
        let result_ids = vec![1, 2, 3];
        let score = graph_relevance_from_traversal(&neighbors, &result_ids);
        // 1/(1+1) = 0.5
        assert!((score - 0.5).abs() < 0.001);
    }

    #[test]
    fn graph_relevance_no_overlap() {
        let neighbors = vec![(10, 1), (11, 2)];
        let result_ids = vec![1, 2, 3];
        let score = graph_relevance_from_traversal(&neighbors, &result_ids);
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn blend_reranker_full_weight() {
        let blended = blend_reranker_score(0.5, 0.9, 1.0);
        assert!((blended - 0.9).abs() < 0.001);
    }

    #[test]
    fn blend_reranker_zero_weight() {
        let blended = blend_reranker_score(0.5, 0.9, 0.0);
        assert!((blended - 0.5).abs() < 0.001);
    }

    #[test]
    fn blend_reranker_half_weight() {
        let blended = blend_reranker_score(0.4, 0.8, 0.5);
        assert!((blended - 0.6).abs() < 0.001);
    }

    #[test]
    fn rank_and_dedup_orders_correctly() {
        use crate::memory::types::{Memory, MemoryCategory, MemoryMetadata};
        use chrono::Utc;

        let make = |id: u64, score: f32| ScoredMemory {
            memory: Memory {
                id,
                category: MemoryCategory::Episodic,
                content: format!("mem-{id}"),
                created_at: Utc::now(),
                last_accessed: Utc::now(),
                access_count: 0,
                importance: 0.5,
                session_id: None,
                metadata: MemoryMetadata::Episodic {
                    event_type: "test".into(),
                },
                importance_state: Default::default(),
            },
            vector_similarity: score,
            graph_relevance: 0.0,
            temporal_recency: 0.0,
            importance_score: 0.0,
            final_score: score,
            related: vec![],
        };

        let scored = vec![make(1, 0.3), make(2, 0.9), make(3, 0.6), make(2, 0.8)];
        let result = rank_and_dedup(scored, 3);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].memory.id, 2); // highest score
        assert_eq!(result[1].memory.id, 3);
        assert_eq!(result[2].memory.id, 1);
    }
}

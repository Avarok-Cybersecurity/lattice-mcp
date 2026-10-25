//! Retrieval-time retraction filter: a retracted or superseded memory is not
//! returned unless the caller asks for it (2609.08258 — revoked memories
//! otherwise keep steering answers). Filtering happens before truncation, so
//! a filtered hit's place goes to the next candidate. lattice-mcp's recall
//! calls it when `[retrieval_filter]` is enabled.

use alloc::collections::BTreeMap;

use crate::prelude::*;
use crate::types::ScoredMemory;

/// A returned hit, tagged with the memory that retracted it, if any.
#[derive(Debug, Clone)]
pub struct RetrievalHit {
    pub scored: ScoredMemory,
    pub retracted_by: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct FilteredHits {
    pub hits: Vec<RetrievalHit>,
    /// Retracted candidates dropped (0 when `include_retracted`).
    pub filtered_retracted: usize,
}

/// Take up to `limit` hits from `ranked` (best first). `retracted_by` maps a
/// retracted memory id to the id that retracted it. With
/// `include_retracted = false` those memories are dropped and counted; with
/// `true` they are kept in rank order and tagged.
pub fn filter_retracted(
    ranked: Vec<ScoredMemory>,
    retracted_by: &BTreeMap<u64, u64>,
    include_retracted: bool,
    limit: usize,
) -> FilteredHits {
    let mut filtered_retracted = 0;
    let mut hits = Vec::new();
    for scored in ranked {
        let by = retracted_by.get(&scored.memory.id).copied();
        if by.is_some() && !include_retracted {
            filtered_retracted += 1;
            continue;
        }
        if hits.len() < limit {
            hits.push(RetrievalHit {
                scored,
                retracted_by: by,
            });
        }
    }
    FilteredHits {
        hits,
        filtered_retracted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Memory, MemoryCategory, MemoryMetadata};

    fn hit(id: u64, score: f32) -> ScoredMemory {
        let t = chrono::DateTime::from_timestamp(1_780_000_000, 0).expect("valid timestamp");
        ScoredMemory {
            memory: Memory {
                id,
                category: MemoryCategory::Semantic,
                content: format!("fact {id}"),
                created_at: t,
                last_accessed: t,
                access_count: 0,
                importance: 0.5,
                session_id: None,
                metadata: MemoryMetadata::Semantic {
                    entity_type: "fact".into(),
                    confidence: 0.9,
                    provenance: vec![],
                    superseded_by: None,
                },
                importance_state: Default::default(),
            },
            vector_similarity: score,
            graph_relevance: 0.0,
            temporal_recency: 0.0,
            importance_score: 0.0,
            final_score: score,
            related: vec![],
        }
    }

    fn ids(f: &FilteredHits) -> Vec<u64> {
        f.hits.iter().map(|h| h.scored.memory.id).collect()
    }

    fn ranked() -> Vec<ScoredMemory> {
        vec![hit(1, 0.9), hit(2, 0.8), hit(3, 0.7), hit(4, 0.6)]
    }

    #[test]
    fn retracting_the_top_hit_removes_it_and_promotes_the_next() {
        let r = BTreeMap::from([(1, 40)]);
        let f = filter_retracted(ranked(), &r, false, 2);
        assert_eq!(ids(&f), vec![2, 3]);
        assert_eq!(f.filtered_retracted, 1);
        assert!(f.hits.iter().all(|h| h.retracted_by.is_none()));
    }

    #[test]
    fn asking_for_retracted_returns_them_tagged_in_rank_order() {
        let r = BTreeMap::from([(1, 40)]);
        let f = filter_retracted(ranked(), &r, true, 2);
        assert_eq!(ids(&f), vec![1, 2]);
        assert_eq!(f.hits[0].retracted_by, Some(40));
        assert_eq!(f.filtered_retracted, 0);
    }

    #[test]
    fn nothing_retracted_is_plain_truncation() {
        let f = filter_retracted(ranked(), &BTreeMap::new(), false, 3);
        assert_eq!(ids(&f), vec![1, 2, 3]);
        assert_eq!(f.filtered_retracted, 0);
    }

    #[test]
    fn every_retracted_candidate_is_counted_even_below_the_cut() {
        let r = BTreeMap::from([(2, 9), (4, 9)]);
        let f = filter_retracted(ranked(), &r, false, 1);
        assert_eq!(ids(&f), vec![1]);
        assert_eq!(f.filtered_retracted, 2);
    }
}

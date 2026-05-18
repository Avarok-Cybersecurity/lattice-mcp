//! Memory consolidation — both algorithmic (pure) and LLM-based modes.
//!
//! Clusters related episodic memories and synthesizes semantic knowledge.

use crate::memory::types::Memory;
use std::collections::HashMap;

/// Algorithmic consolidation: extract key terms and build a summary.
///
/// Pure function — no I/O. Scans content for frequently occurring
/// meaningful terms and constructs a knowledge statement.
pub fn algorithmic_summarize(memories: &[Memory]) -> String {
    if memories.is_empty() {
        return String::new();
    }

    if memories.len() == 1 {
        return memories[0].content.clone();
    }

    // Extract term frequencies across all memories
    let mut term_freq: HashMap<String, usize> = HashMap::new();
    let stop_words = stop_words();

    for mem in memories {
        let mut seen_in_doc = std::collections::HashSet::new();
        for word in mem.content.split_whitespace() {
            let normalized = word
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if normalized.len() > 2 && !stop_words.contains(normalized.as_str()) {
                if seen_in_doc.insert(normalized.clone()) {
                    *term_freq.entry(normalized).or_insert(0) += 1;
                }
            }
        }
    }

    // Find terms appearing in multiple memories (shared concepts)
    let mut shared_terms: Vec<(String, usize)> = term_freq
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .collect();
    shared_terms.sort_by(|a, b| b.1.cmp(&a.1));

    let top_terms: Vec<&str> = shared_terms
        .iter()
        .take(5)
        .map(|(t, _)| t.as_str())
        .collect();

    // Build summary from the most important memory + shared terms
    let most_important = memories
        .iter()
        .max_by(|a, b| {
            a.importance
                .partial_cmp(&b.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    if top_terms.is_empty() {
        format!(
            "Consolidated from {} related memories: {}",
            memories.len(),
            most_important.content
        )
    } else {
        format!(
            "Pattern across {} memories (key topics: {}): {}",
            memories.len(),
            top_terms.join(", "),
            most_important.content
        )
    }
}

/// Explicit parameters for clustering (PCND).
pub struct ClusterParams {
    pub min_overlap: usize,
    pub min_importance: f32,
    pub temporal_half_life_days: f64,
}

/// Detect clusters of related memories based on weighted term overlap.
///
/// Pre-filters by importance, then uses temporal proximity to weight
/// the raw term overlap. Memories closer in time cluster more easily.
pub fn detect_clusters(memories: &[Memory], params: ClusterParams) -> Vec<Vec<usize>> {
    let stop = stop_words();

    // Pre-filter: build (original_index, term_set) only for eligible memories
    let eligible: Vec<(usize, std::collections::HashSet<String>)> = memories
        .iter()
        .enumerate()
        .filter(|(_, m)| m.importance >= params.min_importance)
        .map(|(idx, m)| {
            let terms = m
                .content
                .split_whitespace()
                .map(|w| {
                    w.trim_matches(|c: char| !c.is_alphanumeric())
                        .to_lowercase()
                })
                .filter(|w| w.len() > 2 && !stop.contains(w.as_str()))
                .collect();
            (idx, terms)
        })
        .collect();

    let mut used = vec![false; eligible.len()];
    let mut clusters = Vec::new();

    for i in 0..eligible.len() {
        if used[i] {
            continue;
        }

        let mut cluster = vec![eligible[i].0];
        used[i] = true;

        for j in (i + 1)..eligible.len() {
            if used[j] {
                continue;
            }

            let overlap = eligible[i].1.intersection(&eligible[j].1).count();
            let weight = temporal_weight(
                &memories[eligible[i].0],
                &memories[eligible[j].0],
                params.temporal_half_life_days,
            );
            let effective = overlap as f64 * weight;

            if effective >= params.min_overlap as f64 {
                cluster.push(eligible[j].0);
                used[j] = true;
            }
        }

        if cluster.len() > 1 {
            clusters.push(cluster);
        }
    }

    clusters
}

/// Temporal proximity weight: 1.0 for same-time, decays with distance.
///
/// Formula: 1.0 / (1.0 + days_apart / half_life)
fn temporal_weight(a: &Memory, b: &Memory, half_life_days: f64) -> f64 {
    let secs_apart = (a.created_at - b.created_at)
        .num_seconds()
        .unsigned_abs() as f64;
    let days_apart = secs_apart / 86400.0;
    1.0 / (1.0 + days_apart / half_life_days)
}

fn stop_words() -> std::collections::HashSet<&'static str> {
    [
        "the", "a", "an", "is", "was", "are", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "for", "and", "nor", "but", "or", "yet", "so", "in", "on",
        "at", "to", "of", "with", "by", "from", "as", "into",
        "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "each", "every", "both", "few", "more", "most",
        "other", "some", "such", "no", "not", "only", "own", "same",
        "than", "too", "very", "just", "that", "this", "these",
        "those", "it", "its", "i", "we", "you", "they", "he", "she",
        "him", "her", "his", "their", "our", "your", "my",
    ]
    .into_iter()
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::{MemoryCategory, MemoryMetadata};
    use chrono::Utc;

    fn make_memory(content: &str, importance: f32) -> Memory {
        Memory {
            id: Memory::content_id(content, MemoryCategory::Episodic),
            category: MemoryCategory::Episodic,
            content: content.to_string(),
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 0,
            importance,
            session_id: None,
            metadata: MemoryMetadata::Episodic {
                event_type: "test".into(),
            },
            importance_state: Default::default(),
        }
    }

    #[test]
    fn summarize_single() {
        let mems = vec![make_memory("Fixed auth bug", 0.8)];
        let summary = algorithmic_summarize(&mems);
        assert_eq!(summary, "Fixed auth bug");
    }

    #[test]
    fn summarize_empty() {
        let summary = algorithmic_summarize(&[]);
        assert!(summary.is_empty());
    }

    #[test]
    fn summarize_multiple_shared_terms() {
        let mems = vec![
            make_memory("Fixed authentication error in login module", 0.9),
            make_memory("Authentication token validation error found", 0.7),
            make_memory("Login authentication flow needs refactoring", 0.6),
        ];
        let summary = algorithmic_summarize(&mems);
        assert!(summary.contains("3 memories"));
        assert!(summary.contains("authentication"));
    }

    fn default_params() -> ClusterParams {
        ClusterParams {
            min_overlap: 1,
            min_importance: 0.0,
            temporal_half_life_days: 365.0, // large → negligible temporal effect
        }
    }

    fn strict_params() -> ClusterParams {
        ClusterParams {
            min_overlap: 2,
            min_importance: 0.4,
            temporal_half_life_days: 7.0,
        }
    }

    #[test]
    fn detect_clusters_finds_related() {
        let mems = vec![
            make_memory("Fixed authentication error in login", 0.8),
            make_memory("Authentication token validation fixed", 0.7),
            make_memory("Database migration script updated", 0.5),
            make_memory("Database schema migration completed", 0.6),
        ];
        let clusters = detect_clusters(&mems, default_params());
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn detect_clusters_no_overlap() {
        let mems = vec![
            make_memory("alpha bravo charlie", 0.5),
            make_memory("delta echo foxtrot", 0.5),
        ];
        let clusters = detect_clusters(&mems, default_params());
        assert!(clusters.is_empty());
    }

    #[test]
    fn prefilter_excludes_low_importance() {
        let mems = vec![
            make_memory("Session started for project lattice-mcp", 0.3),
            make_memory("Session started for project lattice-mcp testing", 0.3),
            make_memory("Fixed authentication error in lattice-mcp", 0.8),
        ];
        // With min_importance=0.4, the two 0.3 session-starts are excluded
        let clusters = detect_clusters(&mems, strict_params());
        assert!(clusters.is_empty());
    }

    #[test]
    fn prefilter_keeps_high_importance() {
        let mems = vec![
            make_memory("Fixed authentication error in lattice-mcp login", 0.8),
            make_memory("Authentication token validation in lattice-mcp fixed", 0.7),
        ];
        let clusters = detect_clusters(&mems, strict_params());
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 2);
    }

    fn make_memory_at(content: &str, importance: f32, days_ago: i64) -> Memory {
        use chrono::Duration;
        let created = Utc::now() - Duration::days(days_ago);
        Memory {
            id: Memory::content_id(content, MemoryCategory::Episodic),
            category: MemoryCategory::Episodic,
            content: content.to_string(),
            created_at: created,
            last_accessed: created,
            access_count: 0,
            importance,
            session_id: None,
            metadata: MemoryMetadata::Episodic {
                event_type: "test".into(),
            },
            importance_state: Default::default(),
        }
    }

    #[test]
    fn temporal_proximity_boosts_clustering() {
        // Same day, 2 shared terms → should cluster
        let mems = vec![
            make_memory_at("Fixed authentication error in login", 0.8, 0),
            make_memory_at("Authentication error found in tests", 0.7, 0),
        ];
        let params = ClusterParams {
            min_overlap: 2,
            min_importance: 0.0,
            temporal_half_life_days: 7.0,
        };
        let clusters = detect_clusters(&mems, params);
        assert_eq!(clusters.len(), 1);
    }

    #[test]
    fn temporal_distance_weakens_clustering() {
        // 60 days apart with half_life=7 → temporal_weight ≈ 0.10
        // 2 shared terms × 0.10 = 0.20 < 2.0 threshold → should NOT cluster
        let mems = vec![
            make_memory_at("Fixed authentication error in login", 0.8, 0),
            make_memory_at("Authentication error found in tests", 0.7, 60),
        ];
        let params = ClusterParams {
            min_overlap: 2,
            min_importance: 0.0,
            temporal_half_life_days: 7.0,
        };
        let clusters = detect_clusters(&mems, params);
        assert!(clusters.is_empty());
    }
}

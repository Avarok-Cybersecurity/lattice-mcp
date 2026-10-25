//! Memory consolidation — both algorithmic (pure) and LLM-based modes.
//!
//! Clusters related episodic memories and synthesizes semantic knowledge.

use crate::prelude::*;

use crate::types::Memory;
use alloc::collections::BTreeMap;

pub mod distill;
pub mod trigger;

#[cfg(test)]
mod tests;

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
    let mut term_freq: BTreeMap<String, usize> = BTreeMap::new();
    let stop_words = stop_words();

    for mem in memories {
        let mut seen_in_doc = alloc::collections::BTreeSet::new();
        for word in mem.content.split_whitespace() {
            let normalized = word
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if normalized.len() > 2
                && !stop_words.contains(normalized.as_str())
                && seen_in_doc.insert(normalized.clone())
            {
                *term_freq.entry(normalized).or_insert(0) += 1;
            }
        }
    }

    // Find terms appearing in multiple memories (shared concepts)
    let mut shared_terms: Vec<(String, usize)> = term_freq
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .collect();
    shared_terms.sort_by_key(|t| core::cmp::Reverse(t.1));

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
                .unwrap_or(core::cmp::Ordering::Equal)
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
    let eligible: Vec<(usize, alloc::collections::BTreeSet<String>)> = memories
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
    let secs_apart = (a.created_at - b.created_at).num_seconds().unsigned_abs() as f64;
    let days_apart = secs_apart / 86400.0;
    1.0 / (1.0 + days_apart / half_life_days)
}

/// Shared English stopword set (SSOT — also reused by query transformation).
pub(crate) fn stop_words() -> alloc::collections::BTreeSet<&'static str> {
    [
        "the", "a", "an", "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "shall",
        "for", "and", "nor", "but", "or", "yet", "so", "in", "on", "at", "to", "of", "with", "by",
        "from", "as", "into", "through", "during", "before", "after", "above", "below", "between",
        "out", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "each", "every", "both", "few", "more", "most", "other",
        "some", "such", "no", "not", "only", "own", "same", "than", "too", "very", "just", "that",
        "this", "these", "those", "it", "its", "i", "we", "you", "they", "he", "she", "him", "her",
        "his", "their", "our", "your", "my",
    ]
    .into_iter()
    .collect()
}

/// Most frequent non-stopword term across a memory cluster (SSOT term scan
/// shared by distillation). Returns `None` for an empty/all-stopword cluster.
pub(crate) fn top_shared_term(memories: &[Memory]) -> Option<String> {
    let stop = stop_words();
    let mut freq: BTreeMap<String, usize> = BTreeMap::new();
    for m in memories {
        for w in m.content.split_whitespace() {
            let t = w
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if t.len() > 2 && !stop.contains(t.as_str()) {
                *freq.entry(t).or_insert(0) += 1;
            }
        }
    }
    // Deterministic: highest frequency, ties broken lexicographically
    // (BTreeMap iteration order is unspecified — never tie-break on it).
    freq.into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
        .map(|(t, _)| t)
}

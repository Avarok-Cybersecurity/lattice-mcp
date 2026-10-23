//! Benchmark harness: scripted memory trajectories + deterministic metrics.
//!
//! Purpose: quantify retrieval/ranking/policy quality so each SOTA feature
//! can be measured OFF vs ON (`assert_improves`). Uses the deterministic
//! `MockEmbedder` (bag-of-words, no network).
//!
//! Honest scope: this reliably gates *geometry/ranking/policy* features
//! (adaptive fusion, novelty gating, MMR, eviction, auto-consolidation)
//! because those are decidable from the mock's keyword geometry. It is
//! only *directional* for LLM-semantic features (contradiction detection,
//! query transformation, structured distillation) — those must be judged
//! with a real chat provider, not blocked by this harness.
#![allow(dead_code)]

use lattice_mcp::memory::types::MemoryCategory;
use lattice_mcp::memory::MemorySystem;

/// One step in a scripted agent trajectory.
pub enum Step {
    /// Store a memory.
    Write {
        content: &'static str,
        category: MemoryCategory,
        importance: f32,
        event_type: &'static str,
    },
    /// Query memory; relevance is judged by substring match on returned content
    /// (deterministic and independent of the content-hash IDs).
    Recall {
        query: &'static str,
        k: usize,
        /// Substrings that a *relevant* result is expected to contain.
        relevant: &'static [&'static str],
    },
    /// Provide recall feedback for a previously written memory.
    Rate {
        content: &'static str,
        category: MemoryCategory,
        quality: u8,
    },
}

/// Aggregate retrieval-quality metrics over a scenario.
#[derive(Debug, Clone, Copy)]
pub struct Metrics {
    /// Mean recall@k across all `Recall` steps: fraction of expected
    /// relevant substrings that appear within the top-k results.
    pub recall_at_k: f32,
    /// Mean precision@k: fraction of top-k results that are relevant.
    /// This is the ranking-sensitive discriminator — distractors leaking
    /// into the top-k push it below 1.0, leaving headroom for fusion/MMR.
    pub precision_at_k: f32,
    /// Mean reciprocal rank of the first relevant result across `Recall` steps.
    pub mrr: f32,
    /// Number of `Recall` steps measured (denominator sanity check).
    pub queries: usize,
}

/// Execute a scripted trajectory and return aggregate metrics.
pub async fn run_scenario(sys: &mut MemorySystem, steps: &[Step]) -> Metrics {
    let mut recall_sum = 0.0_f32;
    let mut precision_sum = 0.0_f32;
    let mut mrr_sum = 0.0_f32;
    let mut queries = 0_usize;

    for step in steps {
        match step {
            Step::Write {
                content,
                category,
                importance,
                event_type,
            } => {
                sys.remember(
                    content,
                    *category,
                    Some(*importance),
                    Some(event_type),
                    None,
                    None,
                )
                .await
                .expect("scenario write");
            }
            Step::Rate {
                content,
                category,
                quality,
            } => {
                let id = lattice_mcp::memory::types::Memory::content_id(content, *category);
                sys.rate_recall(*category, id, *quality)
                    .await
                    .expect("scenario rate");
            }
            Step::Recall { query, k, relevant } => {
                let results = sys.recall(query, &[], *k).await.expect("scenario recall");
                let top: Vec<String> = results
                    .iter()
                    .take(*k)
                    .map(|s| s.memory.content.clone())
                    .collect();

                // recall@k: fraction of expected relevant substrings found in top-k.
                let found = relevant
                    .iter()
                    .filter(|sub| top.iter().any(|c| c.contains(*sub)))
                    .count();
                recall_sum += found as f32 / relevant.len().max(1) as f32;

                // precision@k: fraction of returned top-k that are relevant.
                let relevant_hits = top
                    .iter()
                    .filter(|c| relevant.iter().any(|sub| c.contains(sub)))
                    .count();
                precision_sum += relevant_hits as f32 / (*k).max(1) as f32;

                // MRR: 1/rank of first result matching any relevant substring.
                let rr = top
                    .iter()
                    .position(|c| relevant.iter().any(|sub| c.contains(sub)))
                    .map(|pos| 1.0 / (pos as f32 + 1.0))
                    .unwrap_or(0.0);
                mrr_sum += rr;

                queries += 1;
            }
        }
    }

    let denom = queries.max(1) as f32;
    Metrics {
        recall_at_k: recall_sum / denom,
        precision_at_k: precision_sum / denom,
        mrr: mrr_sum / denom,
        queries,
    }
}

/// Assert an ON configuration is at least as good as OFF on every metric.
/// Used by feature steps to gate "smarter" claims.
pub fn assert_improves(feature: &str, off: Metrics, on: Metrics) {
    assert!(
        on.recall_at_k >= off.recall_at_k - 1e-4,
        "{feature}: recall@k regressed OFF={:.4} ON={:.4}",
        off.recall_at_k,
        on.recall_at_k
    );
    assert!(
        on.precision_at_k >= off.precision_at_k - 1e-4,
        "{feature}: precision@k regressed OFF={:.4} ON={:.4}",
        off.precision_at_k,
        on.precision_at_k
    );
    assert!(
        on.mrr >= off.mrr - 1e-4,
        "{feature}: MRR regressed OFF={:.4} ON={:.4}",
        off.mrr,
        on.mrr
    );
}

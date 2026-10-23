//! Reusable retrieval-evaluation harness (embedder-agnostic).
//!
//! Seeds a labelled corpus into a `MemorySystem`, runs each query through
//! `recall`, and scores recall@k / precision@k / MRR / nDCG@k, broken down by
//! LoCoMo-style question category. The LAN-wired driver lives in
//! `tests/eval_lan.rs`; this module is pure measurement + the run loop.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use lattice_mcp::memory::types::MemoryCategory;
use lattice_mcp::memory::MemorySystem;

/// LoCoMo-style question categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QKind {
    SingleHop,
    Temporal,
    MultiHop,
    OpenDomain,
    Lexical, // exact-token / identifier queries where lexical overlap helps
}

pub struct EvalDoc {
    pub label: &'static str,
    pub content: &'static str,
    pub category: MemoryCategory,
    /// Labels of docs this one should link to (related_to), for multi-hop.
    pub links: &'static [&'static str],
}

pub struct EvalQuery {
    pub text: &'static str,
    pub relevant: &'static [&'static str], // doc labels considered relevant
    pub kind: QKind,
}

pub struct EvalCorpus {
    pub docs: Vec<EvalDoc>,
    pub queries: Vec<EvalQuery>,
}

#[derive(Debug, Clone, Default)]
pub struct Metrics {
    pub recall_at_k: f32,
    pub precision_at_k: f32,
    pub mrr: f32,
    pub ndcg_at_k: f32,
    pub n: usize,
}

#[derive(Debug, Clone)]
pub struct EvalReport {
    pub k: usize,
    pub overall: Metrics,
    pub by_kind: HashMap<QKind, Metrics>,
}

/// Seed the corpus; returns label -> stored memory id. Two passes so that
/// `links` (related_to) can reference already-stored docs.
pub async fn seed(sys: &mut MemorySystem, corpus: &EvalCorpus) -> HashMap<&'static str, u64> {
    let mut ids: HashMap<&'static str, u64> = HashMap::new();
    for d in &corpus.docs {
        let m = sys
            .remember(d.content, d.category, Some(0.7), Some("eval"), None, None)
            .await
            .expect("seed remember");
        ids.insert(d.label, m.id);
    }
    // Second pass: add explicit links (same-category only; cross-collection
    // edges are unsupported by LatticeDB).
    for d in &corpus.docs {
        if d.links.is_empty() {
            continue;
        }
        let from = ids[d.label];
        for l in d.links {
            if let Some(&to) = ids.get(l) {
                let _ = sys.connect(from, to, "RELATES_TO", 0.9, d.category).await;
            }
        }
    }
    ids
}

fn dcg(rels: &[bool]) -> f32 {
    rels.iter()
        .enumerate()
        .map(|(i, &r)| {
            if r {
                1.0 / ((i as f32 + 2.0).log2())
            } else {
                0.0
            }
        })
        .sum()
}

/// Run all queries and aggregate metrics overall and per category.
pub async fn run(
    sys: &mut MemorySystem,
    corpus: &EvalCorpus,
    ids: &HashMap<&'static str, u64>,
    k: usize,
) -> EvalReport {
    let mut overall = Metrics::default();
    let mut buckets: HashMap<QKind, Metrics> = HashMap::new();

    for q in &corpus.queries {
        let hits = sys.recall(q.text, &[], k).await.expect("eval recall");
        let retrieved: Vec<u64> = hits.iter().take(k).map(|h| h.memory.id).collect();
        let relevant: HashSet<u64> = q
            .relevant
            .iter()
            .filter_map(|l| ids.get(l).copied())
            .collect();
        let rel_n = relevant.len().max(1);

        let hit_flags: Vec<bool> = retrieved.iter().map(|id| relevant.contains(id)).collect();
        let found = hit_flags.iter().filter(|b| **b).count();

        let recall = found as f32 / rel_n as f32;
        let precision = found as f32 / k as f32;
        let mrr = hit_flags
            .iter()
            .position(|b| *b)
            .map(|p| 1.0 / (p as f32 + 1.0))
            .unwrap_or(0.0);
        let ideal = dcg(&vec![true; relevant.len().min(k)]);
        let ndcg = if ideal > 0.0 {
            dcg(&hit_flags) / ideal
        } else {
            0.0
        };

        for m in [&mut overall, buckets.entry(q.kind).or_default()] {
            m.recall_at_k += recall;
            m.precision_at_k += precision;
            m.mrr += mrr;
            m.ndcg_at_k += ndcg;
            m.n += 1;
        }
    }

    let finalize = |m: &mut Metrics| {
        let d = m.n.max(1) as f32;
        m.recall_at_k /= d;
        m.precision_at_k /= d;
        m.mrr /= d;
        m.ndcg_at_k /= d;
    };
    finalize(&mut overall);
    for m in buckets.values_mut() {
        finalize(m);
    }
    EvalReport {
        k,
        overall,
        by_kind: buckets,
    }
}

pub fn print_report(label: &str, r: &EvalReport) {
    eprintln!(
        "[eval:{label}] k={} overall recall@k={:.3} precision@k={:.3} mrr={:.3} ndcg@k={:.3} (n={})",
        r.k, r.overall.recall_at_k, r.overall.precision_at_k, r.overall.mrr, r.overall.ndcg_at_k, r.overall.n
    );
    let mut kinds: Vec<_> = r.by_kind.keys().copied().collect();
    kinds.sort_by_key(|k| format!("{k:?}"));
    for kind in kinds {
        let m = &r.by_kind[&kind];
        eprintln!(
            "[eval:{label}]   {:?}: recall@k={:.3} prec@k={:.3} mrr={:.3} ndcg@k={:.3} (n={})",
            kind, m.recall_at_k, m.precision_at_k, m.mrr, m.ndcg_at_k, m.n
        );
    }
}

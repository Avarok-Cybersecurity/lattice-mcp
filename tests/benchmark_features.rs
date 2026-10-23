//! Benchmark harness — consolidation / policy / experimental gates.

mod common;

use lattice_mcp::config::{ContradictionConfig, DiversifyConfig, QueryTransformConfig};

use common::bench_driver::{make_system, measure};
use common::bench_fixtures::assert_improves;
use common::config::make_test_config;

#[tokio::test]
async fn mmr_tradeoff_is_bounded_and_pipeline_intact() {
    let off = measure(make_test_config(tempfile::tempdir().unwrap().path())).await;

    let mut on_cfg = make_test_config(tempfile::tempdir().unwrap().path());
    on_cfg.diversify = Some(DiversifyConfig {
        enabled: true,
        lambda: 0.9, // mild diversification
    });
    let on = measure(on_cfg).await;

    eprintln!(
        "[mmr] OFF recall={:.4} prec={:.4} mrr={:.4} | ON recall={:.4} prec={:.4} mrr={:.4}",
        off.recall_at_k, off.precision_at_k, off.mrr, on.recall_at_k, on.precision_at_k, on.mrr
    );
    assert_eq!(
        on.queries, off.queries,
        "pipeline integrity: same query count"
    );
    // Bounded tradeoff: a mild λ may shuffle within top-k for diversity but
    // must not collapse recall (≥ 80% of the relevance-only baseline).
    assert!(
        on.recall_at_k >= off.recall_at_k * 0.8,
        "MMR recall collapsed: OFF={:.4} ON={:.4}",
        off.recall_at_k,
        on.recall_at_k
    );
    assert!(
        on.mrr >= off.mrr * 0.8,
        "MMR MRR collapsed: OFF={:.4} ON={:.4}",
        off.mrr,
        on.mrr
    );
}

/// Step 7 functional proof: with distillation enabled, a consolidated
/// semantic memory carries its source IDs in `metadata.provenance` — the
/// provenance-loss fix (previously only in summary text). The algorithmic
/// distillation itself is unit-tested in `consolidation::distill`.
#[tokio::test]
async fn distillation_records_provenance() {
    use lattice_mcp::memory::types::{MemoryCategory, MemoryMetadata};

    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = make_test_config(tmp.path());
    cfg.consolidation.distill_enabled = Some(true);
    let mut sys = make_system(tmp.path(), cfg).await;

    let mut src_ids = Vec::new();
    for c in [
        "authentication bcrypt error found in login flow",
        "authentication bcrypt error fixed in login module",
        "authentication bcrypt error retest passed for login",
    ] {
        let m = sys
            .remember(
                c,
                MemoryCategory::Episodic,
                Some(0.8),
                Some("error"),
                None,
                None,
            )
            .await
            .unwrap();
        src_ids.push(m.id);
    }

    let created = sys
        .reflect(
            Some(MemoryCategory::Episodic),
            Some("authentication bcrypt login"),
        )
        .await
        .expect("reflect");

    assert!(
        !created.is_empty(),
        "distillation should consolidate the cluster"
    );
    let consolidated = &created[0];
    match &consolidated.metadata {
        MemoryMetadata::Semantic { provenance, .. } => {
            assert!(
                !provenance.is_empty(),
                "consolidated memory must record source provenance"
            );
            assert!(
                provenance.iter().all(|p| src_ids.contains(p)),
                "provenance IDs must reference the source episodic memories"
            );
        }
        other => panic!("consolidated memory should be Semantic, got {other:?}"),
    }
}

/// Regression: the consolidated semantic memory must keep the auto-link edges
/// that `run_inner` creates after storing it. Provenance is written on the
/// SINGLE authoritative upsert (threaded into `run_inner`); an earlier version
/// did a SECOND upsert solely to set provenance, which wiped those edges
/// (lattice-core replaces a point wholesale on upsert). Consolidating mutually
/// similar Semantic sources yields a summary derived from their text, so the
/// new point is similar enough to auto-link back — giving a deterministic edge
/// whose survival proves the wipe is gone.
#[tokio::test]
async fn reflect_preserves_consolidated_edges() {
    use lattice_mcp::memory::types::MemoryCategory;

    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = make_test_config(tmp.path());
    cfg.consolidation.distill_enabled = Some(true); // exercise the provenance path
    let mut sys = make_system(tmp.path(), cfg).await;

    for c in [
        "authentication bcrypt password hashed in login module",
        "authentication bcrypt password validation fixed in login",
        "authentication bcrypt password error found during login",
    ] {
        sys.remember(
            c,
            MemoryCategory::Semantic,
            Some(0.8),
            Some("decision"),
            None,
            None,
        )
        .await
        .unwrap();
    }

    let created = sys
        .reflect(
            Some(MemoryCategory::Semantic),
            Some("authentication bcrypt password login"),
        )
        .await
        .expect("reflect");
    assert!(
        !created.is_empty(),
        "reflect should consolidate the cluster"
    );

    let consolidated = &created[0];
    let hits = sys
        .search_graph(MemoryCategory::Semantic, consolidated.id, 1, None)
        .await
        .expect("traverse");
    assert!(
        !hits.is_empty(),
        "consolidated memory lost its edges — second-upsert edge-wipe regressed"
    );
}

/// Step 8 functional proof: with auto-consolidation enabled, reaching the
/// threshold then calling `session_end` triggers `reflect` automatically
/// (semantic memories appear; the trigger predicate is unit-tested in
/// `consolidation::trigger`). Recursion-safety is structural — the trigger
/// is checked at the session boundary, never inside `remember`.
#[tokio::test]
async fn auto_consolidation_fires_at_session_end() {
    use lattice_mcp::memory::types::MemoryCategory;

    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = make_test_config(tmp.path());
    cfg.memory.consolidation_threshold = 3;
    cfg.consolidation.auto_consolidate = Some(true);
    let mut sys = make_system(tmp.path(), cfg).await;

    sys.session_start(Some("proj"), Some("auth work"))
        .await
        .unwrap();
    for c in [
        "authentication bcrypt error found in login flow",
        "authentication bcrypt error fixed in login module",
        "authentication bcrypt error retest passed for login",
        "authentication bcrypt error documented for login",
    ] {
        sys.remember(
            c,
            MemoryCategory::Episodic,
            Some(0.8),
            Some("error"),
            None,
            None,
        )
        .await
        .unwrap();
    }
    let semantic_before = sys
        .stats()
        .await
        .unwrap()
        .by_category
        .get("semantic")
        .copied()
        .unwrap_or(0);

    sys.session_end(Some("done")).await.unwrap();

    let semantic_after = sys
        .stats()
        .await
        .unwrap()
        .by_category
        .get("semantic")
        .copied()
        .unwrap_or(0);
    assert!(
        semantic_after > semantic_before,
        "auto-consolidation should create semantic memories at session_end \
         (before={semantic_before}, after={semantic_after})"
    );
}

/// Step 9 functional proof: a negation-polarity flip on a semantically
/// near-identical Semantic memory creates a `CONTRADICTS` edge (algorithmic
/// path, no chat). Control: with the feature off, no such edge appears.
/// The disagreement scoring itself is unit-tested in `memory::contradiction`.
#[tokio::test]
async fn contradiction_creates_edge_on_polarity_flip() {
    use lattice_mcp::memory::types::MemoryCategory;

    async fn run(enabled: bool) -> bool {
        let tmp = tempfile::tempdir().unwrap();
        let mut cfg = make_test_config(tmp.path());
        if enabled {
            cfg.contradiction = Some(ContradictionConfig {
                enabled: true,
                min_similarity: 0.6,
                auto_supersede: false,
            });
        }
        let mut sys = make_system(tmp.path(), cfg).await;

        let m1 = sys
            .remember(
                "authentication must use bcrypt for password hashing",
                MemoryCategory::Semantic,
                Some(0.8),
                Some("decision"),
                None,
                None,
            )
            .await
            .unwrap();
        let m2 = sys
            .remember(
                "authentication must not use bcrypt for password hashing",
                MemoryCategory::Semantic,
                Some(0.8),
                Some("decision"),
                None,
                None,
            )
            .await
            .unwrap();

        let hits = sys
            .search_graph(MemoryCategory::Semantic, m2.id, 1, Some(&["CONTRADICTS"]))
            .await
            .unwrap();
        hits.iter().any(|h| h.target_id == m1.id)
    }

    assert!(
        run(true).await,
        "polarity flip must create a CONTRADICTS edge"
    );
    assert!(
        !run(false).await,
        "feature off ⇒ no CONTRADICTS edge (behavior preserved)"
    );
}

/// Step 10 gate: the conservative algorithmic query transform (stopword
/// strip + dedupe) must not regress retrieval. The rewrite logic itself is
/// unit-tested in `retrieval::query`.
#[tokio::test]
async fn query_transform_does_not_regress() {
    let off = measure(make_test_config(tempfile::tempdir().unwrap().path())).await;

    let mut on_cfg = make_test_config(tempfile::tempdir().unwrap().path());
    on_cfg.query_transform = Some(QueryTransformConfig { enabled: true });
    let on = measure(on_cfg).await;

    eprintln!(
        "[qxform] OFF recall={:.4} prec={:.4} mrr={:.4} | ON recall={:.4} prec={:.4} mrr={:.4}",
        off.recall_at_k, off.precision_at_k, off.mrr, on.recall_at_k, on.precision_at_k, on.mrr
    );
    assert_improves("query transformation", off, on);
}

//! Benchmark harness — core metrics + ranking-feature gates.
//! Run: `cargo test --test benchmark` (features: `--test benchmark_features`).

mod common;

use lattice_mcp::config::{FusionConfig, FusionMode, LifecycleConfig, WritePolicyConfig};

use common::bench_driver::{make_system, measure};
use common::bench_fixtures::assert_improves;
use common::config::make_test_config;

/// Baseline regression guard: with the default (all-features-off) config the
/// harness must compute sane, non-degenerate retrieval metrics. This both
/// proves the measurement plumbing works and pins the OFF numbers that later
/// feature scenarios compare against.
#[tokio::test]
async fn baseline_recall_quality() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let baseline = measure(make_test_config(tmp.path())).await;

    eprintln!(
        "[benchmark baseline] recall@k={:.4} precision@k={:.4} mrr={:.4} queries={}",
        baseline.recall_at_k, baseline.precision_at_k, baseline.mrr, baseline.queries
    );

    assert_eq!(baseline.queries, 5, "scenario should run 5 recall queries");
    assert!(
        baseline.recall_at_k > 0.5,
        "baseline recall@k should be well above chance, got {:.4}",
        baseline.recall_at_k
    );
    assert!(
        baseline.mrr > 0.5,
        "baseline MRR should be well above chance, got {:.4}",
        baseline.mrr
    );
    // The gate is only meaningful if OFF is NOT saturated: there must be
    // measurable headroom for later ranking/policy features to improve.
    assert!(
        baseline.precision_at_k < 1.0,
        "baseline precision@k must leave headroom (got {:.4} — scenario too easy)",
        baseline.precision_at_k
    );
    assert!(
        baseline.precision_at_k > 0.3,
        "baseline precision@k should be above chance, got {:.4}",
        baseline.precision_at_k
    );
}

/// Step 3 gate: enabling novelty-gated writes (skip near-duplicates) must
/// not regress retrieval quality — collapsing the redundant near-dup should
/// free a top-k slot rather than hurt recall.
#[tokio::test]
async fn novelty_gating_does_not_regress() {
    let off = measure(make_test_config(tempfile::tempdir().unwrap().path())).await;

    let mut on_cfg = make_test_config(tempfile::tempdir().unwrap().path());
    on_cfg.write_policy = Some(WritePolicyConfig {
        enabled: true,
        near_dup_threshold: 0.85,
        skip_near_dups: true,
    });
    let on = measure(on_cfg).await;

    eprintln!(
        "[novelty] OFF recall={:.4} prec={:.4} mrr={:.4} | ON recall={:.4} prec={:.4} mrr={:.4}",
        off.recall_at_k, off.precision_at_k, off.mrr, on.recall_at_k, on.precision_at_k, on.mrr
    );
    assert_improves("novelty-gated writes", off, on);
}

/// Step 3 functional proof: a near-duplicate write is collapsed onto the
/// existing memory (returns the same id; episodic count does not grow).
/// The retrieval metrics above are flat for novelty (the dup and original
/// share the relevant substring), so correctness is asserted directly here
/// rather than manufactured into the ranking scenario.
#[tokio::test]
async fn novelty_skip_collapses_near_duplicate() {
    use lattice_mcp::memory::types::MemoryCategory;

    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = make_test_config(tmp.path());
    cfg.write_policy = Some(WritePolicyConfig {
        enabled: true,
        near_dup_threshold: 0.85,
        skip_near_dups: true,
    });
    let mut sys = make_system(tmp.path(), cfg).await;

    let original = sys
        .remember(
            "Fixed authentication error bcrypt hash comparison was case sensitive",
            MemoryCategory::Episodic,
            Some(0.9),
            Some("error"),
            None,
            None,
        )
        .await
        .expect("write original");

    let count_after_original = sys
        .stats()
        .await
        .unwrap()
        .by_category
        .get("episodic")
        .copied()
        .unwrap_or(0);

    let dup = sys
        .remember(
            "Fixed authentication bug bcrypt hash comparison was case sensitive too",
            MemoryCategory::Episodic,
            Some(0.6),
            Some("error"),
            None,
            None,
        )
        .await
        .expect("write near-dup");

    let count_after_dup = sys
        .stats()
        .await
        .unwrap()
        .by_category
        .get("episodic")
        .copied()
        .unwrap_or(0);

    assert_eq!(
        dup.id, original.id,
        "near-duplicate write should return the existing memory"
    );
    assert_eq!(
        count_after_dup, count_after_original,
        "near-duplicate must not grow the episodic collection"
    );
}

/// Step 4 gate: confidence-weighted fusion must not regress retrieval.
/// Rated memories (JWT q5, bcrypt q4) gain Bayesian evidence, so importance
/// is weighted up for them — relevant rated items should hold or rise.
#[tokio::test]
async fn confidence_fusion_does_not_regress() {
    let off = measure(make_test_config(tempfile::tempdir().unwrap().path())).await;

    let mut on_cfg = make_test_config(tempfile::tempdir().unwrap().path());
    on_cfg.fusion = Some(FusionConfig {
        enabled: true,
        mode: FusionMode::ConfidenceWeighted,
        rrf_k: 60.0,
    });
    let on = measure(on_cfg).await;

    eprintln!(
        "[fusion-cw] OFF recall={:.4} prec={:.4} mrr={:.4} | ON recall={:.4} prec={:.4} mrr={:.4}",
        off.recall_at_k, off.precision_at_k, off.mrr, on.recall_at_k, on.precision_at_k, on.mrr
    );
    assert_improves("confidence-weighted fusion", off, on);
}

/// Step 5 functional proof: a utility-driven prune enforces the per-category
/// budget (evicts low-utility overflow when `allow_delete`), and is a no-op
/// while within budget.
#[tokio::test]
async fn lifecycle_prune_enforces_budget() {
    use lattice_mcp::memory::types::MemoryCategory;

    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = make_test_config(tmp.path());
    cfg.lifecycle = Some(LifecycleConfig {
        enabled: true,
        budget_per_category: 2,
        allow_delete: true,
        delete_below: 1.0, // any overflow (utility < 1.0) is evicted
        prune_on_session_end: false,
    });
    let mut sys = make_system(tmp.path(), cfg).await;

    for i in 0..5 {
        sys.remember(
            &format!("episodic memory number {i} about authentication work"),
            MemoryCategory::Episodic,
            Some(0.5),
            Some("change"),
            None,
            None,
        )
        .await
        .expect("write");
    }
    let before = sys
        .stats()
        .await
        .unwrap()
        .by_category
        .get("episodic")
        .copied()
        .unwrap_or(0);
    assert_eq!(before, 5, "5 distinct episodic memories written");

    let report = sys.prune(MemoryCategory::Episodic).await.expect("prune");
    let after = sys
        .stats()
        .await
        .unwrap()
        .by_category
        .get("episodic")
        .copied()
        .unwrap_or(0);

    assert_eq!(after, 2, "prune must enforce budget_per_category");
    assert_eq!(report.kept, 2);
    assert_eq!(report.evicted, 3);

    // Second prune is a no-op (already within budget).
    let again = sys.prune(MemoryCategory::Episodic).await.expect("prune2");
    assert_eq!(
        again.demoted + again.evicted,
        0,
        "within-budget prune is a no-op"
    );
}

/// Step 6 gate: MMR is a relevance/diversity tradeoff — `assert_improves`
/// on recall is the wrong gate (it would *penalize* correct
/// diversification). Two honest guarantees instead: (1) pipeline integrity
/// (results stay ≤ limit, no error), and (2) at a mild λ the tradeoff is
/// *bounded* — recall must not collapse. The diversification mechanics are
/// unit-tested in `retrieval::diversify` (`diversity_demotes_near_duplicate`).
/// Determinism guard: identical config + scenario ⇒ identical metrics.
/// Feature OFF-vs-ON comparisons rely on this stability.
#[tokio::test]
async fn scenario_is_deterministic() {
    let tmp_a = tempfile::tempdir().expect("tempdir");
    let tmp_b = tempfile::tempdir().expect("tempdir");
    let a = measure(make_test_config(tmp_a.path())).await;
    let b = measure(make_test_config(tmp_b.path())).await;
    assert!(
        (a.recall_at_k - b.recall_at_k).abs() < 1e-6 && (a.mrr - b.mrr).abs() < 1e-6,
        "scenario must be deterministic: a={a:?} b={b:?}"
    );
}

//! Integration test: temporal range filtering on recall.

mod common;

use chrono::{Duration, Utc};
use lattice_mcp::memory::types::MemoryCategory;
use lattice_mcp::memory::MemorySystem;
use lattice_mcp::store::embedded::EmbeddedStore;

use common::config::make_test_config;
use common::mock_embedder::MockEmbedder;

async fn sys(dir: &std::path::Path) -> MemorySystem {
    let cfg = make_test_config(dir);
    let mut s = MemorySystem::new(
        Box::new(EmbeddedStore::new(dir.to_path_buf(), 64)),
        Box::new(MockEmbedder),
        None,
        None,
        cfg,
    );
    s.initialize().await.unwrap();
    s
}

#[tokio::test]
async fn temporal_range_filters_by_created_at() {
    let tmp = tempfile::tempdir().unwrap();
    let mut s = sys(tmp.path()).await;

    for c in [
        "authentication uses JWT bearer tokens",
        "passwords hashed with bcrypt",
        "redis handles session caching",
    ] {
        s.remember(
            c,
            MemoryCategory::Semantic,
            Some(0.7),
            Some("decision"),
            None,
            None,
        )
        .await
        .unwrap();
    }
    let now = Utc::now();

    // No filter: finds matches.
    let base = s
        .recall("how are users authenticated", &[], 5)
        .await
        .unwrap();
    assert!(!base.is_empty(), "baseline recall should return results");

    // after_date in the future -> everything is older -> empty.
    let future = s
        .recall_in_range(
            "how are users authenticated",
            &[],
            5,
            Some(now + Duration::hours(1)),
            None,
        )
        .await
        .unwrap();
    assert!(
        future.is_empty(),
        "after_date in the future must exclude all memories"
    );

    // before_date in the past -> everything is newer -> empty.
    let past = s
        .recall_in_range(
            "how are users authenticated",
            &[],
            5,
            None,
            Some(now - Duration::hours(1)),
        )
        .await
        .unwrap();
    assert!(
        past.is_empty(),
        "before_date in the past must exclude all memories"
    );

    // after_date in the past -> all retained.
    let included = s
        .recall_in_range(
            "how are users authenticated",
            &[],
            5,
            Some(now - Duration::hours(1)),
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        included.len(),
        base.len(),
        "a past after_date should not drop anything"
    );
}

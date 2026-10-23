//! Every timestamp an operation writes comes from the injected `Clock`.

mod common;

use chrono::{DateTime, Duration, Utc};
use lattice_mcp::memory::types::MemoryCategory;
use lattice_mcp::memory::MemorySystem;
use lattice_mcp::store::embedded::EmbeddedStore;
use lattice_memory_core::clock::FixedClock;

use common::config::make_test_config;
use common::mock_embedder::MockEmbedder;

fn fixed() -> DateTime<Utc> {
    DateTime::from_timestamp(1_600_000_000, 0).expect("valid timestamp")
}

async fn sys(dir: &std::path::Path, clock: Option<FixedClock>) -> MemorySystem {
    let mut s = MemorySystem::new(
        Box::new(EmbeddedStore::new(dir.to_path_buf(), 64)),
        Box::new(MockEmbedder),
        None,
        None,
        make_test_config(dir),
    );
    if let Some(c) = clock {
        s = s.with_clock(Box::new(c));
    }
    s.initialize().await.unwrap();
    s
}

#[tokio::test]
async fn remember_and_update_stamp_the_injected_clock() {
    let tmp = tempfile::tempdir().unwrap();
    let mut s = sys(tmp.path(), Some(FixedClock(fixed()))).await;
    let m = s
        .remember(
            "passwords hashed with bcrypt",
            MemoryCategory::Semantic,
            Some(0.7),
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!((m.created_at, m.last_accessed), (fixed(), fixed()));

    let u = s
        .update_memory(MemoryCategory::Semantic, m.id, None, Some(0.9))
        .await
        .unwrap()
        .expect("exists");
    assert_eq!(u.last_accessed, fixed());

    // The recall window sees the stored, injected timestamp.
    let hits = s
        .recall_in_range("bcrypt", &[], 5, None, Some(fixed() + Duration::seconds(1)))
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
}

#[tokio::test]
async fn without_an_injected_clock_the_wall_clock_is_used() {
    let tmp = tempfile::tempdir().unwrap();
    let mut s = sys(tmp.path(), None).await;
    let before = Utc::now();
    let m = s
        .remember(
            "redis handles session caching",
            MemoryCategory::Semantic,
            Some(0.7),
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert!(m.created_at >= before && m.created_at <= Utc::now());
}

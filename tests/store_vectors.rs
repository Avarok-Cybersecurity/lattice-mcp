//! Step 2C: `MemoryStore::get_vectors` roundtrip on the embedded store.
//!
//! Verifies stored embedding vectors can be fetched back by ID — the
//! load-bearing capability for MMR diversification (Step 6).

use chrono::Utc;

use lattice_mcp::memory::types::{Memory, MemoryCategory, MemoryMetadata};
use lattice_mcp::store::embedded::EmbeddedStore;
use lattice_mcp::store::MemoryStore;

fn make_memory(content: &str) -> Memory {
    Memory {
        id: Memory::content_id(content, MemoryCategory::Episodic),
        category: MemoryCategory::Episodic,
        content: content.to_string(),
        created_at: Utc::now(),
        last_accessed: Utc::now(),
        access_count: 0,
        importance: 0.5,
        session_id: None,
        metadata: MemoryMetadata::Episodic {
            event_type: "test".into(),
        },
        importance_state: Default::default(),
    }
}

#[tokio::test]
async fn get_vectors_roundtrips_and_skips_missing() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut store = EmbeddedStore::new(tmp.path().to_path_buf(), 4);
    store.initialize().await.expect("initialize");

    let m1 = make_memory("alpha vector memory");
    let m2 = make_memory("beta vector memory");
    let v1 = vec![1.0_f32, 0.0, 0.0, 0.0];
    let v2 = vec![0.0_f32, 1.0, 0.0, 0.0];

    store
        .upsert_memory(&m1, v1.clone())
        .await
        .expect("upsert m1");
    store
        .upsert_memory(&m2, v2.clone())
        .await
        .expect("upsert m2");

    let missing_id = 999_999_u64;
    let fetched = store
        .get_vectors(MemoryCategory::Episodic, &[m1.id, m2.id, missing_id])
        .await
        .expect("get_vectors");

    // Missing IDs are skipped, present ones returned with exact vectors.
    assert_eq!(fetched.len(), 2, "only existing IDs returned");

    let got1 = fetched
        .iter()
        .find(|(id, _)| *id == m1.id)
        .expect("m1 present");
    let got2 = fetched
        .iter()
        .find(|(id, _)| *id == m2.id)
        .expect("m2 present");
    assert_eq!(got1.1, v1, "m1 vector roundtrips exactly");
    assert_eq!(got2.1, v2, "m2 vector roundtrips exactly");
}

/// A state-only re-upsert (importance/pagerank/feedback update) must NOT erase
/// the point's graph edges. lattice-core stores edges on the source point and
/// replaces it wholesale on upsert; `upsert_memory` carries existing edges
/// across the write so rate_recall / update_pagerank / prune / update_memory
/// never silently destroy the graph.
#[tokio::test]
async fn upsert_preserves_existing_edges() {
    use lattice_mcp::memory::types::relations;

    let tmp = tempfile::tempdir().expect("tempdir");
    let mut store = EmbeddedStore::new(tmp.path().to_path_buf(), 4);
    store.initialize().await.expect("initialize");

    let m1 = make_memory("alpha source memory");
    let m2 = make_memory("beta target memory");
    store
        .upsert_memory(&m1, vec![1.0, 0.0, 0.0, 0.0])
        .await
        .expect("upsert m1");
    store
        .upsert_memory(&m2, vec![0.0, 1.0, 0.0, 0.0])
        .await
        .expect("upsert m2");

    store
        .add_edge(
            MemoryCategory::Episodic,
            m1.id,
            m2.id,
            relations::PRECEDED_BY,
            0.9,
        )
        .await
        .expect("add_edge");

    let before = store
        .traverse(MemoryCategory::Episodic, m1.id, 1, None)
        .await
        .expect("traverse before");
    assert!(
        before.iter().any(|h| h.target_id == m2.id),
        "precondition: edge m1->m2 must exist"
    );

    // Re-upsert m1 with a changed state (same id; importance bumped).
    let mut updated = m1.clone();
    updated.importance = 0.95;
    store
        .upsert_memory(&updated, vec![1.0, 0.0, 0.0, 0.0])
        .await
        .expect("re-upsert m1");

    let after = store
        .traverse(MemoryCategory::Episodic, m1.id, 1, None)
        .await
        .expect("traverse after");
    assert!(
        after.iter().any(|h| h.target_id == m2.id),
        "state-only re-upsert wiped the edge m1->m2 (edge-preservation regressed)"
    );
}

#[tokio::test]
async fn get_vectors_empty_for_unknown_ids() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut store = EmbeddedStore::new(tmp.path().to_path_buf(), 4);
    store.initialize().await.expect("initialize");

    let fetched = store
        .get_vectors(MemoryCategory::Semantic, &[1, 2, 3])
        .await
        .expect("get_vectors");
    assert!(fetched.is_empty(), "no points ⇒ empty result");
}

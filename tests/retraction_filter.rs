//! `[retrieval_filter]`: a memory superseded by an LLM-confirmed
//! contradiction is hidden from recall, even when its superseder is not
//! among the candidates (a top-1 recall whose query is the old text: the
//! single candidate is the old memory).
//!
//! Mock justification: `ChatProvider::entailment` is the LLM port; the
//! supersede path only runs on an LLM `Contradicts` verdict, so the test
//! supplies one. Store and embedder are the real embedded store and the
//! deterministic test embedder.

mod common;

use async_trait::async_trait;
use lattice_mcp::config::{ContradictionConfig, RetrievalFilterConfig};
use lattice_mcp::embedding::chat::ChatProvider;
use lattice_mcp::embedding::chat_types::EntailmentVerdict;
use lattice_mcp::error::Result;
use lattice_mcp::memory::types::{Memory, MemoryCategory};
use lattice_mcp::memory::MemorySystem;
use lattice_mcp::store::embedded::EmbeddedStore;

use common::config::make_test_config;
use common::mock_embedder::MockEmbedder;

struct AlwaysContradicts;

#[async_trait]
impl ChatProvider for AlwaysContradicts {
    async fn summarize(&self, memories: &[Memory]) -> Result<String> {
        Ok(memories
            .first()
            .map(|m| m.content.clone())
            .unwrap_or_default())
    }
    async fn entailment(&self, _e: &str, _c: &str) -> Result<Option<EntailmentVerdict>> {
        Ok(Some(EntailmentVerdict::Contradicts))
    }
}

const OLD: &str = "auth hashes every password with bcrypt in the login api";
const NEW: &str = "auth no longer uses bcrypt for password";
const OTHER: &str = "the database schema keeps users in one table";

async fn open(dir: &std::path::Path, filter: bool) -> MemorySystem {
    let mut cfg = make_test_config(dir);
    cfg.contradiction = Some(ContradictionConfig {
        enabled: true,
        min_similarity: 0.5,
        auto_supersede: true,
    });
    cfg.retrieval_filter = filter.then_some(RetrievalFilterConfig { enabled: true });
    let mut s = MemorySystem::new(
        Box::new(EmbeddedStore::new(dir.to_path_buf(), 64)),
        Box::new(MockEmbedder),
        Some(Box::new(AlwaysContradicts)),
        None,
        cfg,
    );
    s.initialize().await.unwrap();
    s
}

/// Store OLD and OTHER, then NEW, which supersedes OLD.
async fn supersede(s: &mut MemorySystem) -> (u64, u64) {
    let sem = MemoryCategory::Semantic;
    let old = s
        .remember(OLD, sem, Some(0.7), None, None, None)
        .await
        .unwrap();
    s.remember(OTHER, sem, Some(0.7), None, None, None)
        .await
        .unwrap();
    let new = s
        .remember(NEW, sem, Some(0.7), None, None, None)
        .await
        .unwrap();
    (old.id, new.id)
}

/// Top-1 recall: the candidate set is the single nearest memory.
async fn top1(s: &mut MemorySystem, q: &str) -> Vec<u64> {
    let hits = s.recall(q, &[MemoryCategory::Semantic], 1).await.unwrap();
    hits.iter().map(|h| h.memory.id).collect()
}

#[tokio::test]
async fn superseded_memory_is_hidden_when_its_superseder_is_not_a_candidate() {
    let tmp = tempfile::tempdir().unwrap();
    let mut s = open(tmp.path(), true).await;
    let (old, new) = supersede(&mut s).await;

    let stored = s
        .get_memory(MemoryCategory::Semantic, old)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        stored.superseded_by(),
        Some(new),
        "the mark is on the old memory"
    );
    // The only candidate is OLD; NEW never enters the candidate set.
    assert_eq!(top1(&mut s, OLD).await, Vec::<u64>::new());

    // Unrelated memories are untouched.
    let other = Memory::content_id(OTHER, MemoryCategory::Semantic);
    assert_eq!(top1(&mut s, OTHER).await, vec![other]);
}

#[tokio::test]
async fn the_same_marked_store_returns_it_when_the_filter_is_off() {
    let tmp = tempfile::tempdir().unwrap();
    let old = {
        let mut s = open(tmp.path(), true).await;
        supersede(&mut s).await.0
    };
    let mut s = open(tmp.path(), false).await;
    let stored = s
        .get_memory(MemoryCategory::Semantic, old)
        .await
        .unwrap()
        .unwrap();
    assert!(
        stored.superseded_by().is_some(),
        "mark persisted across reopen"
    );
    assert_eq!(top1(&mut s, OLD).await, vec![old]);
}

#[tokio::test]
async fn filter_off_writes_no_mark_and_leaves_payload_unchanged() {
    let tmp = tempfile::tempdir().unwrap();
    let mut s = open(tmp.path(), false).await;
    let (old, _) = supersede(&mut s).await;
    let stored = s
        .get_memory(MemoryCategory::Semantic, old)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored.superseded_by(), None);
    let meta = String::from_utf8(stored.to_payload()["metadata"].clone()).unwrap();
    assert!(!meta.contains("superseded_by"), "{meta}");
    assert_eq!(top1(&mut s, OLD).await, vec![old]);
}

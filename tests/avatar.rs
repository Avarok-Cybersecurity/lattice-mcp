//! Integration tests for the persistent user-avatar subsystem.

mod common;

use lattice_mcp::config::{AvatarConfig, Config, DatabaseMode};
use lattice_mcp::memory::types::{MemoryCategory, MemoryMetadata};
use lattice_mcp::memory::MemorySystem;
use lattice_mcp::store::embedded::EmbeddedStore;

use common::config::make_test_config;
use common::mock_embedder::MockEmbedder;

fn avatar_config(dir: &std::path::Path, auto_derive: bool, redact: Vec<String>) -> Config {
    let mut cfg = make_test_config(dir);
    cfg.avatar = Some(AvatarConfig {
        enabled: true,
        allow_freeform: false,
        redact_patterns: redact,
        auto_derive,
        allow_remote: false,
    });
    cfg
}

async fn system(dir: &std::path::Path, cfg: Config) -> MemorySystem {
    let store = EmbeddedStore::new(dir.to_path_buf(), 64);
    let mut sys = MemorySystem::new(Box::new(store), Box::new(MockEmbedder), None, None, cfg);
    sys.initialize().await.expect("init");
    sys
}

fn slot_of(m: &lattice_mcp::memory::types::Memory) -> (&str, &str, f32, u32) {
    match &m.metadata {
        MemoryMetadata::Avatar {
            slot_key,
            value,
            confidence,
            evidence_count,
            ..
        } => (slot_key, value, *confidence, *evidence_count),
        _ => panic!("expected Avatar metadata"),
    }
}

#[tokio::test]
async fn store_get_search_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let mut sys = system(tmp.path(), avatar_config(tmp.path(), false, vec![])).await;

    sys.avatar_store("default", "preferred_language", "Rust", &[])
        .await
        .unwrap();
    sys.avatar_store("default", "communication_tone", "concise", &[])
        .await
        .unwrap();

    let all = sys.avatar_get("default").await.unwrap();
    assert_eq!(all.len(), 2, "two slots stored");
    assert!(all.iter().all(|m| m.category == MemoryCategory::Avatar));

    let hits = sys
        .avatar_search("default", "what language does the user prefer", 5)
        .await
        .unwrap();
    assert!(
        hits.iter().any(|s| s.memory.content.contains("Rust")),
        "search should surface the preferred_language slot"
    );
}

#[tokio::test]
async fn reobservation_upserts_and_sharpens() {
    let tmp = tempfile::tempdir().unwrap();
    let mut sys = system(tmp.path(), avatar_config(tmp.path(), false, vec![])).await;

    let m1 = sys
        .avatar_store("default", "preferred_language", "Rust", &[])
        .await
        .unwrap();
    let (_, _, c1, e1) = slot_of(&m1);
    let m2 = sys
        .avatar_store("default", "preferred_language", "rust", &[])
        .await
        .unwrap();
    let (_, _, c2, e2) = slot_of(&m2);

    assert_eq!(
        m1.id, m2.id,
        "same slot upserts the same point (no duplicate)"
    );
    assert_eq!(
        sys.avatar_get("default").await.unwrap().len(),
        1,
        "no duplicate slot"
    );
    assert!(
        c2 > c1,
        "confidence sharpens on reinforcement: {c1} -> {c2}"
    );
    assert_eq!(e2, e1 + 1, "evidence count increments");
}

#[tokio::test]
async fn changed_value_supersedes_with_audit() {
    let tmp = tempfile::tempdir().unwrap();
    let mut sys = system(tmp.path(), avatar_config(tmp.path(), false, vec![])).await;

    sys.avatar_store("default", "editor", "Vim", &[])
        .await
        .unwrap();
    let updated = sys
        .avatar_store("default", "editor", "Helix", &[])
        .await
        .unwrap();

    match &updated.metadata {
        MemoryMetadata::Avatar {
            value, prior_value, ..
        } => {
            assert_eq!(value, "Helix");
            assert_eq!(
                prior_value.as_deref(),
                Some("Vim"),
                "prior value recorded for audit"
            );
        }
        _ => panic!("expected Avatar"),
    }
    assert_eq!(
        sys.avatar_get("default").await.unwrap().len(),
        1,
        "still one editor slot"
    );
}

#[tokio::test]
async fn multiple_avatars_are_isolated() {
    let tmp = tempfile::tempdir().unwrap();
    let mut sys = system(tmp.path(), avatar_config(tmp.path(), false, vec![])).await;

    sys.avatar_store("default", "timezone", "UTC", &[])
        .await
        .unwrap();
    sys.avatar_store("work", "timezone", "PST", &[])
        .await
        .unwrap();

    let d = sys.avatar_get("default").await.unwrap();
    let w = sys.avatar_get("work").await.unwrap();
    assert_eq!(d.len(), 1);
    assert_eq!(w.len(), 1);
    assert_eq!(slot_of(&d[0]).1, "UTC");
    assert_eq!(slot_of(&w[0]).1, "PST");
}

#[tokio::test]
async fn pii_is_redacted_before_store() {
    let tmp = tempfile::tempdir().unwrap();
    let mut sys = system(tmp.path(), avatar_config(tmp.path(), false, vec![])).await;

    let m = sys
        .avatar_store("default", "goal", "reach me at jane@example.com", &[])
        .await
        .unwrap();
    let (_, value, _, _) = slot_of(&m);
    assert!(
        !value.contains("example.com"),
        "email must be redacted: {value}"
    );
    assert!(value.contains("<redacted:email>"));
}

#[tokio::test]
async fn unknown_slot_rejected_without_freeform() {
    let tmp = tempfile::tempdir().unwrap();
    let mut sys = system(tmp.path(), avatar_config(tmp.path(), false, vec![])).await;
    let r = sys
        .avatar_store("default", "favorite_color", "blue", &[])
        .await;
    assert!(
        r.is_err(),
        "unknown slot must be rejected when allow_freeform=false"
    );
}

#[tokio::test]
async fn disabled_avatar_errors() {
    let tmp = tempfile::tempdir().unwrap();
    // make_test_config leaves avatar = None (disabled).
    let mut sys = system(tmp.path(), make_test_config(tmp.path())).await;
    assert!(sys
        .avatar_store("default", "timezone", "UTC", &[])
        .await
        .is_err());
    assert!(sys.avatar_get("default").await.is_err());
}

#[tokio::test]
async fn fail_fast_on_remote() {
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = avatar_config(tmp.path(), false, vec![]);
    cfg.database.mode = DatabaseMode::Remote;
    cfg.database.remote_url = Some("http://localhost:6333".into());
    assert!(
        cfg.validate().is_err(),
        "avatar.enabled + remote must fail validation by default (strict local-first)"
    );

    // Escape hatch: allow_remote = true permits the avatar on a remote DB.
    if let Some(a) = cfg.avatar.as_mut() {
        a.allow_remote = true;
    }
    assert!(
        cfg.validate().is_ok(),
        "avatar.allow_remote = true must permit a remote database"
    );
}

#[tokio::test]
async fn auto_derive_extracts_explicit_slots_at_session_end() {
    let tmp = tempfile::tempdir().unwrap();
    let mut sys = system(tmp.path(), avatar_config(tmp.path(), true, vec![])).await;

    sys.session_start(Some("proj"), Some("setup"))
        .await
        .unwrap();
    sys.remember(
        "Note: my preferred language is Rust and timezone: UTC for this project",
        MemoryCategory::Episodic,
        Some(0.6),
        Some("conversation"),
        None,
        None,
    )
    .await
    .unwrap();
    sys.session_end(Some("done")).await.unwrap();

    let slots = sys.avatar_get("default").await.unwrap();
    let keys: Vec<&str> = slots.iter().map(|m| slot_of(m).0).collect();
    assert!(
        keys.contains(&"preferred_language"),
        "auto-derived language slot, got {keys:?}"
    );
    assert!(
        keys.contains(&"timezone"),
        "auto-derived timezone slot, got {keys:?}"
    );
}

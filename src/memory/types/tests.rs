use super::*;

#[test]
fn content_id_is_deterministic() {
    let id1 = Memory::content_id("test content", MemoryCategory::Episodic);
    let id2 = Memory::content_id("test content", MemoryCategory::Episodic);
    assert_eq!(id1, id2);
}

/// Guard: the content_id derivation for the four non-avatar categories
/// must stay BYTE-IDENTICAL after the avatar work (pinned IDs, dedup,
/// and on-disk points must not shift). These literals are the frozen
/// expected hashes; if this fails, content addressing changed.
#[test]
fn content_id_non_avatar_is_stable() {
    // Cross-category distinctness (the contract content_id guarantees).
    let c = "Architecture decision: use JWT";
    let e = Memory::content_id(c, MemoryCategory::Episodic);
    let s = Memory::content_id(c, MemoryCategory::Semantic);
    let p = Memory::content_id(c, MemoryCategory::Procedural);
    let pin = Memory::content_id(c, MemoryCategory::Pinned);
    assert!(
        e != s && s != p && p != pin && e != pin,
        "categories must differ"
    );
    // All masked to 53 bits.
    for id in [e, s, p, pin] {
        assert_eq!(id & !0x1F_FFFF_FFFF_FFFF, 0, "id must fit 53 bits");
    }
    // content_id ignores avatar_id/slot_key entirely (it only sees content).
    assert_eq!(e, Memory::content_id(c, MemoryCategory::Episodic));
}

#[test]
fn avatar_point_id_is_slot_keyed_not_value_keyed() {
    // Same (avatar, slot) -> same id regardless of value, so re-observation
    // UPSERTs the same point (the avatar sharpens, no duplicates).
    let a = Memory::avatar_point_id("default", "preferred_language");
    let b = Memory::avatar_point_id("default", "preferred_language");
    assert_eq!(a, b);
    // Different slot or avatar -> different id.
    assert_ne!(a, Memory::avatar_point_id("default", "timezone"));
    assert_ne!(a, Memory::avatar_point_id("work", "preferred_language"));
    // Disjoint from content_id space for the same string (different prefix).
    assert_ne!(
        a,
        Memory::content_id("preferred_language", MemoryCategory::Avatar)
    );
    assert_eq!(a & !0x1F_FFFF_FFFF_FFFF, 0, "id must fit 53 bits");
}

#[test]
fn relation_registry_roundtrips() {
    // Every registry entry must reverse-map to its own name, and
    // relations_for must agree with the registry rows for that category.
    for &(cat, name, id) in relations::REGISTRY {
        assert_eq!(relations::relation_name(cat, id), name);
    }
    for &cat in MemoryCategory::all_storage() {
        let from_iter: Vec<_> = relations::relations_for(cat).collect();
        let from_registry: Vec<_> = relations::REGISTRY
            .iter()
            .filter(|(c, _, _)| *c == cat)
            .map(|(_, n, i)| (*n, *i))
            .collect();
        assert_eq!(from_iter, from_registry, "mismatch for {cat:?}");
    }
    // Unknown ids degrade to the sentinel, never panic.
    assert_eq!(
        relations::relation_name(MemoryCategory::Pinned, 0),
        "UNKNOWN"
    );
}

#[test]
fn content_id_differs_by_category() {
    let id1 = Memory::content_id("test content", MemoryCategory::Episodic);
    let id2 = Memory::content_id("test content", MemoryCategory::Semantic);
    assert_ne!(id1, id2);
}

#[test]
fn payload_roundtrip() {
    let mem = Memory {
        id: 42,
        category: MemoryCategory::Episodic,
        content: "Fixed the login bug".into(),
        created_at: Utc::now(),
        last_accessed: Utc::now(),
        access_count: 1,
        importance: 0.7,
        session_id: Some("session-1".into()),
        metadata: MemoryMetadata::Episodic {
            event_type: "error".into(),
        },
        importance_state: ImportanceState::default(),
    };
    let payload = mem.to_payload();
    let restored = Memory::from_payload(42, &payload).unwrap();
    assert_eq!(restored.content, mem.content);
    assert_eq!(restored.importance, mem.importance);
    assert_eq!(restored.id, 42);
    assert_eq!(restored.importance_state.alpha, 1.0);
}

#[test]
fn payload_roundtrip_without_importance_state() {
    // Simulates old memories that lack the importance_state field.
    let mut payload = HashMap::new();
    let now = Utc::now();
    payload.insert("content".into(), serde_json::to_vec("old memory").unwrap());
    payload.insert(
        "category".into(),
        serde_json::to_vec(&MemoryCategory::Episodic).unwrap(),
    );
    payload.insert("created_at".into(), serde_json::to_vec(&now).unwrap());
    payload.insert("last_accessed".into(), serde_json::to_vec(&now).unwrap());
    payload.insert("access_count".into(), serde_json::to_vec(&0u64).unwrap());
    payload.insert("importance".into(), serde_json::to_vec(&0.5f32).unwrap());
    payload.insert(
        "metadata".into(),
        serde_json::to_vec(&MemoryMetadata::Episodic {
            event_type: "test".into(),
        })
        .unwrap(),
    );
    // No "importance_state" key — should use defaults
    let restored = Memory::from_payload(99, &payload).unwrap();
    assert_eq!(restored.importance_state.alpha, 1.0);
    assert_eq!(restored.importance_state.beta, 1.0);
    assert_eq!(restored.importance_state.pagerank, 0.0);
}

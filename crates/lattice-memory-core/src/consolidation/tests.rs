use super::*;
use crate::types::{MemoryCategory, MemoryMetadata};
use chrono::Utc;

fn make_memory(content: &str, importance: f32) -> Memory {
    Memory {
        id: Memory::content_id(content, MemoryCategory::Episodic),
        category: MemoryCategory::Episodic,
        content: content.to_string(),
        created_at: Utc::now(),
        last_accessed: Utc::now(),
        access_count: 0,
        importance,
        session_id: None,
        metadata: MemoryMetadata::Episodic {
            event_type: "test".into(),
        },
        importance_state: Default::default(),
    }
}

#[test]
fn summarize_single() {
    let mems = vec![make_memory("Fixed auth bug", 0.8)];
    let summary = algorithmic_summarize(&mems);
    assert_eq!(summary, "Fixed auth bug");
}

#[test]
fn summarize_empty() {
    let summary = algorithmic_summarize(&[]);
    assert!(summary.is_empty());
}

#[test]
fn summarize_multiple_shared_terms() {
    let mems = vec![
        make_memory("Fixed authentication error in login module", 0.9),
        make_memory("Authentication token validation error found", 0.7),
        make_memory("Login authentication flow needs refactoring", 0.6),
    ];
    let summary = algorithmic_summarize(&mems);
    assert!(summary.contains("3 memories"));
    assert!(summary.contains("authentication"));
}

fn default_params() -> ClusterParams {
    ClusterParams {
        min_overlap: 1,
        min_importance: 0.0,
        temporal_half_life_days: 365.0, // large → negligible temporal effect
    }
}

fn strict_params() -> ClusterParams {
    ClusterParams {
        min_overlap: 2,
        min_importance: 0.4,
        temporal_half_life_days: 7.0,
    }
}

#[test]
fn detect_clusters_finds_related() {
    let mems = vec![
        make_memory("Fixed authentication error in login", 0.8),
        make_memory("Authentication token validation fixed", 0.7),
        make_memory("Database migration script updated", 0.5),
        make_memory("Database schema migration completed", 0.6),
    ];
    let clusters = detect_clusters(&mems, default_params());
    assert_eq!(clusters.len(), 2);
}

#[test]
fn detect_clusters_no_overlap() {
    let mems = vec![
        make_memory("alpha bravo charlie", 0.5),
        make_memory("delta echo foxtrot", 0.5),
    ];
    let clusters = detect_clusters(&mems, default_params());
    assert!(clusters.is_empty());
}

#[test]
fn prefilter_excludes_low_importance() {
    let mems = vec![
        make_memory("Session started for project lattice-mcp", 0.3),
        make_memory("Session started for project lattice-mcp testing", 0.3),
        make_memory("Fixed authentication error in lattice-mcp", 0.8),
    ];
    // With min_importance=0.4, the two 0.3 session-starts are excluded
    let clusters = detect_clusters(&mems, strict_params());
    assert!(clusters.is_empty());
}

#[test]
fn prefilter_keeps_high_importance() {
    let mems = vec![
        make_memory("Fixed authentication error in lattice-mcp login", 0.8),
        make_memory("Authentication token validation in lattice-mcp fixed", 0.7),
    ];
    let clusters = detect_clusters(&mems, strict_params());
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].len(), 2);
}

fn make_memory_at(content: &str, importance: f32, days_ago: i64) -> Memory {
    use chrono::Duration;
    let created = Utc::now() - Duration::days(days_ago);
    Memory {
        id: Memory::content_id(content, MemoryCategory::Episodic),
        category: MemoryCategory::Episodic,
        content: content.to_string(),
        created_at: created,
        last_accessed: created,
        access_count: 0,
        importance,
        session_id: None,
        metadata: MemoryMetadata::Episodic {
            event_type: "test".into(),
        },
        importance_state: Default::default(),
    }
}

#[test]
fn temporal_proximity_boosts_clustering() {
    // Same day, 2 shared terms → should cluster
    let mems = vec![
        make_memory_at("Fixed authentication error in login", 0.8, 0),
        make_memory_at("Authentication error found in tests", 0.7, 0),
    ];
    let params = ClusterParams {
        min_overlap: 2,
        min_importance: 0.0,
        temporal_half_life_days: 7.0,
    };
    let clusters = detect_clusters(&mems, params);
    assert_eq!(clusters.len(), 1);
}

#[test]
fn temporal_distance_weakens_clustering() {
    // 60 days apart with half_life=7 → temporal_weight ≈ 0.10
    // 2 shared terms × 0.10 = 0.20 < 2.0 threshold → should NOT cluster
    let mems = vec![
        make_memory_at("Fixed authentication error in login", 0.8, 0),
        make_memory_at("Authentication error found in tests", 0.7, 60),
    ];
    let params = ClusterParams {
        min_overlap: 2,
        min_importance: 0.0,
        temporal_half_life_days: 7.0,
    };
    let clusters = detect_clusters(&mems, params);
    assert!(clusters.is_empty());
}

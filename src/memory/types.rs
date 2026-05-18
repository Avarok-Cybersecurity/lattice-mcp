use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::importance_state::ImportanceState;

/// Memory category — maps 1:1 to a LatticeDB collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MemoryCategory {
    Episodic,
    Semantic,
    Procedural,
    Pinned,
}

impl MemoryCategory {
    pub fn collection_name(&self) -> &'static str {
        match self {
            Self::Episodic => "episodic",
            Self::Semantic => "semantic",
            Self::Procedural => "procedural",
            Self::Pinned => "pinned",
        }
    }

    pub fn all() -> &'static [MemoryCategory] {
        &[Self::Episodic, Self::Semantic, Self::Procedural, Self::Pinned]
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "episodic" => Some(Self::Episodic),
            "semantic" => Some(Self::Semantic),
            "procedural" => Some(Self::Procedural),
            "pinned" => Some(Self::Pinned),
            _ => None,
        }
    }
}

/// Relation constants — each maps to a u16 ID in LatticeDB's relation registry.
pub mod relations {
    // Episodic relations
    pub const PRECEDED_BY: &str = "PRECEDED_BY";
    pub const CAUSED: &str = "CAUSED";
    pub const RESOLVED_BY: &str = "RESOLVED_BY";

    // Semantic relations
    pub const RELATES_TO: &str = "RELATES_TO";
    pub const DEPENDS_ON: &str = "DEPENDS_ON";
    pub const CONTRADICTS: &str = "CONTRADICTS";
    pub const SUPERSEDES: &str = "SUPERSEDES";

    // Procedural relations
    pub const APPLIES_TO: &str = "APPLIES_TO";
    pub const ALTERNATIVE_TO: &str = "ALTERNATIVE_TO";
    pub const REQUIRES: &str = "REQUIRES";

    // Cross-category (stored in the target collection)
    pub const LEARNED_FROM: &str = "LEARNED_FROM";
    pub const EVIDENCED_BY: &str = "EVIDENCED_BY";
    pub const USES_KNOWLEDGE: &str = "USES_KNOWLEDGE";
}

/// Core memory record — the authoritative in-flight representation (SSOT).
/// Persisted as Point payload fields in LatticeDB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: u64,
    pub category: MemoryCategory,
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u64,
    pub importance: f32,
    pub session_id: Option<String>,
    pub metadata: MemoryMetadata,
    /// Adaptive importance state for all 7 scoring algorithms.
    /// Old memories without this field deserialize to defaults.
    #[serde(default)]
    pub importance_state: ImportanceState,
}

/// Category-specific metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MemoryMetadata {
    Episodic {
        event_type: String,
    },
    Semantic {
        entity_type: String,
        confidence: f32,
    },
    Procedural {
        pattern_type: String,
        trigger: String,
        success_rate: f32,
        usage_count: u64,
    },
    Pinned {
        label: String,
    },
}

impl Memory {
    /// Deterministic point ID from content + category.
    /// Uses xxh3 so the same content always gets the same ID.
    pub fn content_id(content: &str, category: MemoryCategory) -> u64 {
        use xxhash_rust::xxh3::xxh3_64;
        let mut input = category.collection_name().as_bytes().to_vec();
        input.push(b':');
        input.extend_from_slice(content.as_bytes());
        // Mask to 53 bits so IDs survive JSON round-trips through JavaScript
        // runtimes (Number.MAX_SAFE_INTEGER = 2^53 - 1).
        xxh3_64(&input) & 0x1F_FFFF_FFFF_FFFF
    }

    /// Serialize into a LatticeDB point payload (HashMap<String, Vec<u8>>).
    pub fn to_payload(&self) -> HashMap<String, Vec<u8>> {
        let mut payload = HashMap::new();
        payload.insert(
            "content".into(),
            serde_json::to_vec(&self.content).expect("content serialization"),
        );
        payload.insert(
            "category".into(),
            serde_json::to_vec(&self.category).expect("category serialization"),
        );
        payload.insert(
            "created_at".into(),
            serde_json::to_vec(&self.created_at).expect("created_at serialization"),
        );
        payload.insert(
            "last_accessed".into(),
            serde_json::to_vec(&self.last_accessed).expect("last_accessed serialization"),
        );
        payload.insert(
            "access_count".into(),
            serde_json::to_vec(&self.access_count).expect("access_count serialization"),
        );
        payload.insert(
            "importance".into(),
            serde_json::to_vec(&self.importance).expect("importance serialization"),
        );
        if let Some(ref sid) = self.session_id {
            payload.insert(
                "session_id".into(),
                serde_json::to_vec(sid).expect("session_id serialization"),
            );
        }
        payload.insert(
            "metadata".into(),
            serde_json::to_vec(&self.metadata).expect("metadata serialization"),
        );
        payload.insert(
            "importance_state".into(),
            serde_json::to_vec(&self.importance_state).expect("importance_state serialization"),
        );
        payload
    }

    /// Deserialize from a LatticeDB point payload.
    pub fn from_payload(id: u64, payload: &HashMap<String, Vec<u8>>) -> crate::error::Result<Self> {
        let get = |key: &str| -> crate::error::Result<&Vec<u8>> {
            payload.get(key).ok_or_else(|| crate::error::McpMemoryError::Store {
                message: format!("missing payload field: {key}"),
            })
        };

        Ok(Self {
            id,
            category: serde_json::from_slice(get("category")?)?,
            content: serde_json::from_slice(get("content")?)?,
            created_at: serde_json::from_slice(get("created_at")?)?,
            last_accessed: serde_json::from_slice(get("last_accessed")?)?,
            access_count: serde_json::from_slice(get("access_count")?)?,
            importance: serde_json::from_slice(get("importance")?)?,
            session_id: payload
                .get("session_id")
                .map(|v| serde_json::from_slice(v))
                .transpose()?,
            metadata: serde_json::from_slice(get("metadata")?)?,
            importance_state: payload
                .get("importance_state")
                .map(|v| serde_json::from_slice(v))
                .transpose()?
                .unwrap_or_default(),
        })
    }
}

/// A hit from vector search before full scoring.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub id: u64,
    pub score: f32,
    pub category: MemoryCategory,
}

/// A hit from graph traversal.
#[derive(Debug, Clone)]
pub struct TraversalHit {
    pub target_id: u64,
    pub depth: usize,
    pub relation: String,
}

/// Fully scored retrieval result with graph context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredMemory {
    pub memory: Memory,
    pub vector_similarity: f32,
    pub graph_relevance: f32,
    pub temporal_recency: f32,
    pub importance_score: f32,
    pub final_score: f32,
    pub related: Vec<RelatedMemory>,
}

/// A memory discovered via graph traversal (context).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedMemory {
    pub memory: Memory,
    pub relation: String,
    pub hop_distance: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_id_is_deterministic() {
        let id1 = Memory::content_id("test content", MemoryCategory::Episodic);
        let id2 = Memory::content_id("test content", MemoryCategory::Episodic);
        assert_eq!(id1, id2);
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
        payload.insert("category".into(), serde_json::to_vec(&MemoryCategory::Episodic).unwrap());
        payload.insert("created_at".into(), serde_json::to_vec(&now).unwrap());
        payload.insert("last_accessed".into(), serde_json::to_vec(&now).unwrap());
        payload.insert("access_count".into(), serde_json::to_vec(&0u64).unwrap());
        payload.insert("importance".into(), serde_json::to_vec(&0.5f32).unwrap());
        payload.insert("metadata".into(), serde_json::to_vec(&MemoryMetadata::Episodic {
            event_type: "test".into(),
        }).unwrap());
        // No "importance_state" key — should use defaults
        let restored = Memory::from_payload(99, &payload).unwrap();
        assert_eq!(restored.importance_state.alpha, 1.0);
        assert_eq!(restored.importance_state.beta, 1.0);
        assert_eq!(restored.importance_state.pagerank, 0.0);
    }
}

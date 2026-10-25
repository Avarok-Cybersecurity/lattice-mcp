use chrono::{DateTime, Utc};
#[cfg(feature = "schema")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::importance_state::ImportanceState;
use crate::prelude::*;

/// A persisted point's payload: field name → JSON bytes. With `std` this is
/// std's hash map, the type LatticeDB points carry; without it a `BTreeMap`.
/// Only keyed access is used, so the two behave the same.
#[cfg(feature = "std")]
pub type Payload = std::collections::HashMap<String, Vec<u8>>;
#[cfg(not(feature = "std"))]
pub type Payload = alloc::collections::BTreeMap<String, Vec<u8>>;

/// Memory category — maps 1:1 to a LatticeDB collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum MemoryCategory {
    Episodic,
    Semantic,
    Procedural,
    Pinned,
    /// Persistent user model. Slot-keyed, cross-session; accessed via the
    /// avatar_* tools, NOT via remember/recall-default. Opt-in feature.
    Avatar,
}

impl MemoryCategory {
    pub fn collection_name(&self) -> &'static str {
        match self {
            Self::Episodic => "episodic",
            Self::Semantic => "semantic",
            Self::Procedural => "procedural",
            Self::Pinned => "pinned",
            Self::Avatar => "avatar",
        }
    }

    /// The base memory categories used by generic recall/reflect/stats.
    /// Avatar is intentionally EXCLUDED so the user model never leaks into
    /// ordinary task recall — it is queried explicitly via avatar_search.
    pub fn all() -> &'static [MemoryCategory] {
        &[
            Self::Episodic,
            Self::Semantic,
            Self::Procedural,
            Self::Pinned,
        ]
    }

    /// Every category that backs a physical collection (includes Avatar).
    /// Used for store initialization and total counts.
    pub fn all_storage() -> &'static [MemoryCategory] {
        &[
            Self::Episodic,
            Self::Semantic,
            Self::Procedural,
            Self::Pinned,
            Self::Avatar,
        ]
    }

    // Intentionally inherent (not the `FromStr` trait): returns Option, no
    // error type, used as a lightweight category parser at call sites.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "episodic" => Some(Self::Episodic),
            "semantic" => Some(Self::Semantic),
            "procedural" => Some(Self::Procedural),
            "pinned" => Some(Self::Pinned),
            "avatar" => Some(Self::Avatar),
            _ => None,
        }
    }
}

pub mod relations;

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
        /// Source memory IDs this knowledge was consolidated from (Step 7).
        /// Empty for non-consolidated semantic memories. `#[serde(default)]`
        /// keeps pre-provenance payloads deserializable.
        #[serde(default)]
        provenance: Vec<u64>,
        /// The memory that superseded this one, written with the
        /// `SUPERSEDES` edge while `[retrieval_filter]` is enabled. Omitted
        /// from the payload when `None`, so unmarked payloads are unchanged.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        superseded_by: Option<u64>,
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
    /// Persistent user-model slot (see `memory::avatar`).
    Avatar {
        /// Which avatar this belongs to (default "default"); enables multi-avatar.
        avatar_id: String,
        /// Curated slot key, e.g. "preferred_language" (or freeform if allowed).
        slot_key: String,
        /// Authoritative (redacted) slot value. The memory `content` is the
        /// derived search projection `render(slot_key, value)` — SSOT is here.
        value: String,
        /// Shape of the slot value.
        value_kind: ValueKind,
        /// Belief strength in [0,1] = α/(α+β) from the Bayesian state; never
        /// hand-set — it sharpens as evidence accumulates.
        confidence: f32,
        /// Source memory/interaction IDs this slot was derived from.
        #[serde(default)]
        provenance: Vec<u64>,
        /// How many observations have reinforced this slot.
        evidence_count: u32,
        /// The immediately-prior value this one replaced (one-step reversible
        /// audit trail for confidence-gated last-writer-wins). `None` on first
        /// write or reinforcement.
        #[serde(default)]
        prior_value: Option<String>,
    },
}

/// Shape of an avatar slot's value.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ValueKind {
    /// A single stable value (e.g. timezone).
    Scalar,
    /// One choice from a known set (e.g. communication_tone).
    Enum,
    /// Multiple values (e.g. tooling).
    List,
    /// Unstructured free text.
    FreeText,
}

impl Memory {
    /// The id of the memory that superseded this one, if it was marked.
    pub fn superseded_by(&self) -> Option<u64> {
        match self.metadata {
            MemoryMetadata::Semantic { superseded_by, .. } => superseded_by,
            _ => None,
        }
    }

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

    /// Deterministic point ID for an avatar slot.
    ///
    /// Hashes `collection + avatar_id + slot_key` — NOT the value — so that
    /// re-observing the same slot UPSERTs the same point and the avatar
    /// sharpens over time instead of accreting duplicates. This is the
    /// guarded counterpart to `content_id`: non-avatar IDs stay byte-identical.
    pub fn avatar_point_id(avatar_id: &str, slot_key: &str) -> u64 {
        use xxhash_rust::xxh3::xxh3_64;
        let mut input = MemoryCategory::Avatar.collection_name().as_bytes().to_vec();
        input.push(b':');
        input.extend_from_slice(avatar_id.as_bytes());
        input.push(b':');
        input.extend_from_slice(slot_key.as_bytes());
        xxh3_64(&input) & 0x1F_FFFF_FFFF_FFFF
    }

    /// Serialize into a LatticeDB point payload (field name → JSON bytes).
    pub fn to_payload(&self) -> Payload {
        let mut payload = Payload::new();
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
    pub fn from_payload(id: u64, payload: &Payload) -> crate::error::Result<Self> {
        let get = |key: &str| -> crate::error::Result<&Vec<u8>> {
            payload
                .get(key)
                .ok_or_else(|| crate::error::MemoryCoreError::MissingPayloadField {
                    key: key.to_string(),
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
mod tests;

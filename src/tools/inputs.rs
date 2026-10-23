//! MCP tool input parameter types (extracted from mod.rs for size).

use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RememberInput {
    /// The memory content to store
    pub content: String,
    /// Category: "episodic", "semantic", or "procedural" (auto-detected if omitted)
    pub category: Option<String>,
    /// Importance score 0.0-1.0 (auto-computed if omitted)
    pub importance: Option<f32>,
    /// Event type (for episodic: "decision", "error", "change", "conversation")
    pub event_type: Option<String>,
    /// Point IDs of related memories to link to
    pub related_to: Option<Vec<u64>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecallInput {
    /// Search query — finds semantically similar memories
    pub query: String,
    /// Filter by categories (e.g. ["episodic", "semantic"]). Searches all if omitted.
    pub categories: Option<Vec<String>>,
    /// Max results to return
    pub limit: Option<usize>,
    /// Only return memories created on/after this RFC3339 timestamp (e.g. "2026-01-01T00:00:00Z").
    pub after_date: Option<String>,
    /// Only return memories created on/before this RFC3339 timestamp.
    pub before_date: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SessionStartInput {
    /// Project name for context
    pub project: Option<String>,
    /// Description of the task being started
    pub task_description: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SessionEndInput {
    /// Brief summary of what was accomplished
    pub summary: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetContextInput {
    /// Query for relevant context
    pub query: String,
    /// Max results
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReflectInput {
    /// Category to consolidate (default: episodic)
    pub category: Option<String>,
    /// Topic to focus consolidation on
    pub topic: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ConnectInput {
    /// Source memory ID
    pub from_id: u64,
    /// Target memory ID
    pub to_id: u64,
    /// Relation type (e.g. "RELATES_TO", "CAUSED", "DEPENDS_ON")
    pub relation: String,
    /// Edge weight 0.0-1.0 (default: 0.8)
    pub weight: Option<f32>,
    /// Category of the source memory
    pub category: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ForgetInput {
    /// Memory ID to forget
    pub memory_id: u64,
    /// Category of the memory
    pub category: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpdateMemoryInput {
    /// Memory ID to update
    pub memory_id: u64,
    /// Category of the memory
    pub category: String,
    /// New content (if updating)
    pub content: Option<String>,
    /// New importance score (if updating)
    pub importance: Option<f32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchGraphInput {
    /// Starting memory ID
    pub start_id: u64,
    /// Category of the starting memory
    pub category: String,
    /// Max traversal depth (default: 2)
    pub max_depth: Option<usize>,
    /// Filter by relation types
    pub relations: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct PinInput {
    /// The content to pin as a persistent reminder
    pub content: String,
    /// Optional label for the pinned memory
    pub label: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UnpinInput {
    /// Memory ID of the pinned reminder to remove
    pub memory_id: u64,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AvatarStoreInput {
    /// Trait/preference slot, e.g. "preferred_language", "communication_tone".
    pub slot_key: String,
    /// The observed value for the slot (PII is redacted before storage).
    pub value: String,
    /// Which avatar to update (defaults to "default"). Use distinct ids for
    /// multiple personas/users.
    pub avatar_id: Option<String>,
    /// Optional source memory IDs this observation was derived from.
    pub provenance: Option<Vec<u64>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AvatarSearchInput {
    /// What to look up about the user, e.g. "how does the user like code reviewed".
    pub query: String,
    /// Which avatar to search (defaults to "default").
    pub avatar_id: Option<String>,
    /// Max results (default 10).
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AvatarGetInput {
    /// Which avatar to dump (defaults to "default").
    pub avatar_id: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct PruneInput {
    /// Category to prune: "episodic", "semantic", or "procedural".
    /// Omit to prune all three (pinned memories are never pruned).
    pub category: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RateRecallInput {
    /// Memory ID that was recalled
    pub memory_id: u64,
    /// Category of the memory
    pub category: String,
    /// Recall quality (0-5): 0=blackout, 1-2=failed, 3=hard, 4=good, 5=perfect
    pub quality: u8,
}

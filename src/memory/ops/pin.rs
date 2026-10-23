//! Pinned-memory operations.

use tracing::info;

use crate::error::{McpMemoryError, Result};
use crate::memory::importance_state;
use crate::memory::types::*;
use crate::memory::MemorySystem;

pub(crate) async fn pin(
    sys: &mut MemorySystem,
    content: &str,
    label: Option<&str>,
) -> Result<Memory> {
    let max_len = sys.config.memory.max_pinned_content_length;
    if content.len() > max_len {
        return Err(McpMemoryError::PinLimit {
            message: format!("Content length {} exceeds max {max_len}", content.len()),
        });
    }

    let existing = sys.store.list_all(MemoryCategory::Pinned).await?;
    let max_count = sys.config.memory.max_pinned_memories;
    if existing.len() >= max_count {
        return Err(McpMemoryError::PinLimit {
            message: format!("Already at max pinned memories ({max_count}). Unpin one first."),
        });
    }

    let now = sys.clock.now();
    let id = Memory::content_id(content, MemoryCategory::Pinned);
    let label_str = label.unwrap_or("pinned").to_string();

    let memory = Memory {
        id,
        category: MemoryCategory::Pinned,
        content: content.to_string(),
        created_at: now,
        last_accessed: now,
        access_count: 0,
        importance: 1.0,
        session_id: sys.current_session_id.clone(),
        metadata: MemoryMetadata::Pinned { label: label_str },
        importance_state: importance_state::ImportanceState::default(),
    };

    let zero_vec = vec![0.0_f32; sys.config.embedding.dimensions];
    sys.store.upsert_memory(&memory, zero_vec).await?;

    info!(id, "Pinned memory stored");
    Ok(memory)
}

pub(crate) async fn unpin(sys: &mut MemorySystem, memory_id: u64) -> Result<bool> {
    sys.store
        .delete_memory(MemoryCategory::Pinned, memory_id)
        .await
}

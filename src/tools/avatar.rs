//! MCP handlers for the persistent user-avatar tools.
//!
//! `avatar_id` is optional on every tool and resolved to "default" here at
//! the boundary (the core always takes an explicit id), so multi-avatar is
//! available without burdening single-avatar callers.

use rmcp::model::CallToolResult;
use rmcp::ErrorData as McpError;

use super::{
    error_result, json_result, AvatarGetInput, AvatarSearchInput, AvatarStoreInput, MemoryServer,
};
use crate::memory::ops::avatar::DEFAULT_AVATAR;

pub async fn handle_store(
    server: &MemoryServer,
    input: AvatarStoreInput,
) -> Result<CallToolResult, McpError> {
    let avatar_id = input.avatar_id.as_deref().unwrap_or(DEFAULT_AVATAR);
    let provenance = input.provenance.unwrap_or_default();
    let mut mem = server.memory.lock().await;
    match mem
        .avatar_store(avatar_id, &input.slot_key, &input.value, &provenance)
        .await
    {
        Ok(m) => Ok(json_result(&m)),
        Err(e) => error_result(&format!("avatar_store failed: {e}")),
    }
}

pub async fn handle_search(
    server: &MemoryServer,
    input: AvatarSearchInput,
) -> Result<CallToolResult, McpError> {
    let avatar_id = input.avatar_id.as_deref().unwrap_or(DEFAULT_AVATAR);
    let limit = input.limit.unwrap_or(10);
    let mut mem = server.memory.lock().await;
    match mem.avatar_search(avatar_id, &input.query, limit).await {
        Ok(results) => Ok(json_result(&results)),
        Err(e) => error_result(&format!("avatar_search failed: {e}")),
    }
}

pub async fn handle_get(
    server: &MemoryServer,
    input: AvatarGetInput,
) -> Result<CallToolResult, McpError> {
    let avatar_id = input.avatar_id.as_deref().unwrap_or(DEFAULT_AVATAR);
    let mem = server.memory.lock().await;
    match mem.avatar_get(avatar_id).await {
        Ok(slots) => Ok(json_result(&slots)),
        Err(e) => error_result(&format!("avatar_get failed: {e}")),
    }
}

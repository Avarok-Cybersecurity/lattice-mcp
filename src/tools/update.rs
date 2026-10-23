use rmcp::model::CallToolResult;
use rmcp::ErrorData as McpError;

use super::{
    error_result, json_result, parse_category, text_result, MemoryServer, UpdateMemoryInput,
};
use crate::memory::types::MemoryCategory;

pub async fn handle(
    server: &MemoryServer,
    input: UpdateMemoryInput,
) -> Result<CallToolResult, McpError> {
    let category = parse_category(&input.category).unwrap_or(MemoryCategory::Episodic);

    let mut mem = server.memory.lock().await;
    match mem
        .update_memory(
            category,
            input.memory_id,
            input.content.as_deref(),
            input.importance,
        )
        .await
    {
        Ok(Some(updated)) => Ok(json_result(&updated)),
        Ok(None) => Ok(text_result(&format!(
            "Memory {} not found in {}",
            input.memory_id,
            category.collection_name()
        ))),
        Err(e) => error_result(&format!("Update failed: {e}")),
    }
}

use rmcp::model::CallToolResult;
use rmcp::ErrorData as McpError;

use super::{error_result, json_result, parse_category, MemoryServer, RememberInput};
use crate::memory::types::MemoryCategory;

pub async fn handle(
    server: &MemoryServer,
    input: RememberInput,
) -> Result<CallToolResult, McpError> {
    let category = input
        .category
        .as_deref()
        .and_then(parse_category)
        .unwrap_or(MemoryCategory::Episodic);

    let mut mem = server.memory.lock().await;
    let related_refs: Option<Vec<u64>> = input.related_to;

    match mem
        .remember(
            &input.content,
            category,
            input.importance,
            input.event_type.as_deref(),
            related_refs.as_deref(),
            None,
        )
        .await
    {
        Ok(memory) => Ok(json_result(&memory)),
        Err(e) => error_result(&format!("Failed to store memory: {e}")),
    }
}

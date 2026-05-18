use rmcp::model::CallToolResult;
use rmcp::ErrorData as McpError;

use super::{error_result, parse_category, text_result, ForgetInput, MemoryServer};
use crate::memory::types::MemoryCategory;

pub async fn handle(
    server: &MemoryServer,
    input: ForgetInput,
) -> Result<CallToolResult, McpError> {
    let category = parse_category(&input.category).unwrap_or(MemoryCategory::Episodic);

    let mut mem = server.memory.lock().await;
    match mem.forget(category, input.memory_id).await {
        Ok(true) => Ok(text_result(&format!("Memory {} deleted", input.memory_id))),
        Ok(false) => Ok(text_result(&format!(
            "Memory {} not found in {}",
            input.memory_id,
            category.collection_name()
        ))),
        Err(e) => error_result(&format!("Forget failed: {e}")),
    }
}

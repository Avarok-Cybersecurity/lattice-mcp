use rmcp::model::CallToolResult;
use rmcp::ErrorData as McpError;

use super::{error_result, text_result, MemoryServer, UnpinInput};

pub async fn handle(server: &MemoryServer, input: UnpinInput) -> Result<CallToolResult, McpError> {
    let mut mem = server.memory.lock().await;
    match mem.unpin(input.memory_id).await {
        Ok(true) => Ok(text_result("Memory unpinned successfully")),
        Ok(false) => error_result(&format!("Pinned memory {} not found", input.memory_id)),
        Err(e) => error_result(&format!("Unpin failed: {e}")),
    }
}

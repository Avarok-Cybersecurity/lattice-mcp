use rmcp::model::CallToolResult;
use rmcp::ErrorData as McpError;

use super::{error_result, json_result, MemoryServer};

pub async fn handle(server: &MemoryServer) -> Result<CallToolResult, McpError> {
    let mem = server.memory.lock().await;
    match mem.list_pinned().await {
        Ok(pinned) => Ok(json_result(&pinned)),
        Err(e) => error_result(&format!("List pinned failed: {e}")),
    }
}

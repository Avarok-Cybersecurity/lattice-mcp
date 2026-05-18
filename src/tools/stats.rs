use rmcp::model::CallToolResult;
use rmcp::ErrorData as McpError;

use super::{error_result, json_result, MemoryServer};

pub async fn handle(server: &MemoryServer) -> Result<CallToolResult, McpError> {
    let mem = server.memory.lock().await;
    match mem.stats().await {
        Ok(stats) => Ok(json_result(&stats)),
        Err(e) => error_result(&format!("Stats failed: {e}")),
    }
}

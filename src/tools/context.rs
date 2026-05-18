use rmcp::model::CallToolResult;
use rmcp::ErrorData as McpError;

use super::{error_result, json_result, GetContextInput, MemoryServer};

pub async fn handle(
    server: &MemoryServer,
    input: GetContextInput,
) -> Result<CallToolResult, McpError> {
    let limit = input.limit.unwrap_or(10);

    let mut mem = server.memory.lock().await;
    match mem.recall(&input.query, &[], limit).await {
        Ok(results) => Ok(json_result(&results)),
        Err(e) => error_result(&format!("Get context failed: {e}")),
    }
}

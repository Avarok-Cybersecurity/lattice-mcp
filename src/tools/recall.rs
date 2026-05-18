use rmcp::model::CallToolResult;
use rmcp::ErrorData as McpError;

use super::{error_result, json_result, parse_categories, MemoryServer, RecallInput};

pub async fn handle(
    server: &MemoryServer,
    input: RecallInput,
) -> Result<CallToolResult, McpError> {
    let categories = parse_categories(&input.categories);
    let limit = input.limit.unwrap_or(10);

    let mut mem = server.memory.lock().await;
    match mem.recall(&input.query, &categories, limit).await {
        Ok(results) => Ok(json_result(&results)),
        Err(e) => error_result(&format!("Recall failed: {e}")),
    }
}

use rmcp::model::CallToolResult;
use rmcp::ErrorData as McpError;

use super::{error_result, json_result, parse_category, MemoryServer, ReflectInput};

pub async fn handle(
    server: &MemoryServer,
    input: ReflectInput,
) -> Result<CallToolResult, McpError> {
    let category = input.category.as_deref().and_then(parse_category);

    let mut mem = server.memory.lock().await;
    match mem.reflect(category, input.topic.as_deref()).await {
        Ok(created) => Ok(json_result(&created)),
        Err(e) => error_result(&format!("Reflection failed: {e}")),
    }
}

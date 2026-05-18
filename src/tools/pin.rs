use rmcp::model::CallToolResult;
use rmcp::ErrorData as McpError;

use super::{error_result, json_result, MemoryServer, PinInput};

pub async fn handle(
    server: &MemoryServer,
    input: PinInput,
) -> Result<CallToolResult, McpError> {
    let mut mem = server.memory.lock().await;
    match mem.pin(&input.content, input.label.as_deref()).await {
        Ok(memory) => Ok(json_result(&memory)),
        Err(e) => error_result(&format!("Pin failed: {e}")),
    }
}

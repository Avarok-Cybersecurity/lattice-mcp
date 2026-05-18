use rmcp::model::CallToolResult;
use rmcp::ErrorData as McpError;

use super::{error_result, parse_category, text_result, ConnectInput, MemoryServer};
use crate::memory::types::MemoryCategory;

pub async fn handle(
    server: &MemoryServer,
    input: ConnectInput,
) -> Result<CallToolResult, McpError> {
    let category = parse_category(&input.category).unwrap_or(MemoryCategory::Episodic);
    let weight = input.weight.unwrap_or(0.8);

    let mut mem = server.memory.lock().await;
    match mem
        .connect(input.from_id, input.to_id, &input.relation, weight, category)
        .await
    {
        Ok(()) => Ok(text_result(&format!(
            "Edge created: {} --[{}]--> {} (weight: {weight})",
            input.from_id, input.relation, input.to_id
        ))),
        Err(e) => error_result(&format!("Connect failed: {e}")),
    }
}

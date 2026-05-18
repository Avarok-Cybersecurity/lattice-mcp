use rmcp::model::CallToolResult;
use rmcp::ErrorData as McpError;

use super::{error_result, json_result, parse_category, MemoryServer, RateRecallInput};
use crate::memory::types::MemoryCategory;

pub async fn handle(
    server: &MemoryServer,
    input: RateRecallInput,
) -> Result<CallToolResult, McpError> {
    let category = parse_category(&input.category).unwrap_or(MemoryCategory::Episodic);
    let quality = input.quality.min(5);

    let mut mem = server.memory.lock().await;
    match mem.rate_recall(category, input.memory_id, quality).await {
        Ok(Some(updated)) => Ok(json_result(&updated)),
        Ok(None) => error_result(&format!(
            "Memory {} not found in {}",
            input.memory_id,
            category.collection_name()
        )),
        Err(e) => error_result(&format!("Rate recall failed: {e}")),
    }
}

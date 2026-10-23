use rmcp::model::CallToolResult;
use rmcp::ErrorData as McpError;

use super::{error_result, json_result, parse_category, MemoryServer, PruneInput};
use crate::memory::types::MemoryCategory;

pub async fn handle(server: &MemoryServer, input: PruneInput) -> Result<CallToolResult, McpError> {
    let cats: Vec<MemoryCategory> = match input.category.as_deref() {
        Some(c) => match parse_category(c) {
            Some(MemoryCategory::Pinned) => {
                return error_result("Pinned memories are never pruned");
            }
            Some(MemoryCategory::Avatar) => {
                return error_result(
                    "Avatar memories are not pruned here; use forget() or supersede",
                );
            }
            Some(cat) => vec![cat],
            None => return error_result(&format!("Unknown category: {c}")),
        },
        None => vec![
            MemoryCategory::Episodic,
            MemoryCategory::Semantic,
            MemoryCategory::Procedural,
        ],
    };

    let mut mem = server.memory.lock().await;
    let mut reports = Vec::new();
    for cat in cats {
        match mem.prune(cat).await {
            Ok(r) => reports.push(r),
            Err(e) => return error_result(&format!("Prune failed: {e}")),
        }
    }
    Ok(json_result(&reports))
}

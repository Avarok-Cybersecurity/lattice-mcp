use rmcp::model::CallToolResult;
use rmcp::ErrorData as McpError;
use serde::Serialize;

use super::{error_result, json_result, MemoryServer, SessionEndInput, SessionStartInput};

#[derive(Serialize)]
struct SessionStartResponse {
    session_id: String,
    context_count: usize,
    context: Vec<crate::memory::types::ScoredMemory>,
}

pub async fn handle_start(
    server: &MemoryServer,
    input: SessionStartInput,
) -> Result<CallToolResult, McpError> {
    let mut mem = server.memory.lock().await;
    match mem
        .session_start(
            input.project.as_deref(),
            input.task_description.as_deref(),
        )
        .await
    {
        Ok((session_id, context)) => {
            let resp = SessionStartResponse {
                session_id,
                context_count: context.len(),
                context,
            };
            Ok(json_result(&resp))
        }
        Err(e) => error_result(&format!("Session start failed: {e}")),
    }
}

pub async fn handle_end(
    server: &MemoryServer,
    input: SessionEndInput,
) -> Result<CallToolResult, McpError> {
    let mut mem = server.memory.lock().await;
    match mem.session_end(input.summary.as_deref()).await {
        Ok(()) => Ok(super::text_result("Session ended successfully")),
        Err(e) => error_result(&format!("Session end failed: {e}")),
    }
}

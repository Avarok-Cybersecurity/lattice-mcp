use chrono::{DateTime, Utc};
use rmcp::model::CallToolResult;
use rmcp::ErrorData as McpError;

use super::{error_result, json_result, parse_categories, MemoryServer, RecallInput};

/// Parse an optional RFC3339 timestamp; Err message on malformed input.
fn parse_ts(s: &Option<String>, field: &str) -> Result<Option<DateTime<Utc>>, String> {
    match s {
        None => Ok(None),
        Some(raw) => DateTime::parse_from_rfc3339(raw)
            .map(|d| Some(d.with_timezone(&Utc)))
            .map_err(|e| format!("{field} must be an RFC3339 timestamp: {e}")),
    }
}

pub async fn handle(server: &MemoryServer, input: RecallInput) -> Result<CallToolResult, McpError> {
    let categories = parse_categories(&input.categories);
    let limit = input.limit.unwrap_or(10);

    let (after, before) = match (
        parse_ts(&input.after_date, "after_date"),
        parse_ts(&input.before_date, "before_date"),
    ) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(m), _) | (_, Err(m)) => return error_result(&m),
    };

    let mut mem = server.memory.lock().await;
    let result = if after.is_some() || before.is_some() {
        mem.recall_in_range(&input.query, &categories, limit, after, before)
            .await
    } else {
        mem.recall(&input.query, &categories, limit).await
    };
    match result {
        Ok(results) => Ok(json_result(&results)),
        Err(e) => error_result(&format!("Recall failed: {e}")),
    }
}

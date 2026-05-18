use rmcp::model::CallToolResult;
use rmcp::ErrorData as McpError;
use serde::Serialize;

use super::{error_result, json_result, parse_category, MemoryServer, SearchGraphInput};
use crate::memory::types::MemoryCategory;

#[derive(Serialize)]
struct GraphResult {
    start_id: u64,
    edges: Vec<GraphEdge>,
}

#[derive(Serialize)]
struct GraphEdge {
    target_id: u64,
    relation: String,
    depth: usize,
    target_content: Option<String>,
}

pub async fn handle(
    server: &MemoryServer,
    input: SearchGraphInput,
) -> Result<CallToolResult, McpError> {
    let category = parse_category(&input.category).unwrap_or(MemoryCategory::Episodic);
    let max_depth = input.max_depth.unwrap_or(2);

    let mem = server.memory.lock().await;

    let rel_strs: Option<Vec<String>> = input.relations;
    let rel_refs: Option<Vec<&str>> = rel_strs
        .as_ref()
        .map(|v| v.iter().map(|s| s.as_str()).collect());

    match mem
        .search_graph(category, input.start_id, max_depth, rel_refs.as_deref())
        .await
    {
        Ok(hits) => {
            let mut edges = Vec::new();
            for hit in hits {
                let content = mem
                    .get_memory(category, hit.target_id)
                    .await
                    .ok()
                    .flatten()
                    .map(|m| m.content);
                edges.push(GraphEdge {
                    target_id: hit.target_id,
                    relation: hit.relation,
                    depth: hit.depth,
                    target_content: content,
                });
            }

            Ok(json_result(&GraphResult {
                start_id: input.start_id,
                edges,
            }))
        }
        Err(e) => error_result(&format!("Graph search failed: {e}")),
    }
}

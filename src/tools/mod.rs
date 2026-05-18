//! MCP tool definitions and the MemoryServer handler.

mod connect;
mod context;
mod forget;
mod graph;
mod list_pinned;
mod pin;
mod rate_recall;
mod recall;
mod reflect;
mod remember;
mod session;
mod stats;
mod unpin;
mod update;

use std::sync::Arc;
use tokio::sync::Mutex;

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::*;
use rmcp::{tool, tool_handler, tool_router, ErrorData as McpError};

use crate::memory::types::MemoryCategory;
use crate::memory::MemorySystem;

use schemars::JsonSchema;
use serde::Deserialize;

// --- Tool input types ---

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RememberInput {
    /// The memory content to store
    pub content: String,
    /// Category: "episodic", "semantic", or "procedural" (auto-detected if omitted)
    pub category: Option<String>,
    /// Importance score 0.0-1.0 (auto-computed if omitted)
    pub importance: Option<f32>,
    /// Event type (for episodic: "decision", "error", "change", "conversation")
    pub event_type: Option<String>,
    /// Point IDs of related memories to link to
    pub related_to: Option<Vec<u64>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecallInput {
    /// Search query — finds semantically similar memories
    pub query: String,
    /// Filter by categories (e.g. ["episodic", "semantic"]). Searches all if omitted.
    pub categories: Option<Vec<String>>,
    /// Max results to return
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SessionStartInput {
    /// Project name for context
    pub project: Option<String>,
    /// Description of the task being started
    pub task_description: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SessionEndInput {
    /// Brief summary of what was accomplished
    pub summary: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetContextInput {
    /// Query for relevant context
    pub query: String,
    /// Max results
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReflectInput {
    /// Category to consolidate (default: episodic)
    pub category: Option<String>,
    /// Topic to focus consolidation on
    pub topic: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ConnectInput {
    /// Source memory ID
    pub from_id: u64,
    /// Target memory ID
    pub to_id: u64,
    /// Relation type (e.g. "RELATES_TO", "CAUSED", "DEPENDS_ON")
    pub relation: String,
    /// Edge weight 0.0-1.0 (default: 0.8)
    pub weight: Option<f32>,
    /// Category of the source memory
    pub category: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ForgetInput {
    /// Memory ID to forget
    pub memory_id: u64,
    /// Category of the memory
    pub category: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpdateMemoryInput {
    /// Memory ID to update
    pub memory_id: u64,
    /// Category of the memory
    pub category: String,
    /// New content (if updating)
    pub content: Option<String>,
    /// New importance score (if updating)
    pub importance: Option<f32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchGraphInput {
    /// Starting memory ID
    pub start_id: u64,
    /// Category of the starting memory
    pub category: String,
    /// Max traversal depth (default: 2)
    pub max_depth: Option<usize>,
    /// Filter by relation types
    pub relations: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct PinInput {
    /// The content to pin as a persistent reminder
    pub content: String,
    /// Optional label for the pinned memory
    pub label: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UnpinInput {
    /// Memory ID of the pinned reminder to remove
    pub memory_id: u64,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RateRecallInput {
    /// Memory ID that was recalled
    pub memory_id: u64,
    /// Category of the memory
    pub category: String,
    /// Recall quality (0-5): 0=blackout, 1-2=failed, 3=hard, 4=good, 5=perfect
    pub quality: u8,
}

// --- MemoryServer ---

#[derive(Clone)]
pub struct MemoryServer {
    memory: Arc<Mutex<MemorySystem>>,
    tool_router: ToolRouter<Self>,
}

fn parse_category(s: &str) -> Option<MemoryCategory> {
    MemoryCategory::from_str(s)
}

fn parse_categories(input: &Option<Vec<String>>) -> Vec<MemoryCategory> {
    match input {
        Some(cats) => cats.iter().filter_map(|s| parse_category(s)).collect(),
        None => Vec::new(),
    }
}

fn text_result(text: &str) -> CallToolResult {
    CallToolResult::success(vec![Content::text(text.to_string())])
}

fn json_result(value: &impl serde::Serialize) -> CallToolResult {
    let json = serde_json::to_string_pretty(value).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"));
    CallToolResult::success(vec![Content::text(json)])
}

fn error_result(msg: &str) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::error(vec![Content::text(msg.to_string())]))
}

fn format_pinned_block(pinned: &[crate::memory::types::Memory]) -> String {
    let mut lines = String::from("\n---\n[Pinned Reminders]\n");
    for m in pinned {
        lines.push_str(&format!("- {}\n", m.content));
    }
    lines
}

#[tool_router]
impl MemoryServer {
    pub fn new(memory: MemorySystem) -> Self {
        Self {
            memory: Arc::new(Mutex::new(memory)),
            tool_router: Self::tool_router(),
        }
    }

    async fn append_pinned(&self, mut result: CallToolResult) -> CallToolResult {
        let mem = self.memory.lock().await;
        match mem.list_pinned().await {
            Ok(pinned) if !pinned.is_empty() => {
                result.content.push(Content::text(format_pinned_block(&pinned)));
                result
            }
            _ => result,
        }
    }

    #[tool(
        name = "remember",
        description = "Store a new memory. Auto-categorizes, generates embedding, and creates graph edges to similar memories. Use after making decisions, fixing bugs, learning preferences, or significant code changes."
    )]
    async fn remember(
        &self,
        input: Parameters<RememberInput>,
    ) -> Result<CallToolResult, McpError> {
        let result = remember::handle(self, input.0).await?;
        Ok(self.append_pinned(result).await)
    }

    #[tool(
        name = "recall",
        description = "Retrieve relevant memories using hybrid vector+graph search. Returns ranked results with context from graph neighborhood. Use before starting tasks to check prior knowledge."
    )]
    async fn recall(
        &self,
        input: Parameters<RecallInput>,
    ) -> Result<CallToolResult, McpError> {
        let result = recall::handle(self, input.0).await?;
        Ok(self.append_pinned(result).await)
    }

    #[tool(
        name = "session_start",
        description = "Begin a new coding session. Loads relevant context from previous sessions. Call at the start of every conversation."
    )]
    async fn session_start(
        &self,
        input: Parameters<SessionStartInput>,
    ) -> Result<CallToolResult, McpError> {
        let result = session::handle_start(self, input.0).await?;
        Ok(self.append_pinned(result).await)
    }

    #[tool(
        name = "session_end",
        description = "End the current coding session. Consolidates episodic memories from this session. Call at the end of every conversation."
    )]
    async fn session_end(
        &self,
        input: Parameters<SessionEndInput>,
    ) -> Result<CallToolResult, McpError> {
        let result = session::handle_end(self, input.0).await?;
        Ok(self.append_pinned(result).await)
    }

    #[tool(
        name = "get_context",
        description = "Get relevant memories for the current task without storing anything. Read-only retrieval."
    )]
    async fn get_context(
        &self,
        input: Parameters<GetContextInput>,
    ) -> Result<CallToolResult, McpError> {
        let result = context::handle(self, input.0).await?;
        Ok(self.append_pinned(result).await)
    }

    #[tool(
        name = "reflect",
        description = "Consolidate related memories. Summarizes clusters of episodic memories into semantic knowledge. Call periodically to strengthen long-term memory."
    )]
    async fn reflect(
        &self,
        input: Parameters<ReflectInput>,
    ) -> Result<CallToolResult, McpError> {
        let result = reflect::handle(self, input.0).await?;
        Ok(self.append_pinned(result).await)
    }

    #[tool(
        name = "connect",
        description = "Create a directed edge between two memories. Use to explicitly link related knowledge."
    )]
    async fn connect(
        &self,
        input: Parameters<ConnectInput>,
    ) -> Result<CallToolResult, McpError> {
        let result = connect::handle(self, input.0).await?;
        Ok(self.append_pinned(result).await)
    }

    #[tool(
        name = "forget",
        description = "Delete a memory. Use for outdated or incorrect information."
    )]
    async fn forget(
        &self,
        input: Parameters<ForgetInput>,
    ) -> Result<CallToolResult, McpError> {
        let result = forget::handle(self, input.0).await?;
        Ok(self.append_pinned(result).await)
    }

    #[tool(
        name = "update_memory",
        description = "Update an existing memory's content or importance score."
    )]
    async fn update_memory(
        &self,
        input: Parameters<UpdateMemoryInput>,
    ) -> Result<CallToolResult, McpError> {
        let result = update::handle(self, input.0).await?;
        Ok(self.append_pinned(result).await)
    }

    #[tool(
        name = "search_graph",
        description = "Execute a direct graph traversal from a starting memory. Returns connected memories with their relations and distances."
    )]
    async fn search_graph(
        &self,
        input: Parameters<SearchGraphInput>,
    ) -> Result<CallToolResult, McpError> {
        let result = graph::handle(self, input.0).await?;
        Ok(self.append_pinned(result).await)
    }

    #[tool(
        name = "get_stats",
        description = "Get memory system statistics: total counts, counts by category, current session info."
    )]
    async fn get_stats(&self) -> Result<CallToolResult, McpError> {
        let result = stats::handle(self).await?;
        Ok(self.append_pinned(result).await)
    }

    #[tool(
        name = "pin",
        description = "Pin a persistent reminder that appears in every response. Use for user preferences, workflow rules, or standing instructions."
    )]
    async fn pin(
        &self,
        input: Parameters<PinInput>,
    ) -> Result<CallToolResult, McpError> {
        let result = pin::handle(self, input.0).await?;
        Ok(self.append_pinned(result).await)
    }

    #[tool(
        name = "unpin",
        description = "Remove a pinned reminder by its memory ID."
    )]
    async fn unpin(
        &self,
        input: Parameters<UnpinInput>,
    ) -> Result<CallToolResult, McpError> {
        let result = unpin::handle(self, input.0).await?;
        Ok(self.append_pinned(result).await)
    }

    #[tool(
        name = "list_pinned",
        description = "List all currently pinned reminders."
    )]
    async fn list_pinned(&self) -> Result<CallToolResult, McpError> {
        let result = list_pinned::handle(self).await?;
        Ok(self.append_pinned(result).await)
    }

    #[tool(
        name = "rate_recall",
        description = "Provide feedback on a recalled memory's usefulness. Updates adaptive importance (Bayesian, FSRS, SM-2) so future retrieval improves. Quality: 0=blackout, 1-2=not useful, 3=hard but useful, 4=good, 5=perfect."
    )]
    async fn rate_recall(
        &self,
        input: Parameters<RateRecallInput>,
    ) -> Result<CallToolResult, McpError> {
        let result = rate_recall::handle(self, input.0).await?;
        Ok(self.append_pinned(result).await)
    }
}

#[tool_handler]
impl rmcp::handler::server::ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Lattice Memory: A persistent, intelligent memory system for agentic coding. \
                 Stores episodic (events), semantic (knowledge), and procedural (patterns) \
                 memories with graph relationships and vector similarity search. \
                 Call session_start at conversation begin, remember after significant events, \
                 recall before new tasks, and session_end at conversation close. \
                 Pinned reminders appear in every tool response — always follow them."
                    .into(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

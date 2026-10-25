//! Memory system orchestrator — ties together store, embedding, scoring, and retrieval.
//!
//! Operation bodies live in `ops::*`; methods here are thin delegations
//! plus small store-passthrough utilities.

pub mod clock;
pub mod ops;

// The pure model lives in `lattice-memory-core`; re-exported under the
// paths it had before the extraction.
pub use lattice_memory_core::{
    avatar, consolidation, contradiction, importance, importance_state, lifecycle, retrieval,
    scoring, types, write_surprise,
};

use crate::config::Config;
use crate::embedding::chat::ChatProvider;
use crate::embedding::EmbeddingProvider;
use crate::error::Result;
use crate::reranker::RerankerProvider;
use crate::store::MemoryStore;
use lattice_memory_core::clock::Clock;

use self::types::*;

pub struct MemorySystem {
    store: Box<dyn MemoryStore>,
    embedder: Box<dyn EmbeddingProvider>,
    chat: Option<Box<dyn ChatProvider>>,
    reranker: Option<Box<dyn RerankerProvider>>,
    config: Config,
    /// The only source of "now" for every operation.
    clock: Box<dyn Clock + Send + Sync>,
    current_session_id: Option<String>,
    /// Consolidatable memories written since the last consolidation
    /// (Step 8 auto-trigger). Counts `remember` writes (episodic/semantic/
    /// procedural) only; the special namespaces `pin` and `avatar_store` are
    /// EXCLUDED by design — `reflect` consolidates episodic memories, and the
    /// avatar has its own `session_end` derivation, so feeding avatar/pinned
    /// writes here would mis-trigger episodic consolidation. Reset by
    /// `reflect`, checked at `session_end`.
    writes_since_consolidation: usize,
}

impl MemorySystem {
    pub fn new(
        store: Box<dyn MemoryStore>,
        embedder: Box<dyn EmbeddingProvider>,
        chat: Option<Box<dyn ChatProvider>>,
        reranker: Option<Box<dyn RerankerProvider>>,
        config: Config,
    ) -> Self {
        Self {
            store,
            embedder,
            chat,
            reranker,
            config,
            clock: Box::new(clock::SystemClock),
            current_session_id: None,
            writes_since_consolidation: 0,
        }
    }

    /// Replace the wall clock (replays, deterministic hosts).
    pub fn with_clock(mut self, clock: Box<dyn Clock + Send + Sync>) -> Self {
        self.clock = clock;
        self
    }

    pub async fn initialize(&mut self) -> Result<()> {
        self.store.initialize().await
    }

    // --- remember ---

    pub async fn remember(
        &mut self,
        content: &str,
        category: MemoryCategory,
        importance: Option<f32>,
        event_type: Option<&str>,
        related_to: Option<&[u64]>,
        session_id: Option<&str>,
    ) -> Result<Memory> {
        ops::remember::run(
            self, content, category, importance, event_type, related_to, session_id,
        )
        .await
    }

    // --- recall ---

    pub async fn recall(
        &mut self,
        query: &str,
        categories: &[MemoryCategory],
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        ops::recall::run(self, query, categories, limit, None, None).await
    }

    /// `recall` filtered to memories created within `[after, before]`
    /// (either bound optional). Used by the `recall` tool's date-range params.
    pub async fn recall_in_range(
        &mut self,
        query: &str,
        categories: &[MemoryCategory],
        limit: usize,
        after: Option<chrono::DateTime<chrono::Utc>>,
        before: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Result<Vec<ScoredMemory>> {
        ops::recall::run(self, query, categories, limit, after, before).await
    }

    // --- session management ---

    pub async fn session_start(
        &mut self,
        project: Option<&str>,
        task: Option<&str>,
    ) -> Result<(String, Vec<ScoredMemory>)> {
        ops::session::start(self, project, task).await
    }

    pub async fn session_end(&mut self, summary: Option<&str>) -> Result<()> {
        ops::session::end(self, summary).await
    }

    // --- pinned memories ---

    pub async fn pin(&mut self, content: &str, label: Option<&str>) -> Result<Memory> {
        ops::pin::pin(self, content, label).await
    }

    pub async fn unpin(&mut self, memory_id: u64) -> Result<bool> {
        ops::pin::unpin(self, memory_id).await
    }

    pub async fn list_pinned(&self) -> Result<Vec<Memory>> {
        self.store.list_all(MemoryCategory::Pinned).await
    }

    // --- reflect (consolidation) ---

    pub async fn reflect(
        &mut self,
        category: Option<MemoryCategory>,
        topic: Option<&str>,
    ) -> Result<Vec<Memory>> {
        ops::reflect::run(self, category, topic).await
    }

    // --- lifecycle ---

    pub async fn prune(&mut self, category: MemoryCategory) -> Result<lifecycle::PruneReport> {
        ops::maintenance::prune(self, category).await
    }

    // --- avatar (persistent user model) ---

    pub async fn avatar_store(
        &mut self,
        avatar_id: &str,
        slot_key: &str,
        value: &str,
        provenance: &[u64],
    ) -> Result<Memory> {
        ops::avatar::store(self, avatar_id, slot_key, value, provenance).await
    }

    pub async fn avatar_search(
        &mut self,
        avatar_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        ops::avatar::search(self, avatar_id, query, limit).await
    }

    pub async fn avatar_get(&self, avatar_id: &str) -> Result<Vec<Memory>> {
        ops::avatar::get(self, avatar_id).await
    }

    // --- feedback ---

    pub async fn rate_recall(
        &mut self,
        category: MemoryCategory,
        memory_id: u64,
        quality: u8,
    ) -> Result<Option<Memory>> {
        ops::feedback::rate_recall(self, category, memory_id, quality).await
    }

    // --- utility ---

    pub async fn get_memory(&self, category: MemoryCategory, id: u64) -> Result<Option<Memory>> {
        self.store.get_memory(category, id).await
    }

    pub async fn connect(
        &mut self,
        from_id: u64,
        to_id: u64,
        relation: &str,
        weight: f32,
        category: MemoryCategory,
    ) -> Result<()> {
        self.store
            .add_edge(category, from_id, to_id, relation, weight)
            .await
    }

    pub async fn forget(&mut self, category: MemoryCategory, id: u64) -> Result<bool> {
        self.store.delete_memory(category, id).await
    }

    pub async fn update_memory(
        &mut self,
        category: MemoryCategory,
        id: u64,
        content: Option<&str>,
        importance: Option<f32>,
    ) -> Result<Option<Memory>> {
        let mut memory = match self.store.get_memory(category, id).await? {
            Some(m) => m,
            None => return Ok(None),
        };

        if let Some(c) = content {
            memory.content = c.to_string();
        }
        if let Some(i) = importance {
            memory.importance = i;
        }
        memory.last_accessed = self.clock.now();
        memory.access_count += 1;

        let vector = self.embedder.embed(&memory.content).await?;
        self.store.upsert_memory(&memory, vector).await?;

        Ok(Some(memory))
    }

    pub async fn search_graph(
        &self,
        category: MemoryCategory,
        start_id: u64,
        max_depth: usize,
        relations: Option<&[&str]>,
    ) -> Result<Vec<TraversalHit>> {
        self.store
            .traverse(category, start_id, max_depth, relations)
            .await
    }

    pub async fn stats(&self) -> Result<MemoryStats> {
        let mut by_category = std::collections::HashMap::new();
        for &cat in MemoryCategory::all_storage() {
            by_category.insert(
                cat.collection_name().to_string(),
                self.store.count(cat).await?,
            );
        }
        let total = self.store.total_count().await?;

        Ok(MemoryStats {
            total,
            by_category,
            current_session: self.current_session_id.clone(),
        })
    }

    pub fn current_session_id(&self) -> Option<&str> {
        self.current_session_id.as_deref()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total: usize,
    pub by_category: std::collections::HashMap<String, usize>,
    pub current_session: Option<String>,
}

use serde::{Deserialize, Serialize};

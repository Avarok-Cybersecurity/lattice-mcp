//! Memory store abstraction (SBIO boundary).
//!
//! Business logic calls this trait for all database operations.
//! Implementations perform the actual LatticeDB I/O (embedded or remote).

pub mod embedded;
pub mod remote;

use async_trait::async_trait;

use crate::error::Result;
use crate::memory::types::{Memory, MemoryCategory, SearchHit, TraversalHit};

/// SBIO boundary: abstracts all LatticeDB operations.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Initialize collections (create if they don't exist, load from disk if they do).
    async fn initialize(&mut self) -> Result<()>;

    /// Upsert a memory as a Point with its embedding vector.
    async fn upsert_memory(&mut self, memory: &Memory, vector: Vec<f32>) -> Result<()>;

    /// Vector similarity search within a specific collection.
    async fn vector_search(
        &self,
        category: MemoryCategory,
        query_vector: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchHit>>;

    /// Get a memory by point ID from a specific collection.
    async fn get_memory(&self, category: MemoryCategory, id: u64) -> Result<Option<Memory>>;

    /// Traverse graph edges from a starting point.
    async fn traverse(
        &self,
        category: MemoryCategory,
        start_id: u64,
        max_depth: usize,
        relations: Option<&[&str]>,
    ) -> Result<Vec<TraversalHit>>;

    /// Add a directed edge between two points in a collection.
    async fn add_edge(
        &mut self,
        category: MemoryCategory,
        from_id: u64,
        to_id: u64,
        relation: &str,
        weight: f32,
    ) -> Result<()>;

    /// Remove a directed edge between two points.
    async fn remove_edge(
        &mut self,
        category: MemoryCategory,
        from_id: u64,
        to_id: u64,
        relation: Option<&str>,
    ) -> Result<bool>;

    /// Delete a memory point.
    async fn delete_memory(&mut self, category: MemoryCategory, id: u64) -> Result<bool>;

    /// Get count for a specific collection.
    async fn count(&self, category: MemoryCategory) -> Result<usize>;

    /// List all memories in a category (used for small collections like Pinned).
    async fn list_all(&self, category: MemoryCategory) -> Result<Vec<Memory>>;

    /// Get total point count across all collections.
    async fn total_count(&self) -> Result<usize>;
}

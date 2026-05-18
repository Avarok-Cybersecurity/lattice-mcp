//! Embedding provider abstraction (SBIO boundary).
//!
//! Business logic calls this trait to generate vector embeddings.
//! Implementations perform the actual HTTP I/O.

pub mod chat;
pub mod openai;

use async_trait::async_trait;

/// SBIO boundary: abstracts text → vector embedding generation.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate an embedding vector for a single text string.
    async fn embed(&self, text: &str) -> crate::error::Result<Vec<f32>>;

    /// Batch-embed multiple texts (more efficient than individual calls).
    async fn embed_batch(&self, texts: &[&str]) -> crate::error::Result<Vec<Vec<f32>>>;

    /// Configured vector dimensionality.
    fn dimensions(&self) -> usize;
}

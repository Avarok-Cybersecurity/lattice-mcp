//! Reranker provider abstraction (SBIO boundary).
//!
//! Business logic calls this trait to rerank retrieval candidates.
//! Implementations perform the actual HTTP I/O.

pub mod http;

use async_trait::async_trait;

/// A candidate for reranking: document text paired with its pre-existing hybrid score.
pub struct RerankCandidate {
    pub document: String,
    pub hybrid_score: f32,
}

/// A reranked result: original index + the reranker's relevance score.
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f32,
}

/// SBIO boundary: abstracts cross-encoder reranking I/O.
#[async_trait]
pub trait RerankerProvider: Send + Sync {
    /// Rerank a batch of candidate documents against a query.
    async fn rerank(
        &self,
        query: &str,
        candidates: &[RerankCandidate],
    ) -> crate::error::Result<Vec<RerankResult>>;
}

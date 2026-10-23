//! Recall-feedback operation — updates adaptive importance state.

use chrono::Utc;
use tracing::info;

use crate::error::Result;
use crate::memory::importance;
use crate::memory::types::*;
use crate::memory::MemorySystem;

/// Update a memory's adaptive importance state after recall feedback.
///
/// `quality` uses the 0–5 scale (SM-2 compatible):
/// - 0–2: memory was not useful / not recalled
/// - 3–5: memory was useful / recalled correctly
///
/// Updates Bayesian (alpha/beta), FSRS (stability/difficulty),
/// and SM-2 (easiness/interval/repetitions) state, then re-upserts.
#[tracing::instrument(skip_all, fields(category = category.collection_name(), quality))]
pub(crate) async fn rate_recall(
    sys: &mut MemorySystem,
    category: MemoryCategory,
    memory_id: u64,
    quality: u8,
) -> Result<Option<Memory>> {
    let mut memory = match sys.store.get_memory(category, memory_id).await? {
        Some(m) => m,
        None => return Ok(None),
    };

    let now = Utc::now();
    let elapsed_days = {
        let last = memory
            .importance_state
            .last_review
            .unwrap_or(memory.created_at);
        (now - last).num_seconds().max(0) as f64 / 86400.0
    };

    // Bayesian: quality >= 3 counts as relevant, with temporal decay on stale evidence
    importance::bayesian::update(&mut memory.importance_state, quality >= 3, elapsed_days);

    // FSRS: update stability and difficulty
    importance::fsrs::update_state(&mut memory.importance_state, quality, elapsed_days);

    // SM-2: update easiness, interval, repetitions
    importance::sm2::update(&mut memory.importance_state, quality);

    // Record review timestamp
    memory.importance_state.last_review = Some(now);

    // Bump access metadata
    memory.last_accessed = now;
    memory.access_count += 1;

    // Re-upsert with updated state
    let vector = sys.embedder.embed(&memory.content).await?;
    sys.store.upsert_memory(&memory, vector).await?;

    info!(
        id = memory_id,
        quality,
        alpha = memory.importance_state.alpha,
        beta = memory.importance_state.beta,
        stability = memory.importance_state.fsrs_stability,
        "Recall rated"
    );

    Ok(Some(memory))
}

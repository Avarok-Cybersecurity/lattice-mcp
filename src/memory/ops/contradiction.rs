//! Write-time contradiction handling — I/O glue for the pure
//! `memory::contradiction` logic. Experimental, opt-in.
//!
//! Semantic-only: `CONTRADICTS`/`SUPERSEDES` are registered for the Semantic
//! collection only. The conservative algorithmic path may only flag
//! (`CONTRADICTS`); an irreversible `SUPERSEDES` requires LLM confirmation
//! AND `auto_supersede`.

use crate::embedding::chat_types::EntailmentVerdict;
use crate::error::Result;
use crate::memory::contradiction::{is_likely_contradiction, lexical_disagreement};
use crate::memory::types::{relations, Memory, MemoryCategory, MemoryMetadata};
use crate::memory::MemorySystem;

pub(crate) fn retrieval_filter_enabled(sys: &MemorySystem) -> bool {
    sys.config
        .retrieval_filter
        .as_ref()
        .is_some_and(|f| f.enabled)
}

/// Persist `superseded_by` on the superseded memory itself, so recall can
/// filter it without finding its superseder (which may be outside the
/// candidate set, or deleted).
async fn mark_superseded(sys: &mut MemorySystem, mut existing: Memory, by: u64) -> Result<()> {
    if let MemoryMetadata::Semantic { superseded_by, .. } = &mut existing.metadata {
        *superseded_by = Some(by);
    }
    let vector = sys.embedder.embed(&existing.content).await?;
    sys.store.upsert_memory(&existing, vector).await
}

pub(crate) async fn check(
    sys: &mut MemorySystem,
    memory_id: u64,
    content: &str,
    category: MemoryCategory,
    neighbors: &[(u64, f32)],
) -> Result<()> {
    let cfg = match sys.config.contradiction.as_ref().filter(|c| c.enabled) {
        Some(c) => c.clone(),
        None => return Ok(()),
    };
    if category != MemoryCategory::Semantic {
        return Ok(());
    }

    for &(nid, sim) in neighbors {
        if sim < cfg.min_similarity {
            continue;
        }
        let existing = match sys.store.get_memory(MemoryCategory::Semantic, nid).await? {
            Some(m) => m,
            None => continue,
        };

        // LLM path first; `Ok(None)` ⇒ conservative algorithmic fallback (PCND).
        let verdict = match &sys.chat {
            Some(chat) => chat.entailment(&existing.content, content).await?,
            None => None,
        };

        let (is_contra, llm_confirmed) = match verdict {
            Some(EntailmentVerdict::Contradicts) => (true, true),
            Some(_) => (false, false), // Entails / Neutral ⇒ no action
            None => {
                let d = lexical_disagreement(&existing.content, content);
                (is_likely_contradiction(sim, d, cfg.min_similarity), false)
            }
        };

        if is_contra {
            let _ = sys
                .store
                .add_edge(
                    MemoryCategory::Semantic,
                    memory_id,
                    nid,
                    relations::CONTRADICTS,
                    sim,
                )
                .await;
            if llm_confirmed && cfg.auto_supersede {
                let _ = sys
                    .store
                    .add_edge(
                        MemoryCategory::Semantic,
                        memory_id,
                        nid,
                        relations::SUPERSEDES,
                        sim,
                    )
                    .await;
                if retrieval_filter_enabled(sys) {
                    mark_superseded(sys, existing, memory_id).await?;
                }
            }
        }
    }
    Ok(())
}

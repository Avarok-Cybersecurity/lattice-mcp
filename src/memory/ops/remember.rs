//! `remember` operation and similarity auto-linking.
//!
//! Ordering invariant: the new memory's point is upserted **exactly once**,
//! with its final state, and **all edges are created after that upsert**.
//! lattice-core stores outgoing edges on the source point; the embedded store's
//! `upsert_memory` carries existing edges across a re-upsert (it writes point
//! state only), so later state updates are edge-safe. Creating edges before the
//! point exists would still target a missing node, hence the ordering here.

use chrono::Utc;
use tracing::info;

use crate::error::{McpMemoryError, Result};
use crate::memory::types::*;
use crate::memory::{importance, importance_state, scoring, write_surprise, MemorySystem};

pub(crate) async fn run(
    sys: &mut MemorySystem,
    content: &str,
    category: MemoryCategory,
    importance_arg: Option<f32>,
    event_type: Option<&str>,
    related_to: Option<&[u64]>,
    session_id: Option<&str>,
) -> Result<Memory> {
    run_inner(
        sys,
        content,
        category,
        importance_arg,
        event_type,
        related_to,
        session_id,
        Vec::new(),
        None,
    )
    .await
}

/// Internal `remember` with optional provenance and a precomputed embedding
/// vector. `provenance` (non-empty only for consolidated semantic records from
/// `reflect`) is baked into the metadata of the *single* upsert so the edges
/// this function creates afterward are never wiped by a second write. Callers
/// that have already embedded `content` (e.g. `reflect` after a batch embed)
/// pass `Some(vector)` to skip the redundant embedding; everyone else uses the
/// `run` wrapper which passes an empty provenance and `None`.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, fields(category = category.collection_name()))]
pub(crate) async fn run_inner(
    sys: &mut MemorySystem,
    content: &str,
    category: MemoryCategory,
    importance_arg: Option<f32>,
    event_type: Option<&str>,
    related_to: Option<&[u64]>,
    session_id: Option<&str>,
    provenance: Vec<u64>,
    precomputed_vector: Option<Vec<f32>>,
) -> Result<Memory> {
    if matches!(category, MemoryCategory::Pinned) {
        return Err(McpMemoryError::PinLimit {
            message: "Use pin() to create pinned memories".into(),
        });
    }
    if matches!(category, MemoryCategory::Avatar) {
        return Err(McpMemoryError::PinLimit {
            message: "Use avatar_store() to write avatar memories".into(),
        });
    }

    let now = Utc::now();
    let id = Memory::content_id(content, category);

    let et = event_type.unwrap_or("general");
    let imp = importance_arg.unwrap_or_else(|| scoring::auto_importance(content, et));

    let metadata = match category {
        MemoryCategory::Episodic => MemoryMetadata::Episodic {
            event_type: et.to_string(),
        },
        MemoryCategory::Semantic => MemoryMetadata::Semantic {
            entity_type: et.to_string(),
            confidence: imp,
            provenance,
        },
        MemoryCategory::Procedural => MemoryMetadata::Procedural {
            pattern_type: et.to_string(),
            trigger: String::new(),
            success_rate: 1.0,
            usage_count: 1,
        },
        MemoryCategory::Pinned | MemoryCategory::Avatar => unreachable!(),
    };

    // Initialize FSRS stability from category config when adaptive importance enabled.
    let mut init_state = importance_state::ImportanceState::default();
    if let Some(ref imp_cfg) = sys.config.importance {
        if imp_cfg.enabled {
            init_state.fsrs_stability =
                importance::fsrs::initial_stability(category, &imp_cfg.fsrs);
            init_state.fsrs_difficulty = imp_cfg.fsrs.initial_difficulty;
        }
    }

    let mut memory = Memory {
        id,
        category,
        content: content.to_string(),
        created_at: now,
        last_accessed: now,
        access_count: 0,
        importance: imp,
        session_id: session_id
            .or(sys.current_session_id.as_deref())
            .map(String::from),
        metadata,
        importance_state: init_state,
    };

    // Embed once (or reuse a caller-supplied vector); reused for the neighbor
    // search and the upsert.
    let vector = match precomputed_vector {
        Some(v) => v,
        None => sys.embedder.embed(content).await?,
    };

    // Discover similar neighbors (search only — the new point need not be
    // stored to search existing ones). Returns the unified `(id, similarity)`
    // candidate set shared by write-time signals, novelty (Step 3) and
    // contradiction handling (Step 9).
    let neighbors = find_similar(sys, category, id, &vector).await?;
    let neighbor_scores: Vec<f32> = neighbors.iter().map(|(_, s)| *s).collect();

    // Step 3: write-policy — novelty tagging and optional near-dup skip.
    // Decided BEFORE any write, so a skip is a clean no-op (nothing stored,
    // nothing to delete).
    let write_policy_on = sys.config.write_policy.as_ref().is_some_and(|w| w.enabled);
    if write_policy_on {
        let wp = sys.config.write_policy.as_ref().unwrap();
        let nov = write_surprise::novelty(&neighbor_scores);
        memory.importance_state.novelty = nov;

        if wp.skip_near_dups && write_surprise::is_near_dup(nov, wp.near_dup_threshold) {
            let dup = neighbors
                .iter()
                .copied()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(dup_id, _)| dup_id);
            if let Some(dup_id) = dup {
                if let Some(existing) = sys.store.get_memory(category, dup_id).await? {
                    info!(
                        id,
                        dup_id,
                        novelty = nov,
                        "Near-duplicate write skipped; returning existing memory"
                    );
                    return Ok(existing);
                }
            }
            // Fall through and keep the write if the duplicate is gone.
        }
    }

    // Write-time signals: TF-IDF uniqueness + salience from neighbor scores.
    let signals_on =
        sys.config.importance.as_ref().is_some_and(|c| c.enabled) && !neighbor_scores.is_empty();
    if signals_on {
        memory.importance_state.tfidf_uniqueness =
            importance::write_time::tfidf_uniqueness(&neighbor_scores);
        memory.importance_state.salience = importance::write_time::salience(&neighbor_scores);
    }

    // Single authoritative upsert with the final state.
    commit_upsert(sys, &memory, vector).await?;

    // Step 8: count actual stores immediately after the upsert succeeds —
    // BEFORE the fallible edge/contradiction steps below — so a later error
    // cannot desync the auto-consolidation counter from what was stored.
    // (The near-dup skip path returned earlier and is intentionally not
    // counted — no new memory was added.)
    sys.writes_since_consolidation = sys.writes_since_consolidation.saturating_add(1);

    // --- Edges (created strictly AFTER the final upsert) ---

    // Caller-specified related links.
    if let Some(ids) = related_to {
        let rel = match category {
            MemoryCategory::Episodic => relations::PRECEDED_BY,
            MemoryCategory::Semantic => relations::RELATES_TO,
            MemoryCategory::Procedural => relations::APPLIES_TO,
            MemoryCategory::Pinned | MemoryCategory::Avatar => unreachable!(),
        };
        for &related_id in ids {
            // Ignore edge errors (target may not exist).
            let _ = sys.store.add_edge(category, id, related_id, rel, 0.8).await;
        }
    }

    // Auto-link to sufficiently similar neighbors.
    link_similar_edges(sys, category, id, &neighbors).await;

    // Step 9: write-time contradiction detection (experimental, opt-in,
    // Semantic-only). Runs only for kept memories, so any
    // CONTRADICTS/SUPERSEDES edge points from a live node.
    super::contradiction::check(sys, id, &memory.content, category, &neighbors).await?;

    info!(id, category = category.collection_name(), "Memory stored");
    Ok(memory)
}

/// Single authoritative upsert of a memory point. Shared by `remember` and
/// `avatar_store`: a point is upserted once with its final state and all edges
/// are created by the caller AFTER this returns. The store preserves existing
/// edges across re-upserts, so this ordering is about not creating edges before
/// the node exists rather than guarding against edge loss.
pub(crate) async fn commit_upsert(
    sys: &mut MemorySystem,
    memory: &Memory,
    vector: Vec<f32>,
) -> Result<()> {
    sys.store.upsert_memory(memory, vector).await
}

/// Vector-search for similar neighbors. Pure of edge side effects: returns
/// `(neighbor_id, similarity)` for every distinct neighbor, excluding self.
///
/// Note: `vector_search` returns cosine distance `[0, 2]` where 0 = identical;
/// converted to similarity via `1.0 - distance`.
pub(crate) async fn find_similar(
    sys: &MemorySystem,
    category: MemoryCategory,
    self_id: u64,
    vector: &[f32],
) -> Result<Vec<(u64, f32)>> {
    let hits = sys.store.vector_search(category, vector, 5).await?;
    Ok(hits
        .into_iter()
        .filter(|h| h.id != self_id)
        .map(|h| (h.id, 1.0 - h.score))
        .collect())
}

/// Create typed auto-link edges for neighbors above the similarity cutoff.
/// Called only after the source point's final upsert so the edges persist.
async fn link_similar_edges(
    sys: &mut MemorySystem,
    category: MemoryCategory,
    from_id: u64,
    neighbors: &[(u64, f32)],
) {
    for &(nid, similarity) in neighbors {
        if similarity > 0.7 {
            let rel = match category {
                MemoryCategory::Episodic => relations::PRECEDED_BY,
                MemoryCategory::Semantic => relations::RELATES_TO,
                MemoryCategory::Procedural => relations::ALTERNATIVE_TO,
                MemoryCategory::Pinned | MemoryCategory::Avatar => continue,
            };
            let _ = sys
                .store
                .add_edge(category, from_id, nid, rel, similarity)
                .await;
        }
    }
}

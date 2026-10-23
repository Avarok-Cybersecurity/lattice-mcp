//! Maintenance operations — PageRank recomputation over the memory graph.

use chrono::Utc;
use tracing::info;

use crate::error::Result;
use crate::memory::types::MemoryCategory;
use crate::memory::{importance, lifecycle, scoring, MemorySystem};

/// Reuse a memory's already-stored embedding for a metadata-only re-upsert
/// (pagerank/importance changed but the content did not), avoiding a redundant
/// — and on a remote 8B embedder, expensive — embedding call. Falls back to
/// embedding only if the store cannot supply the vector.
///
/// NOTE: lattice-core has no payload-only update, so persisting these scalar
/// updates still re-upserts the whole point (which replaces its outgoing
/// edges). That edge-loss is a separate, pre-existing issue; this helper only
/// removes the wasted embedding.
async fn reuse_or_embed(
    sys: &MemorySystem,
    category: MemoryCategory,
    id: u64,
    content: &str,
) -> Result<Vec<f32>> {
    if let Some((_, v)) = sys
        .store
        .get_vectors(category, &[id])
        .await?
        .into_iter()
        .next()
    {
        Ok(v)
    } else {
        sys.embedder.embed(content).await
    }
}

/// Build adjacency list from graph traversal and run PageRank,
/// then update `importance_state.pagerank` on each affected memory.
#[tracing::instrument(skip_all, fields(category = category.collection_name()))]
pub(crate) async fn update_pagerank(
    sys: &mut MemorySystem,
    category: MemoryCategory,
) -> Result<()> {
    let imp_cfg = match &sys.config.importance {
        Some(c) => c,
        None => return Ok(()),
    };

    let memories = sys.store.list_all(category).await?;
    if memories.is_empty() {
        return Ok(());
    }

    // Build adjacency list: for each memory, get depth-1 neighbors.
    let mut adjacency = Vec::new();
    for mem in &memories {
        let hits = sys.store.traverse(category, mem.id, 1, None).await?;
        let targets: Vec<u64> = hits.iter().map(|h| h.target_id).collect();
        adjacency.push((mem.id, targets));
    }

    let scores = importance::pagerank::compute_pagerank(
        &adjacency,
        imp_cfg.pagerank_damping,
        imp_cfg.pagerank_iterations,
    );

    // Update each memory's pagerank score
    let mut updated = 0_usize;
    for mem in memories {
        if let Some(&pr) = scores.get(&mem.id) {
            let old = mem.importance_state.pagerank;
            if (old - pr).abs() > 0.001 {
                let mut updated_mem = mem;
                updated_mem.importance_state.pagerank = pr;
                let vector =
                    reuse_or_embed(sys, category, updated_mem.id, &updated_mem.content).await?;
                sys.store.upsert_memory(&updated_mem, vector).await?;
                updated += 1;
            }
        }
    }

    if updated > 0 {
        info!(
            category = category.collection_name(),
            updated, "PageRank updated"
        );
    }
    Ok(())
}

use crate::memory::lifecycle::PruneReport;

/// Utility-driven lifecycle pass (Step 5). Keeps the top
/// `lifecycle.budget_per_category` memories by blended utility; demotes the
/// overflow (importance halved — recoverable), hard-deleting only when
/// `allow_delete` and utility is below `delete_below`. No-op when the
/// feature is disabled or the category is within budget (behavior preserving).
/// Pinned memories are never pruned (user-controlled).
#[tracing::instrument(skip_all, fields(category = category.collection_name()))]
pub(crate) async fn prune(sys: &mut MemorySystem, category: MemoryCategory) -> Result<PruneReport> {
    let lc = match sys.config.lifecycle.as_ref().filter(|l| l.enabled) {
        Some(l) => l.clone(),
        None => return Ok(PruneReport::default()),
    };
    // Pinned is user-curated; Avatar has its own retention (forget/supersede).
    // Neither is subject to utility-driven prune.
    if matches!(category, MemoryCategory::Pinned | MemoryCategory::Avatar) {
        return Ok(PruneReport {
            category: category.collection_name().into(),
            ..Default::default()
        });
    }

    let memories = sys.store.list_all(category).await?;
    let total = memories.len();
    let now = Utc::now();

    let half_life = match category {
        MemoryCategory::Episodic => sys.config.memory.episodic_decay_half_life_days,
        MemoryCategory::Semantic => sys.config.memory.semantic_decay_half_life_days,
        MemoryCategory::Procedural => sys.config.memory.procedural_decay_half_life_days,
        MemoryCategory::Pinned | MemoryCategory::Avatar => unreachable!(),
    };
    let imp_cfg = sys.config.importance.clone();

    let items: Vec<lifecycle::LifecycleItem> = memories
        .iter()
        .map(|m| {
            let imp = match &imp_cfg {
                Some(c) if c.enabled => {
                    importance::composite_importance(&m.importance_state, now, &c.weights)
                }
                _ => scoring::access_adjusted_importance(m.importance, m.access_count),
            };
            let recency = scoring::recency_score(m.last_accessed, now, half_life);
            lifecycle::LifecycleItem {
                id: m.id,
                utility: 0.8 * imp + 0.2 * recency,
                created_at: m.created_at,
            }
        })
        .collect();

    let plan = lifecycle::eviction_plan(
        &items,
        lc.budget_per_category,
        lc.allow_delete,
        lc.delete_below,
    );

    for &id in &plan.demote {
        if let Some(mut m) = sys.store.get_memory(category, id).await? {
            m.importance = (m.importance * 0.5).max(0.0);
            let v = reuse_or_embed(sys, category, id, &m.content).await?;
            sys.store.upsert_memory(&m, v).await?;
        }
    }
    for &id in &plan.evict {
        sys.store.delete_memory(category, id).await?;
    }

    let report = PruneReport {
        category: category.collection_name().into(),
        kept: total - plan.demote.len() - plan.evict.len(),
        demoted: plan.demote.len(),
        evicted: plan.evict.len(),
    };
    if report.demoted > 0 || report.evicted > 0 {
        info!(
            category = category.collection_name(),
            demoted = report.demoted,
            evicted = report.evicted,
            "Lifecycle prune applied"
        );
    }
    Ok(report)
}

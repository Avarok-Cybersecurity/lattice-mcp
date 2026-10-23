//! `reflect` operation — cluster related memories and consolidate.

use tracing::info;

use crate::config::ConsolidationMode;
use crate::error::Result;
use crate::memory::ops::maintenance;
use crate::memory::types::*;
use crate::memory::{consolidation, MemorySystem};

#[tracing::instrument(skip_all)]
pub(crate) async fn run(
    sys: &mut MemorySystem,
    category: Option<MemoryCategory>,
    topic: Option<&str>,
) -> Result<Vec<Memory>> {
    let cat = category.unwrap_or(MemoryCategory::Episodic);

    // Find relevant memories to consolidate
    let memories_to_consolidate = if let Some(t) = topic {
        let scored = sys
            .recall(t, &[cat], sys.config.memory.max_retrieval_results)
            .await?;
        scored.into_iter().map(|s| s.memory).collect::<Vec<_>>()
    } else {
        // Get recent episodic memories and cluster them
        // For now, recall all with a broad query
        let scored = sys.recall("recent activity", &[cat], 50).await?;
        scored.into_iter().map(|s| s.memory).collect::<Vec<_>>()
    };

    if memories_to_consolidate.len() < 2 {
        return Ok(Vec::new());
    }

    // Detect clusters with importance + temporal filtering
    let params = consolidation::ClusterParams {
        min_overlap: 2,
        min_importance: sys.config.consolidation.min_importance,
        temporal_half_life_days: sys.config.consolidation.temporal_half_life_days,
    };
    let clusters = consolidation::detect_clusters(&memories_to_consolidate, params);

    // Pass 1: compute (content, provenance) for every cluster (no mutation),
    // so all summaries can be embedded in a single batch call below.
    let distill_on = sys.config.consolidation.distill_enabled == Some(true);
    let mut prepared: Vec<(String, Vec<u64>)> = Vec::new();
    for cluster_indices in &clusters {
        let cluster: Vec<Memory> = cluster_indices
            .iter()
            .map(|&i| memories_to_consolidate[i].clone())
            .collect();

        let (content, provenance) = if distill_on {
            let dk = match &sys.chat {
                Some(chat) => match chat.distill(&cluster).await? {
                    Some(dk) => dk,
                    None => consolidation::distill::algorithmic_distill(&cluster),
                },
                None => consolidation::distill::algorithmic_distill(&cluster),
            };
            (consolidation::distill::render(&dk), dk.provenance)
        } else {
            // Llm and Auto behave identically (chat if available, else algo).
            let summary = match &sys.config.consolidation.mode {
                ConsolidationMode::Llm | ConsolidationMode::Auto => {
                    if let Some(ref chat) = sys.chat {
                        chat.summarize(&cluster).await?
                    } else {
                        consolidation::algorithmic_summarize(&cluster)
                    }
                }
                ConsolidationMode::Algorithmic => consolidation::algorithmic_summarize(&cluster),
            };
            (summary, Vec::new())
        };
        prepared.push((content, provenance));
    }

    // Batch-embed all consolidated summaries in ONE call (was one embed per
    // cluster). embed_batch returns empty for empty input.
    let contents: Vec<&str> = prepared.iter().map(|(c, _)| c.as_str()).collect();
    let vectors = sys.embedder.embed_batch(&contents).await?;
    // Contract: embed_batch returns exactly one vector per input. A mismatch
    // would silently truncate the `zip` below and mis-pair summaries with
    // vectors, so fail fast instead.
    if vectors.len() != prepared.len() {
        return Err(crate::error::McpMemoryError::Embedding {
            message: format!(
                "embed_batch returned {} vectors for {} inputs",
                vectors.len(),
                prepared.len()
            ),
        });
    }

    // Pass 2: store each as a semantic memory, reusing its precomputed vector.
    // Provenance fix: cross-collection edges are unsupported by LatticeDB, so
    // the source IDs live in the consolidated record's own payload (SSOT,
    // survives round-trip) instead of only in the summary text. It is threaded
    // into `run_inner` so provenance lands on the single authoritative upsert
    // (one write, no redundant re-upsert of the same point).
    let mut created = Vec::new();
    for ((content, provenance), vector) in prepared.into_iter().zip(vectors) {
        let semantic_mem = super::remember::run_inner(
            sys,
            &content,
            MemoryCategory::Semantic,
            Some(0.8),
            Some("consolidated"),
            None,
            None,
            provenance,
            Some(vector),
        )
        .await?;

        created.push(semantic_mem);
    }

    // Run PageRank on the target category and update importance_state.
    if sys.config.importance.as_ref().is_some_and(|c| c.enabled) {
        maintenance::update_pagerank(sys, cat).await?;
    }

    // Step 8: consolidation just ran — reset the auto-trigger counter.
    sys.writes_since_consolidation = 0;

    info!(consolidated = created.len(), "Reflection complete");
    Ok(created)
}

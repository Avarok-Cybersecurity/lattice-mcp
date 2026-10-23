//! `recall` operation — HybridRAG retrieval pipeline.

use chrono::{DateTime, Utc};

use crate::error::Result;
use crate::memory::types::*;
use crate::memory::{importance, retrieval, scoring, MemorySystem};
use crate::reranker::RerankCandidate;

#[tracing::instrument(skip_all, fields(query_len = query.len(), limit))]
pub(crate) async fn run(
    sys: &mut MemorySystem,
    query: &str,
    categories: &[MemoryCategory],
    limit: usize,
    after: Option<DateTime<Utc>>,
    before: Option<DateTime<Utc>>,
) -> Result<Vec<ScoredMemory>> {
    // Step 10: optional query transformation before embedding. LLM path
    // first; `Ok(None)` ⇒ conservative algorithmic rewrite (PCND). The
    // original `query` is still used for reranking (semantic intent).
    let effective_query = if sys
        .config
        .query_transform
        .as_ref()
        .is_some_and(|q| q.enabled)
    {
        match &sys.chat {
            Some(chat) => match chat.transform_query(query).await? {
                Some(qt) => qt.rewritten,
                None => retrieval::query::keyword_expand(query),
            },
            None => retrieval::query::keyword_expand(query),
        }
    } else {
        query.to_string()
    };

    let query_vector = sys.embedder.embed(&effective_query).await?;
    let now = Utc::now();
    let mut all_scored = Vec::new();

    let cats = if categories.is_empty() {
        MemoryCategory::all().to_vec()
    } else {
        categories.to_vec()
    };

    // Over-fetch for the reranker, but bound it: an unbounded
    // `limit * candidate_multiplier` could request enormous candidate sets.
    const MAX_CANDIDATES: usize = 1000;
    let search_limit = match &sys.config.reranker {
        Some(rr) => (limit * rr.candidate_multiplier).min(MAX_CANDIDATES),
        None => limit,
    };

    let depth = sys.config.retrieval.graph_hop_depth;
    for category in &cats {
        let hits = sys
            .store
            .vector_search(*category, &query_vector, search_limit)
            .await?;
        if hits.is_empty() {
            continue;
        }

        // Membership set for graph_relevance (O(1) lookups, not O(n*m)).
        let hit_ids_set: std::collections::HashSet<u64> = hits.iter().map(|h| h.id).collect();
        let hit_ids: Vec<u64> = hits.iter().map(|h| h.id).collect();

        // Batch-fetch the hit memories in one round-trip (was N get_memory).
        let mut mem_by_id: std::collections::HashMap<u64, Memory> = sys
            .store
            .get_memories(*category, &hit_ids)
            .await?
            .into_iter()
            .map(|m| (m.id, m))
            .collect();

        // Traverse all hits concurrently (one batched round of round-trips
        // instead of N sequential ones). Errors propagate as before.
        let traversals: Vec<Vec<TraversalHit>> = {
            let store = sys.store.as_ref();
            let futs = hits
                .iter()
                .map(|h| store.traverse(*category, h.id, depth, None));
            let mut out = Vec::with_capacity(hits.len());
            for r in futures::future::join_all(futs).await {
                out.push(r?);
            }
            out
        };

        // Batch-fetch all neighbor memories across traversals in one round-trip.
        let neighbor_ids: Vec<u64> = {
            let mut s: std::collections::HashSet<u64> = std::collections::HashSet::new();
            for t in &traversals {
                for h in t {
                    s.insert(h.target_id);
                }
            }
            s.into_iter().collect()
        };
        let neighbor_by_id: std::collections::HashMap<u64, Memory> = sys
            .store
            .get_memories(*category, &neighbor_ids)
            .await?
            .into_iter()
            .map(|m| (m.id, m))
            .collect();

        for (hit, traversal) in hits.iter().zip(traversals.into_iter()) {
            let memory = match mem_by_id.remove(&hit.id) {
                Some(m) => m,
                None => continue,
            };

            let neighbor_depths: Vec<(u64, usize)> =
                traversal.iter().map(|t| (t.target_id, t.depth)).collect();

            let graph_rel =
                retrieval::graph_relevance_from_traversal(&neighbor_depths, &hit_ids_set);

            // Adaptive importance: use FSRS retention for recency and
            // composite_importance for the importance signal.
            // Legacy fallback: Ebbinghaus decay + access-adjusted importance.
            let use_adaptive = sys.config.importance.as_ref().is_some_and(|c| c.enabled);

            let (recency, imp) = if use_adaptive {
                let elapsed_days = {
                    let secs = (now - memory.last_accessed).num_seconds().max(0) as f64;
                    secs / 86400.0
                };
                let r = importance::fsrs::retention(
                    memory.importance_state.fsrs_stability,
                    elapsed_days,
                );
                let i = importance::composite_importance(
                    &memory.importance_state,
                    now,
                    &sys.config.importance.as_ref().unwrap().weights,
                );
                (r, i)
            } else {
                let half_life = match category {
                    MemoryCategory::Episodic => sys.config.memory.episodic_decay_half_life_days,
                    MemoryCategory::Semantic => sys.config.memory.semantic_decay_half_life_days,
                    MemoryCategory::Procedural => sys.config.memory.procedural_decay_half_life_days,
                    // Pinned and Avatar are stable: they don't decay.
                    MemoryCategory::Pinned | MemoryCategory::Avatar => 36500.0,
                };
                let r = scoring::recency_score(memory.last_accessed, now, half_life);
                let i = scoring::access_adjusted_importance(memory.importance, memory.access_count);
                (r, i)
            };

            // Convert cosine distance [0,2] to similarity: 1.0 - distance
            let vector_sim = 1.0 - hit.score;

            let final_score =
                retrieval::hybrid_score(vector_sim, graph_rel, recency, imp, &sys.config.retrieval);

            // Collect related memories from traversal (batched neighbor map).
            let mut related = Vec::new();
            for t_hit in &traversal {
                if let Some(rel_mem) = neighbor_by_id.get(&t_hit.target_id) {
                    related.push(RelatedMemory {
                        memory: rel_mem.clone(),
                        relation: t_hit.relation.clone(),
                        hop_distance: t_hit.depth,
                    });
                }
            }

            all_scored.push(ScoredMemory {
                memory,
                vector_similarity: vector_sim,
                graph_relevance: graph_rel,
                temporal_recency: recency,
                importance_score: imp,
                final_score,
                related,
            });
        }
    }

    // Temporal range filter (optional): drop memories outside [after, before]
    // by created_at before fusion/rerank/ranking. Pure timestamp comparison.
    if after.is_some() || before.is_some() {
        all_scored.retain(|s| {
            let t = s.memory.created_at;
            after.is_none_or(|a| t >= a) && before.is_none_or(|b| t <= b)
        });
    }

    // Step 4: adaptive fusion (optional) — re-derive final_score from the
    // four sub-signals before reranking. Fixed-weight `hybrid_score` above
    // remains the default when fusion is absent/disabled.
    if let Some(fc) = sys.config.fusion.as_ref().filter(|f| f.enabled) {
        retrieval::fusion::apply(fc, &sys.config.retrieval, &mut all_scored);
    }

    // Multi-signal (optional): blend a lexical keyword-overlap signal into the
    // retrieval score so exact/rare tokens (codes, identifiers, versions) that
    // the embedder blurs still surface. Applied after fusion, before rerank:
    // `final = (1 - w) * final + w * lexical_overlap(query, content)`.
    if let Some(ms) = sys.config.multi_signal.as_ref().filter(|m| m.enabled) {
        let w = ms.lexical_weight;
        for s in &mut all_scored {
            let lex = retrieval::lexical::lexical_overlap(query, &s.memory.content);
            s.final_score = (1.0 - w) * s.final_score + w * lex;
        }
    }

    // Reranker step (optional): refine scores with cross-encoder
    if let (Some(reranker), Some(rr_cfg)) = (&sys.reranker, &sys.config.reranker) {
        let candidates: Vec<RerankCandidate> = all_scored
            .iter()
            .map(|s| RerankCandidate {
                document: s.memory.content.clone(),
                hybrid_score: s.final_score,
            })
            .collect();

        match reranker.rerank(query, &candidates).await {
            Ok(results) => {
                for rr in &results {
                    if rr.index < all_scored.len() {
                        all_scored[rr.index].final_score = retrieval::blend_reranker_score(
                            all_scored[rr.index].final_score,
                            rr.relevance_score,
                            rr_cfg.reranker_weight,
                        );
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Reranker failed, falling back to hybrid scores: {e}");
            }
        }
    }

    // Step 6: MMR diversification (optional). Operates on the full ranked
    // pool, then selects `limit` diverse results. Falls back to relevance
    // truncation when candidate vectors are unavailable (e.g. remote store).
    if let Some(dc) = sys.config.diversify.as_ref().filter(|d| d.enabled) {
        let mut ranked = retrieval::rank_and_dedup(all_scored, usize::MAX);
        if ranked.len() <= limit {
            return Ok(ranked);
        }

        // Fetch stored vectors per category for the ranked candidates.
        let mut by_cat: std::collections::HashMap<MemoryCategory, Vec<u64>> =
            std::collections::HashMap::new();
        for s in &ranked {
            by_cat
                .entry(s.memory.category)
                .or_default()
                .push(s.memory.id);
        }
        let mut vec_by_id: std::collections::HashMap<u64, Vec<f32>> =
            std::collections::HashMap::new();
        for (cat, ids) in &by_cat {
            for (id, v) in sys.store.get_vectors(*cat, ids).await? {
                vec_by_id.insert(id, v);
            }
        }

        // Diversify only if every candidate has a vector; otherwise the
        // documented fallback is pure-relevance truncation.
        if ranked.iter().all(|s| vec_by_id.contains_key(&s.memory.id)) {
            let vectors: Vec<Vec<f32>> = ranked
                .iter()
                .map(|s| vec_by_id[&s.memory.id].clone())
                .collect();
            let relevance: Vec<f32> = ranked.iter().map(|s| s.final_score).collect();
            let picked = retrieval::diversify::mmr(&vectors, &relevance, dc.lambda, limit);
            return Ok(picked.into_iter().map(|i| ranked[i].clone()).collect());
        }
        tracing::warn!(
            "MMR diversification enabled but candidate vectors are unavailable \
             (e.g. remote store does not expose stored vectors); falling back to \
             pure-relevance ranking"
        );
        ranked.truncate(limit);
        return Ok(ranked);
    }

    let result = retrieval::rank_and_dedup(all_scored, limit);
    tracing::info!(returned = result.len(), "recall complete");
    Ok(result)
}

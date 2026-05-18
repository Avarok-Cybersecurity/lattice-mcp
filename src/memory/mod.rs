//! Memory system orchestrator — ties together store, embedding, scoring, and retrieval.

pub mod consolidation;
pub mod importance;
pub mod importance_state;
pub mod retrieval;
pub mod scoring;
pub mod types;

use chrono::Utc;
use tracing::info;

use crate::config::{Config, ConsolidationMode};
use crate::embedding::chat::ChatProvider;
use crate::embedding::EmbeddingProvider;
use crate::error::{McpMemoryError, Result};
use crate::reranker::{RerankCandidate, RerankerProvider};
use crate::store::MemoryStore;

use self::types::*;

pub struct MemorySystem {
    store: Box<dyn MemoryStore>,
    embedder: Box<dyn EmbeddingProvider>,
    chat: Option<Box<dyn ChatProvider>>,
    reranker: Option<Box<dyn RerankerProvider>>,
    config: Config,
    current_session_id: Option<String>,
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
            current_session_id: None,
        }
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
        if matches!(category, MemoryCategory::Pinned) {
            return Err(McpMemoryError::PinLimit {
                message: "Use pin() to create pinned memories".into(),
            });
        }

        let now = Utc::now();
        let id = Memory::content_id(content, category);

        let et = event_type.unwrap_or("general");
        let imp = importance.unwrap_or_else(|| scoring::auto_importance(content, et));

        let metadata = match category {
            MemoryCategory::Episodic => MemoryMetadata::Episodic {
                event_type: et.to_string(),
            },
            MemoryCategory::Semantic => MemoryMetadata::Semantic {
                entity_type: et.to_string(),
                confidence: imp,
            },
            MemoryCategory::Procedural => MemoryMetadata::Procedural {
                pattern_type: et.to_string(),
                trigger: String::new(),
                success_rate: 1.0,
                usage_count: 1,
            },
            MemoryCategory::Pinned => unreachable!(),
        };

        // Initialize FSRS stability from category config when adaptive importance enabled.
        let mut init_state = importance_state::ImportanceState::default();
        if let Some(ref imp_cfg) = self.config.importance {
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
                .or(self.current_session_id.as_deref())
                .map(String::from),
            metadata,
            importance_state: init_state,
        };

        // Generate embedding
        let vector = self.embedder.embed(content).await?;

        // Upsert into store
        self.store.upsert_memory(&memory, vector).await?;

        // Auto-link to related memories
        if let Some(ids) = related_to {
            for &related_id in ids {
                let rel = match category {
                    MemoryCategory::Episodic => relations::PRECEDED_BY,
                    MemoryCategory::Semantic => relations::RELATES_TO,
                    MemoryCategory::Procedural => relations::APPLIES_TO,
                    MemoryCategory::Pinned => unreachable!(),
                };
                // Ignore edge errors (target may not exist)
                let _ = self
                    .store
                    .add_edge(category, id, related_id, rel, 0.8)
                    .await;
            }
        }

        // Auto-discover and link similar memories; returns neighbor scores.
        let neighbor_scores = self.auto_link_similar(&memory).await?;

        // Write-time signals: TF-IDF uniqueness + salience from neighbor scores.
        if self.config.importance.as_ref().is_some_and(|c| c.enabled) && !neighbor_scores.is_empty()
        {
            memory.importance_state.tfidf_uniqueness =
                importance::write_time::tfidf_uniqueness(&neighbor_scores);
            memory.importance_state.salience =
                importance::write_time::salience(&neighbor_scores);
            // Re-upsert with updated write-time signals
            let vector = self.embedder.embed(&memory.content).await?;
            self.store.upsert_memory(&memory, vector).await?;
        }

        info!(id, category = category.collection_name(), "Memory stored");
        Ok(memory)
    }

    /// Auto-link to similar memories and return their similarity scores.
    ///
    /// The returned scores (excluding self) are used by write-time signals
    /// (TF-IDF uniqueness, salience) when adaptive importance is enabled.
    ///
    /// Note: `vector_search` returns cosine distance `[0, 2]` where 0 = identical.
    /// We convert to similarity via `1.0 - distance` before use.
    async fn auto_link_similar(&mut self, memory: &Memory) -> Result<Vec<f32>> {
        let vector = self.embedder.embed(&memory.content).await?;
        let hits = self
            .store
            .vector_search(memory.category, &vector, 5)
            .await?;

        let mut neighbor_scores = Vec::new();

        for hit in hits {
            if hit.id != memory.id {
                let similarity = 1.0 - hit.score;
                neighbor_scores.push(similarity);
                if similarity > 0.7 {
                    let rel = match memory.category {
                        MemoryCategory::Episodic => relations::PRECEDED_BY,
                        MemoryCategory::Semantic => relations::RELATES_TO,
                        MemoryCategory::Procedural => relations::ALTERNATIVE_TO,
                        MemoryCategory::Pinned => continue,
                    };
                    let _ = self
                        .store
                        .add_edge(memory.category, memory.id, hit.id, rel, similarity)
                        .await;
                }
            }
        }
        Ok(neighbor_scores)
    }

    // --- recall ---

    pub async fn recall(
        &mut self,
        query: &str,
        categories: &[MemoryCategory],
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        let query_vector = self.embedder.embed(query).await?;
        let now = Utc::now();
        let mut all_scored = Vec::new();

        let cats = if categories.is_empty() {
            MemoryCategory::all().to_vec()
        } else {
            categories.to_vec()
        };

        let search_limit = match &self.config.reranker {
            Some(rr) => limit * rr.candidate_multiplier,
            None => limit,
        };

        for category in &cats {
            let hits = self
                .store
                .vector_search(*category, &query_vector, search_limit)
                .await?;

            let hit_ids: Vec<u64> = hits.iter().map(|h| h.id).collect();

            for hit in &hits {
                let memory = match self.store.get_memory(*category, hit.id).await? {
                    Some(m) => m,
                    None => continue,
                };

                // Graph neighborhood
                let traversal = self
                    .store
                    .traverse(
                        *category,
                        hit.id,
                        self.config.retrieval.graph_hop_depth,
                        None,
                    )
                    .await?;

                let neighbor_depths: Vec<(u64, usize)> =
                    traversal.iter().map(|t| (t.target_id, t.depth)).collect();

                let graph_rel =
                    retrieval::graph_relevance_from_traversal(&neighbor_depths, &hit_ids);

                // Adaptive importance: use FSRS retention for recency and
                // composite_importance for the importance signal.
                // Legacy fallback: Ebbinghaus decay + access-adjusted importance.
                let use_adaptive = self
                    .config
                    .importance
                    .as_ref()
                    .is_some_and(|c| c.enabled);

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
                        &self.config.importance.as_ref().unwrap().weights,
                    );
                    (r, i)
                } else {
                    let half_life = match category {
                        MemoryCategory::Episodic => {
                            self.config.memory.episodic_decay_half_life_days
                        }
                        MemoryCategory::Semantic => {
                            self.config.memory.semantic_decay_half_life_days
                        }
                        MemoryCategory::Procedural => {
                            self.config.memory.procedural_decay_half_life_days
                        }
                        MemoryCategory::Pinned => 36500.0,
                    };
                    let r = scoring::recency_score(memory.last_accessed, now, half_life);
                    let i = scoring::access_adjusted_importance(
                        memory.importance,
                        memory.access_count,
                    );
                    (r, i)
                };

                // Convert cosine distance [0,2] to similarity: 1.0 - distance
                let vector_sim = 1.0 - hit.score;

                let final_score =
                    retrieval::hybrid_score(vector_sim, graph_rel, recency, imp, &self.config.retrieval);

                // Collect related memories from traversal
                let mut related = Vec::new();
                for t_hit in &traversal {
                    if let Ok(Some(rel_mem)) =
                        self.store.get_memory(*category, t_hit.target_id).await
                    {
                        related.push(RelatedMemory {
                            memory: rel_mem,
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

        // Reranker step (optional): refine scores with cross-encoder
        if let (Some(reranker), Some(rr_cfg)) = (&self.reranker, &self.config.reranker) {
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
                            all_scored[rr.index].final_score =
                                retrieval::blend_reranker_score(
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

        Ok(retrieval::rank_and_dedup(all_scored, limit))
    }

    // --- session management ---

    pub async fn session_start(
        &mut self,
        project: Option<&str>,
        task: Option<&str>,
    ) -> Result<(String, Vec<ScoredMemory>)> {
        let session_id = uuid::Uuid::new_v4().to_string();
        self.current_session_id = Some(session_id.clone());

        let content = match (project, task) {
            (Some(p), Some(t)) => format!("Session started for project '{p}': {t}"),
            (Some(p), None) => format!("Session started for project '{p}'"),
            (None, Some(t)) => format!("Session started: {t}"),
            (None, None) => "Session started".to_string(),
        };

        self.remember(
            &content,
            MemoryCategory::Episodic,
            Some(0.3),
            Some("session"),
            None,
            None,
        )
        .await?;

        // Recall relevant context
        let context = if let Some(t) = task {
            self.recall(t, &[], self.config.memory.max_retrieval_results)
                .await?
        } else {
            Vec::new()
        };

        info!(session_id = %session_id, context_count = context.len(), "Session started");
        Ok((session_id, context))
    }

    pub async fn session_end(&mut self, summary: Option<&str>) -> Result<()> {
        let content = match summary {
            Some(s) => format!("Session ended: {s}"),
            None => "Session ended".to_string(),
        };

        self.remember(
            &content,
            MemoryCategory::Episodic,
            Some(0.3),
            Some("session"),
            None,
            None,
        )
        .await?;

        info!("Session ended");
        self.current_session_id = None;
        Ok(())
    }

    // --- pinned memories ---

    pub async fn pin(&mut self, content: &str, label: Option<&str>) -> Result<Memory> {
        let max_len = self.config.memory.max_pinned_content_length;
        if content.len() > max_len {
            return Err(McpMemoryError::PinLimit {
                message: format!(
                    "Content length {} exceeds max {max_len}",
                    content.len()
                ),
            });
        }

        let existing = self.store.list_all(MemoryCategory::Pinned).await?;
        let max_count = self.config.memory.max_pinned_memories;
        if existing.len() >= max_count {
            return Err(McpMemoryError::PinLimit {
                message: format!(
                    "Already at max pinned memories ({max_count}). Unpin one first."
                ),
            });
        }

        let now = Utc::now();
        let id = Memory::content_id(content, MemoryCategory::Pinned);
        let label_str = label.unwrap_or("pinned").to_string();

        let memory = Memory {
            id,
            category: MemoryCategory::Pinned,
            content: content.to_string(),
            created_at: now,
            last_accessed: now,
            access_count: 0,
            importance: 1.0,
            session_id: self.current_session_id.clone(),
            metadata: MemoryMetadata::Pinned { label: label_str },
            importance_state: importance_state::ImportanceState::default(),
        };

        let zero_vec = vec![0.0_f32; self.config.embedding.dimensions];
        self.store.upsert_memory(&memory, zero_vec).await?;

        info!(id, "Pinned memory stored");
        Ok(memory)
    }

    pub async fn unpin(&mut self, memory_id: u64) -> Result<bool> {
        self.store.delete_memory(MemoryCategory::Pinned, memory_id).await
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
        let cat = category.unwrap_or(MemoryCategory::Episodic);

        // Find relevant memories to consolidate
        let memories_to_consolidate = if let Some(t) = topic {
            let scored = self
                .recall(t, &[cat], self.config.memory.max_retrieval_results)
                .await?;
            scored.into_iter().map(|s| s.memory).collect::<Vec<_>>()
        } else {
            // Get recent episodic memories and cluster them
            // For now, recall all with a broad query
            let scored = self
                .recall("recent activity", &[cat], 50)
                .await?;
            scored.into_iter().map(|s| s.memory).collect::<Vec<_>>()
        };

        if memories_to_consolidate.len() < 2 {
            return Ok(Vec::new());
        }

        // Detect clusters with importance + temporal filtering
        let params = consolidation::ClusterParams {
            min_overlap: 2,
            min_importance: self.config.consolidation.min_importance,
            temporal_half_life_days: self.config.consolidation.temporal_half_life_days,
        };
        let clusters = consolidation::detect_clusters(&memories_to_consolidate, params);

        let mut created = Vec::new();

        for cluster_indices in clusters {
            let cluster: Vec<Memory> = cluster_indices
                .iter()
                .map(|&i| memories_to_consolidate[i].clone())
                .collect();

            // Generate summary (LLM or algorithmic)
            let summary = match &self.config.consolidation.mode {
                ConsolidationMode::Llm => {
                    if let Some(ref chat) = self.chat {
                        chat.summarize(&cluster).await?
                    } else {
                        consolidation::algorithmic_summarize(&cluster)
                    }
                }
                ConsolidationMode::Algorithmic => {
                    consolidation::algorithmic_summarize(&cluster)
                }
                ConsolidationMode::Auto => {
                    if let Some(ref chat) = self.chat {
                        chat.summarize(&cluster).await?
                    } else {
                        consolidation::algorithmic_summarize(&cluster)
                    }
                }
            };

            // Store as semantic memory
            let semantic_mem = self
                .remember(
                    &summary,
                    MemoryCategory::Semantic,
                    Some(0.8),
                    Some("consolidated"),
                    None,
                    None,
                )
                .await?;

            // NOTE: Cross-collection edges (semantic→episodic) are not supported
            // by LatticeDB — both endpoints must exist in the same collection.
            // Provenance is captured in the consolidation summary text instead.

            created.push(semantic_mem);
        }

        // Run PageRank on the target category and update importance_state.
        if self.config.importance.as_ref().is_some_and(|c| c.enabled) {
            self.update_pagerank(cat).await?;
        }

        info!(consolidated = created.len(), "Reflection complete");
        Ok(created)
    }

    /// Build adjacency list from graph traversal and run PageRank,
    /// then update `importance_state.pagerank` on each affected memory.
    async fn update_pagerank(&mut self, category: MemoryCategory) -> Result<()> {
        let imp_cfg = match &self.config.importance {
            Some(c) => c,
            None => return Ok(()),
        };

        let memories = self.store.list_all(category).await?;
        if memories.is_empty() {
            return Ok(());
        }

        // Build adjacency list: for each memory, get depth-1 neighbors.
        let mut adjacency = Vec::new();
        for mem in &memories {
            let hits = self.store.traverse(category, mem.id, 1, None).await?;
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
                    let vector = self.embedder.embed(&updated_mem.content).await?;
                    self.store.upsert_memory(&updated_mem, vector).await?;
                    updated += 1;
                }
            }
        }

        if updated > 0 {
            info!(
                category = category.collection_name(),
                updated,
                "PageRank updated"
            );
        }
        Ok(())
    }

    // --- feedback ---

    /// Update a memory's adaptive importance state after recall feedback.
    ///
    /// `quality` uses the 0–5 scale (SM-2 compatible):
    /// - 0–2: memory was not useful / not recalled
    /// - 3–5: memory was useful / recalled correctly
    ///
    /// Updates Bayesian (alpha/beta), FSRS (stability/difficulty),
    /// and SM-2 (easiness/interval/repetitions) state, then re-upserts.
    pub async fn rate_recall(
        &mut self,
        category: MemoryCategory,
        memory_id: u64,
        quality: u8,
    ) -> Result<Option<Memory>> {
        let mut memory = match self.store.get_memory(category, memory_id).await? {
            Some(m) => m,
            None => return Ok(None),
        };

        let now = Utc::now();
        let elapsed_days = {
            let last = memory.importance_state.last_review.unwrap_or(memory.created_at);
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
        let vector = self.embedder.embed(&memory.content).await?;
        self.store.upsert_memory(&memory, vector).await?;

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

    // --- utility ---

    pub async fn get_memory(
        &self,
        category: MemoryCategory,
        id: u64,
    ) -> Result<Option<Memory>> {
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
        memory.last_accessed = Utc::now();
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
        self.store.traverse(category, start_id, max_depth, relations).await
    }

    pub async fn stats(&self) -> Result<MemoryStats> {
        let mut by_category = std::collections::HashMap::new();
        for &cat in MemoryCategory::all() {
            by_category.insert(cat.collection_name().to_string(), self.store.count(cat).await?);
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

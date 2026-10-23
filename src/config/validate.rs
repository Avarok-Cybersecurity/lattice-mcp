use super::*;

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.embedding.dimensions == 0 {
            return Err(McpMemoryError::Config {
                message: "embedding.dimensions must be > 0".into(),
            });
        }
        if self.embedding.endpoint.is_empty() {
            return Err(McpMemoryError::Config {
                message: "embedding.endpoint must not be empty".into(),
            });
        }
        if let Some(ref key_env) = self.embedding.api_key_env {
            if key_env.is_empty() {
                return Err(McpMemoryError::Config {
                    message: "embedding.api_key_env must not be empty when specified".into(),
                });
            }
        }

        let w = &self.retrieval;
        let sum = w.vector_weight + w.graph_weight + w.recency_weight + w.importance_weight;
        if (sum - 1.0).abs() > 0.01 {
            return Err(McpMemoryError::Config {
                message: format!("retrieval weights must sum to 1.0, got {sum}"),
            });
        }
        if w.graph_hop_depth == 0 {
            return Err(McpMemoryError::Config {
                message: "retrieval.graph_hop_depth must be > 0".into(),
            });
        }

        if self.memory.consolidation_threshold == 0 {
            return Err(McpMemoryError::Config {
                message: "memory.consolidation_threshold must be > 0".into(),
            });
        }

        if self.memory.max_pinned_memories == 0 {
            return Err(McpMemoryError::Config {
                message: "memory.max_pinned_memories must be > 0".into(),
            });
        }

        if self.memory.max_pinned_content_length == 0 {
            return Err(McpMemoryError::Config {
                message: "memory.max_pinned_content_length must be > 0".into(),
            });
        }

        if matches!(self.database.mode, DatabaseMode::Remote) && self.database.remote_url.is_none()
        {
            return Err(McpMemoryError::Config {
                message: "database.remote_url required when mode is 'remote'".into(),
            });
        }

        // `Llm` requires a chat endpoint (no fallback). `Auto` intentionally
        // permits no endpoint: reflect() then uses the EXPLICIT algorithmic
        // path (documented graceful degradation, not a silent default), so it
        // is not a PCND violation.
        if matches!(self.consolidation.mode, ConsolidationMode::Llm)
            && self.consolidation.chat_endpoint.is_none()
        {
            return Err(McpMemoryError::Config {
                message: "consolidation.chat_endpoint required when mode is 'llm'".into(),
            });
        }

        if self.consolidation.min_importance < 0.0 || self.consolidation.min_importance > 1.0 {
            return Err(McpMemoryError::Config {
                message: "consolidation.min_importance must be in [0.0, 1.0]".into(),
            });
        }
        if self.consolidation.temporal_half_life_days <= 0.0 {
            return Err(McpMemoryError::Config {
                message: "consolidation.temporal_half_life_days must be > 0".into(),
            });
        }

        if let Some(ref imp) = self.importance {
            if imp.pagerank_damping < 0.0 || imp.pagerank_damping > 1.0 {
                return Err(McpMemoryError::Config {
                    message: "importance.pagerank_damping must be in [0.0, 1.0]".into(),
                });
            }
            if imp.pagerank_iterations == 0 {
                return Err(McpMemoryError::Config {
                    message: "importance.pagerank_iterations must be > 0".into(),
                });
            }
            if imp.fsrs.initial_difficulty < 0.0 || imp.fsrs.initial_difficulty > 1.0 {
                return Err(McpMemoryError::Config {
                    message: "importance.fsrs.initial_difficulty must be in [0.0, 1.0]".into(),
                });
            }
        }

        if let Some(ref rr) = self.reranker {
            if rr.endpoint.is_empty() {
                return Err(McpMemoryError::Config {
                    message: "reranker.endpoint must not be empty".into(),
                });
            }
            if rr.model.is_empty() {
                return Err(McpMemoryError::Config {
                    message: "reranker.model must not be empty".into(),
                });
            }
            if rr.reranker_weight < 0.0 || rr.reranker_weight > 1.0 {
                return Err(McpMemoryError::Config {
                    message: "reranker.reranker_weight must be in [0.0, 1.0]".into(),
                });
            }
            if rr.candidate_multiplier == 0 {
                return Err(McpMemoryError::Config {
                    message: "reranker.candidate_multiplier must be > 0".into(),
                });
            }
            if let Some(ref key_env) = rr.api_key_env {
                if key_env.is_empty() {
                    return Err(McpMemoryError::Config {
                        message: "reranker.api_key_env must not be empty when specified".into(),
                    });
                }
            }
        }

        if let Some(ref wp) = self.write_policy {
            if !(0.0..=1.0).contains(&wp.near_dup_threshold) {
                return Err(McpMemoryError::Config {
                    message: "write_policy.near_dup_threshold must be in [0.0, 1.0]".into(),
                });
            }
        }

        if let Some(ref f) = self.fusion {
            if matches!(f.mode, FusionMode::Rrf) && f.rrf_k <= 0.0 {
                return Err(McpMemoryError::Config {
                    message: "fusion.rrf_k must be > 0 for mode 'rrf'".into(),
                });
            }
        }

        if let Some(ref lc) = self.lifecycle {
            if lc.enabled && lc.budget_per_category == 0 {
                return Err(McpMemoryError::Config {
                    message: "lifecycle.budget_per_category must be > 0 when enabled".into(),
                });
            }
            if !(0.0..=1.0).contains(&lc.delete_below) {
                return Err(McpMemoryError::Config {
                    message: "lifecycle.delete_below must be in [0.0, 1.0]".into(),
                });
            }
        }

        if let Some(ref d) = self.diversify {
            if !(0.0..=1.0).contains(&d.lambda) {
                return Err(McpMemoryError::Config {
                    message: "diversify.lambda must be in [0.0, 1.0]".into(),
                });
            }
        }

        if let Some(ref c) = self.contradiction {
            if !(0.0..=1.0).contains(&c.min_similarity) {
                return Err(McpMemoryError::Config {
                    message: "contradiction.min_similarity must be in [0.0, 1.0]".into(),
                });
            }
        }

        // Strict local-first by default: a persistent model of a real person
        // stays on-device unless `avatar.allow_remote` explicitly opts in.
        if let Some(ref a) = self.avatar {
            if a.enabled && matches!(self.database.mode, DatabaseMode::Remote) && !a.allow_remote {
                return Err(McpMemoryError::Config {
                    message: "avatar.enabled with a remote database requires \
                              avatar.allow_remote = true (the user model is local-first by \
                              default; set allow_remote to persist it to a trusted remote store)"
                        .into(),
                });
            }
        }

        if let Some(ref ms) = self.multi_signal {
            if !(0.0..=1.0).contains(&ms.lexical_weight) {
                return Err(McpMemoryError::Config {
                    message: "multi_signal.lexical_weight must be in [0.0, 1.0]".into(),
                });
            }
        }

        Ok(())
    }
}

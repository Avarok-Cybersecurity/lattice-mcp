use super::*;

fn test_config() -> Config {
    Config {
        database: DatabaseConfig {
            mode: DatabaseMode::Embedded,
            data_dir: PathBuf::from("/tmp/lattice-mcp-test"),
            remote_url: None,
        },
        embedding: EmbeddingConfig {
            endpoint: "https://api.openai.com/v1/embeddings".into(),
            model: "text-embedding-3-small".into(),
            dimensions: 1536,
            api_key_env: Some("OPENAI_API_KEY".into()),
        },
        memory: MemoryConfig {
            episodic_decay_half_life_days: 30.0,
            semantic_decay_half_life_days: 365.0,
            procedural_decay_half_life_days: 180.0,
            consolidation_threshold: 10,
            max_retrieval_results: 20,
            max_pinned_memories: 10,
            max_pinned_content_length: 500,
        },
        retrieval: RetrievalConfig {
            vector_weight: 0.4,
            graph_weight: 0.25,
            recency_weight: 0.2,
            importance_weight: 0.15,
            graph_hop_depth: 2,
        },
        consolidation: ConsolidationConfig {
            mode: ConsolidationMode::Auto,
            chat_endpoint: None,
            chat_model: None,
            min_importance: 0.4,
            temporal_half_life_days: 7.0,
            distill_enabled: None,
            auto_consolidate: None,
        },
        reranker: None,
        importance: None,
        write_policy: None,
        fusion: None,
        lifecycle: None,
        diversify: None,
        contradiction: None,
        query_transform: None,
        avatar: None,
        multi_signal: None,
    }
}

#[test]
fn valid_config_passes() {
    assert!(test_config().validate().is_ok());
}

#[test]
fn zero_dimensions_fails() {
    let mut cfg = test_config();
    cfg.embedding.dimensions = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn bad_weights_fails() {
    let mut cfg = test_config();
    cfg.retrieval.vector_weight = 0.5;
    // sum now 0.5 + 0.25 + 0.2 + 0.15 = 1.1
    assert!(cfg.validate().is_err());
}

#[test]
fn remote_without_url_fails() {
    let mut cfg = test_config();
    cfg.database.mode = DatabaseMode::Remote;
    cfg.database.remote_url = None;
    assert!(cfg.validate().is_err());
}

fn test_reranker_config() -> RerankerConfig {
    RerankerConfig {
        endpoint: "http://localhost:8081/v1/rerank".into(),
        model: "test-model".into(),
        api_format: RerankerApiFormat::Openai,
        api_key_env: None,
        reranker_weight: 0.6,
        candidate_multiplier: 3,
    }
}

#[test]
fn reranker_config_validates() {
    let mut cfg = test_config();
    cfg.reranker = Some(test_reranker_config());
    assert!(cfg.validate().is_ok());
}

#[test]
fn reranker_bad_weight_fails() {
    let mut cfg = test_config();
    let mut rr = test_reranker_config();
    rr.reranker_weight = 1.5;
    cfg.reranker = Some(rr);
    assert!(cfg.validate().is_err());
}

#[test]
fn reranker_zero_multiplier_fails() {
    let mut cfg = test_config();
    let mut rr = test_reranker_config();
    rr.candidate_multiplier = 0;
    cfg.reranker = Some(rr);
    assert!(cfg.validate().is_err());
}

#[test]
fn bad_min_importance_fails() {
    let mut cfg = test_config();
    cfg.consolidation.min_importance = 1.5;
    assert!(cfg.validate().is_err());
}

#[test]
fn zero_temporal_half_life_fails() {
    let mut cfg = test_config();
    cfg.consolidation.temporal_half_life_days = 0.0;
    assert!(cfg.validate().is_err());
}

// --- New opt-in feature configs: fail-fast validation (PCND) ---

#[test]
fn bad_write_policy_threshold_fails() {
    let mut cfg = test_config();
    cfg.write_policy = Some(WritePolicyConfig {
        enabled: true,
        near_dup_threshold: 1.5,
        skip_near_dups: false,
    });
    assert!(cfg.validate().is_err());
}

#[test]
fn bad_fusion_rrf_k_fails() {
    let mut cfg = test_config();
    cfg.fusion = Some(FusionConfig {
        enabled: true,
        mode: FusionMode::Rrf,
        rrf_k: 0.0,
    });
    assert!(cfg.validate().is_err());
}

#[test]
fn zero_lifecycle_budget_fails() {
    let mut cfg = test_config();
    cfg.lifecycle = Some(LifecycleConfig {
        enabled: true,
        budget_per_category: 0,
        allow_delete: false,
        delete_below: 0.3,
        prune_on_session_end: false,
    });
    assert!(cfg.validate().is_err());
}

#[test]
fn bad_diversify_lambda_fails() {
    let mut cfg = test_config();
    cfg.diversify = Some(DiversifyConfig {
        enabled: true,
        lambda: 2.0,
    });
    assert!(cfg.validate().is_err());
}

#[test]
fn bad_contradiction_similarity_fails() {
    let mut cfg = test_config();
    cfg.contradiction = Some(ContradictionConfig {
        enabled: true,
        min_similarity: 1.5,
        auto_supersede: false,
    });
    assert!(cfg.validate().is_err());
}

#[test]
fn valid_new_feature_configs_pass() {
    let mut cfg = test_config();
    cfg.write_policy = Some(WritePolicyConfig {
        enabled: true,
        near_dup_threshold: 0.9,
        skip_near_dups: true,
    });
    cfg.fusion = Some(FusionConfig {
        enabled: true,
        mode: FusionMode::ConfidenceWeighted,
        rrf_k: 60.0,
    });
    cfg.lifecycle = Some(LifecycleConfig {
        enabled: true,
        budget_per_category: 100,
        allow_delete: false,
        delete_below: 0.2,
        prune_on_session_end: true,
    });
    cfg.diversify = Some(DiversifyConfig {
        enabled: true,
        lambda: 0.7,
    });
    cfg.contradiction = Some(ContradictionConfig {
        enabled: true,
        min_similarity: 0.85,
        auto_supersede: false,
    });
    cfg.query_transform = Some(QueryTransformConfig { enabled: true });
    assert!(cfg.validate().is_ok());
}

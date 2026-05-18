//! Test config builder — all fields explicit (PCND).

use std::path::Path;

use lattice_mcp::config::{
    Config, ConsolidationConfig, ConsolidationMode, DatabaseConfig, DatabaseMode,
    EmbeddingConfig, FsrsConfig, HooksConfig, HooksMode, ImportanceConfig,
    ImportanceWeights, MemoryConfig, RetrievalConfig,
};

/// Build a Config pointing at the given temp directory with importance enabled.
pub fn make_test_config(data_dir: &Path) -> Config {
    Config {
        database: DatabaseConfig {
            mode: DatabaseMode::Embedded,
            data_dir: data_dir.to_path_buf(),
            remote_url: None,
        },
        embedding: EmbeddingConfig {
            endpoint: "mock://unused".into(),
            model: "mock-bag-of-words".into(),
            dimensions: 64,
            api_key_env: None,
        },
        memory: MemoryConfig {
            episodic_decay_half_life_days: 30.0,
            semantic_decay_half_life_days: 365.0,
            procedural_decay_half_life_days: 180.0,
            consolidation_threshold: 5,
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
            mode: ConsolidationMode::Algorithmic,
            chat_endpoint: None,
            chat_model: None,
            min_importance: 0.4,
            temporal_half_life_days: 7.0,
        },
        hooks: HooksConfig {
            mode: HooksMode::Hybrid,
            auto_capture_patterns: vec!["Edit".into(), "Write".into(), "Bash".into()],
            auto_importance: 0.3,
        },
        reranker: None,
        importance: Some(ImportanceConfig {
            enabled: true,
            weights: ImportanceWeights::default(),
            fsrs: FsrsConfig::default(),
            pagerank_damping: 0.85,
            pagerank_iterations: 20,
        }),
    }
}

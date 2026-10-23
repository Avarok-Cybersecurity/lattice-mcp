//! Shared benchmark driver: system builder + scenario measurement.
//! Split out of `benchmark.rs` so both benchmark test crates stay ≤250.
#![allow(dead_code)]

use lattice_mcp::config::Config;
use lattice_mcp::memory::MemorySystem;
use lattice_mcp::store::embedded::EmbeddedStore;

use super::bench_fixtures::{run_scenario, Metrics};
use super::bench_scenarios::auth_project_scenario;
use super::mock_embedder::MockEmbedder;

/// Build a fresh MemorySystem (EmbeddedStore + MockEmbedder, no chat/reranker).
pub async fn make_system(data_dir: &std::path::Path, config: Config) -> MemorySystem {
    let store = EmbeddedStore::new(data_dir.to_path_buf(), 64);
    let mut sys = MemorySystem::new(Box::new(store), Box::new(MockEmbedder), None, None, config);
    sys.initialize().await.expect("initialize");
    sys
}

/// Run the canonical scenario once with the given config; returns metrics.
pub async fn measure(config: Config) -> Metrics {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut sys = make_system(tmp.path(), config).await;
    run_scenario(&mut sys, &auth_project_scenario()).await
}

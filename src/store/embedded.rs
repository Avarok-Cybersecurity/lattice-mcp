//! Embedded LatticeDB store — in-process using lattice-core + DiskStorage.

use async_trait::async_trait;
use std::collections::HashMap;
use std::path::PathBuf;

use lattice_core::engine::collection::CollectionEngineBuilder;
use lattice_core::types::collection::{CollectionConfig, Distance, HnswConfig, VectorConfig};
use lattice_core::types::point::Point;
use lattice_core::types::query::SearchQuery;
use lattice_core::CollectionEngine;
use lattice_storage::DiskStorage;

use crate::error::{McpMemoryError, Result};
use crate::memory::types::{relations, Memory, MemoryCategory, SearchHit, TraversalHit};

use super::MemoryStore;

type DurableEngine = CollectionEngine<DiskStorage, DiskStorage>;

/// Embedded store wrapping three durable LatticeDB collection engines.
pub struct EmbeddedStore {
    engines: HashMap<MemoryCategory, DurableEngine>,
    data_dir: PathBuf,
    dimensions: usize,
}

impl EmbeddedStore {
    pub fn new(data_dir: PathBuf, dimensions: usize) -> Self {
        Self {
            engines: HashMap::new(),
            data_dir,
            dimensions,
        }
    }

    fn engine(&self, category: MemoryCategory) -> Result<&DurableEngine> {
        self.engines
            .get(&category)
            .ok_or_else(|| McpMemoryError::CollectionNotInitialized {
                name: category.collection_name().into(),
            })
    }

    fn engine_mut(&mut self, category: MemoryCategory) -> Result<&mut DurableEngine> {
        self.engines
            .get_mut(&category)
            .ok_or_else(|| McpMemoryError::CollectionNotInitialized {
                name: category.collection_name().into(),
            })
    }

    fn build_config(&self, category: MemoryCategory) -> CollectionConfig {
        let hnsw = HnswConfig {
            m: 16,
            m0: 32,
            ml: HnswConfig::recommended_ml(16),
            ef: 100,
            ef_construction: 200,
        };

        let mut config = CollectionConfig::new(
            category.collection_name(),
            VectorConfig::new(self.dimensions, Distance::Cosine),
            hnsw,
        );

        // Register relations from the single-source-of-truth registry.
        for (name, id) in relations::relations_for(category) {
            config = config.with_relation(name, id);
        }

        config
    }

    async fn open_or_create(&self, category: MemoryCategory) -> Result<DurableEngine> {
        let name = category.collection_name();
        let collection_dir = self.data_dir.join(name);
        let wal_dir = collection_dir.join("wal");
        let data_dir = collection_dir.join("data");

        // Create directories
        tokio::fs::create_dir_all(&wal_dir).await?;
        tokio::fs::create_dir_all(&data_dir).await?;

        // Initialize storage backends
        let wal_storage = DiskStorage::with_defaults(wal_dir);
        wal_storage
            .init()
            .await
            .map_err(|e| McpMemoryError::Store {
                message: format!("WAL storage init failed for {name}: {e}"),
            })?;

        let data_storage = DiskStorage::with_defaults(data_dir);
        data_storage
            .init()
            .await
            .map_err(|e| McpMemoryError::Store {
                message: format!("Data storage init failed for {name}: {e}"),
            })?;

        let config = self.build_config(category);

        let engine = CollectionEngineBuilder::new(config)
            .with_wal(wal_storage)
            .with_data(data_storage)
            .open()
            .await?;

        Ok(engine)
    }
}

#[async_trait]
impl MemoryStore for EmbeddedStore {
    async fn initialize(&mut self) -> Result<()> {
        for &category in MemoryCategory::all_storage() {
            let engine = self.open_or_create(category).await?;
            let count = engine.point_count();
            tracing::info!(
                collection = category.collection_name(),
                points = count,
                "Initialized collection"
            );
            self.engines.insert(category, engine);
        }
        Ok(())
    }

    async fn upsert_memory(&mut self, memory: &Memory, vector: Vec<f32>) -> Result<()> {
        let engine = self.engine_mut(memory.category)?;

        // `Point::new_vector` leaves `outgoing_edges` as None, which lattice-core
        // treats as "update point STATE only, keep existing edges" — so a
        // state-only re-upsert (rate_recall, update_pagerank, prune-demote,
        // update_memory, avatar re-observation) never erases the node's edges.
        let mut point = Point::new_vector(memory.id, vector);
        for (key, value) in memory.to_payload() {
            point = point.with_field(key, value);
        }

        engine.upsert_points_async(vec![point]).await?;
        Ok(())
    }

    async fn vector_search(
        &self,
        category: MemoryCategory,
        query_vector: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchHit>> {
        let engine = self.engine(category)?;
        let query = SearchQuery::new(query_vector.to_vec(), limit);
        let results = engine.search(query)?;

        Ok(results
            .into_iter()
            .map(|r| SearchHit {
                id: r.id,
                score: r.score,
                category,
            })
            .collect())
    }

    async fn get_memory(&self, category: MemoryCategory, id: u64) -> Result<Option<Memory>> {
        let engine = self.engine(category)?;
        let points = engine.get_points(&[id])?;

        match points.into_iter().next().flatten() {
            Some(point) => {
                let mem = Memory::from_payload(point.id, &point.payload)?;
                Ok(Some(mem))
            }
            None => Ok(None),
        }
    }

    async fn get_memories(&self, category: MemoryCategory, ids: &[u64]) -> Result<Vec<Memory>> {
        let engine = self.engine(category)?;
        let points = engine.get_points(ids)?;
        let mut out = Vec::new();
        for point in points.into_iter().flatten() {
            match Memory::from_payload(point.id, &point.payload) {
                Ok(m) => out.push(m),
                Err(e) => tracing::warn!(id = point.id, "Failed to deserialize point: {e}"),
            }
        }
        Ok(out)
    }

    async fn get_vectors(
        &self,
        category: MemoryCategory,
        ids: &[u64],
    ) -> Result<Vec<(u64, Vec<f32>)>> {
        let engine = self.engine(category)?;
        let points = engine.get_points(ids)?;
        Ok(points
            .into_iter()
            .flatten()
            .map(|p| (p.id, p.vector))
            .collect())
    }

    async fn traverse(
        &self,
        category: MemoryCategory,
        start_id: u64,
        max_depth: usize,
        relations: Option<&[&str]>,
    ) -> Result<Vec<TraversalHit>> {
        let engine = self.engine(category)?;
        let result = engine.traverse(start_id, max_depth, relations)?;

        Ok(result
            .paths
            .into_iter()
            .map(|path| TraversalHit {
                target_id: path.target_id,
                depth: path.depth,
                relation: relations::relation_name(category, path.relation_id).to_string(),
            })
            .collect())
    }

    async fn add_edge(
        &mut self,
        category: MemoryCategory,
        from_id: u64,
        to_id: u64,
        relation: &str,
        weight: f32,
    ) -> Result<()> {
        let engine = self.engine_mut(category)?;
        engine
            .add_edge_async(from_id, to_id, relation, weight)
            .await?;
        Ok(())
    }

    async fn remove_edge(
        &mut self,
        category: MemoryCategory,
        from_id: u64,
        to_id: u64,
        relation: Option<&str>,
    ) -> Result<bool> {
        let engine = self.engine_mut(category)?;
        let removed = engine.remove_edge_async(from_id, to_id, relation).await?;
        Ok(removed)
    }

    async fn delete_memory(&mut self, category: MemoryCategory, id: u64) -> Result<bool> {
        let engine = self.engine_mut(category)?;
        let deleted = engine.delete_points_async(&[id]).await?;
        Ok(deleted > 0)
    }

    async fn list_all(&self, category: MemoryCategory) -> Result<Vec<Memory>> {
        let engine = self.engine(category)?;
        let ids = engine.point_ids()?;
        let points = engine.get_points(&ids)?;
        let mut memories = Vec::new();
        for point in points.into_iter().flatten() {
            match Memory::from_payload(point.id, &point.payload) {
                Ok(mem) => memories.push(mem),
                Err(e) => {
                    tracing::warn!(id = point.id, "Failed to deserialize point: {e}");
                }
            }
        }
        Ok(memories)
    }

    async fn count(&self, category: MemoryCategory) -> Result<usize> {
        let engine = self.engine(category)?;
        Ok(engine.point_count())
    }

    async fn total_count(&self) -> Result<usize> {
        let mut total = 0;
        for &cat in MemoryCategory::all_storage() {
            if let Ok(engine) = self.engine(cat) {
                total += engine.point_count();
            }
        }
        Ok(total)
    }
}

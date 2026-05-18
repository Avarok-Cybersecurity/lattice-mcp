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
use crate::memory::types::{
    relations, Memory, MemoryCategory, SearchHit, TraversalHit,
};

use super::MemoryStore;

type DurableEngine = CollectionEngine<DiskStorage, DiskStorage>;

/// Reverse-map a (category, relation_id) pair back to the relation name string.
fn relation_name(category: MemoryCategory, id: u16) -> String {
    match (category, id) {
        (MemoryCategory::Episodic, 0) => relations::PRECEDED_BY,
        (MemoryCategory::Episodic, 1) => relations::CAUSED,
        (MemoryCategory::Episodic, 2) => relations::RESOLVED_BY,
        (MemoryCategory::Episodic, 3) => relations::LEARNED_FROM,
        (MemoryCategory::Episodic, 4) => relations::EVIDENCED_BY,
        (MemoryCategory::Semantic, 0) => relations::RELATES_TO,
        (MemoryCategory::Semantic, 1) => relations::DEPENDS_ON,
        (MemoryCategory::Semantic, 2) => relations::CONTRADICTS,
        (MemoryCategory::Semantic, 3) => relations::SUPERSEDES,
        (MemoryCategory::Semantic, 4) => relations::USES_KNOWLEDGE,
        (MemoryCategory::Procedural, 0) => relations::APPLIES_TO,
        (MemoryCategory::Procedural, 1) => relations::ALTERNATIVE_TO,
        (MemoryCategory::Procedural, 2) => relations::REQUIRES,
        // Pinned has no relations
        _ => "UNKNOWN",
    }
    .to_string()
}

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
        self.engines.get(&category).ok_or_else(|| {
            McpMemoryError::CollectionNotInitialized {
                name: category.collection_name().into(),
            }
        })
    }

    fn engine_mut(&mut self, category: MemoryCategory) -> Result<&mut DurableEngine> {
        self.engines.get_mut(&category).ok_or_else(|| {
            McpMemoryError::CollectionNotInitialized {
                name: category.collection_name().into(),
            }
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

        // Register relations per category
        match category {
            MemoryCategory::Episodic => {
                config = config
                    .with_relation(relations::PRECEDED_BY, 0)
                    .with_relation(relations::CAUSED, 1)
                    .with_relation(relations::RESOLVED_BY, 2)
                    .with_relation(relations::LEARNED_FROM, 3)
                    .with_relation(relations::EVIDENCED_BY, 4);
            }
            MemoryCategory::Semantic => {
                config = config
                    .with_relation(relations::RELATES_TO, 0)
                    .with_relation(relations::DEPENDS_ON, 1)
                    .with_relation(relations::CONTRADICTS, 2)
                    .with_relation(relations::SUPERSEDES, 3)
                    .with_relation(relations::USES_KNOWLEDGE, 4);
            }
            MemoryCategory::Procedural => {
                config = config
                    .with_relation(relations::APPLIES_TO, 0)
                    .with_relation(relations::ALTERNATIVE_TO, 1)
                    .with_relation(relations::REQUIRES, 2);
            }
            MemoryCategory::Pinned => {
                // No relations for pinned memories
            }
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
        wal_storage.init().await.map_err(|e| McpMemoryError::Store {
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
        for &category in MemoryCategory::all() {
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
                relation: relation_name(category, path.relation_id),
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
        let removed = engine
            .remove_edge_async(from_id, to_id, relation)
            .await?;
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
        for &cat in MemoryCategory::all() {
            if let Ok(engine) = self.engine(cat) {
                total += engine.point_count();
            }
        }
        Ok(total)
    }
}

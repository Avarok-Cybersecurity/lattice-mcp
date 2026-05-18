//! Remote LatticeDB store — HTTP client to a running LatticeDB server.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::{McpMemoryError, Result};
use crate::memory::types::{
    relations, Memory, MemoryCategory, SearchHit, TraversalHit,
};

use super::MemoryStore;

/// Remote store connecting to a LatticeDB HTTP API.
pub struct RemoteStore {
    client: reqwest::Client,
    base_url: String,
    dimensions: usize,
}

// --- DTO types matching LatticeDB REST API ---

#[derive(Serialize)]
struct CreateCollectionBody {
    vectors: VectorsDto,
    hnsw_config: HnswDto,
    #[serde(skip_serializing_if = "Option::is_none")]
    relations: Option<std::collections::HashMap<String, u16>>,
}

#[derive(Serialize)]
struct VectorsDto {
    size: usize,
    distance: String,
}

#[derive(Serialize)]
struct HnswDto {
    m: usize,
    m0: usize,
    ef_construct: usize,
}

#[derive(Serialize)]
struct UpsertBody {
    points: Vec<PointDto>,
}

#[derive(Serialize)]
struct PointDto {
    id: u64,
    vector: Vec<f32>,
    payload: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Serialize)]
struct SearchBody {
    vector: Vec<f32>,
    limit: usize,
    with_payload: bool,
}

#[derive(Deserialize)]
struct SearchResponse {
    result: Vec<SearchResultDto>,
}

#[derive(Deserialize)]
struct SearchResultDto {
    id: u64,
    score: f32,
}

#[derive(Serialize)]
struct EdgeBody {
    from_id: u64,
    to_id: u64,
    relation: String,
    weight: f32,
}

#[derive(Serialize)]
struct TraverseBody {
    start_id: u64,
    max_depth: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    relations: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct TraverseResponse {
    result: TraverseResultDto,
}

#[derive(Deserialize)]
struct TraverseResultDto {
    edges: Vec<EdgeRecordDto>,
}

#[derive(Deserialize)]
struct EdgeRecordDto {
    target_id: u64,
    depth: usize,
    relation: String,
}

#[derive(Deserialize)]
struct PointResponse {
    result: Vec<Option<PointResultDto>>,
}

#[derive(Deserialize)]
struct PointResultDto {
    id: u64,
    payload: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct CollectionInfoResponse {
    result: CollectionInfoDto,
}

#[derive(Deserialize)]
struct CollectionInfoDto {
    points_count: usize,
}

impl RemoteStore {
    pub fn new(base_url: String, dimensions: usize) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            dimensions,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    fn collection_url(&self, category: MemoryCategory) -> String {
        format!(
            "{}/collections/{}",
            self.base_url,
            category.collection_name()
        )
    }

    fn build_relations(category: MemoryCategory) -> std::collections::HashMap<String, u16> {
        let mut rels = std::collections::HashMap::new();
        match category {
            MemoryCategory::Episodic => {
                rels.insert(relations::PRECEDED_BY.into(), 0);
                rels.insert(relations::CAUSED.into(), 1);
                rels.insert(relations::RESOLVED_BY.into(), 2);
                rels.insert(relations::LEARNED_FROM.into(), 3);
                rels.insert(relations::EVIDENCED_BY.into(), 4);
            }
            MemoryCategory::Semantic => {
                rels.insert(relations::RELATES_TO.into(), 0);
                rels.insert(relations::DEPENDS_ON.into(), 1);
                rels.insert(relations::CONTRADICTS.into(), 2);
                rels.insert(relations::SUPERSEDES.into(), 3);
                rels.insert(relations::USES_KNOWLEDGE.into(), 4);
            }
            MemoryCategory::Procedural => {
                rels.insert(relations::APPLIES_TO.into(), 0);
                rels.insert(relations::ALTERNATIVE_TO.into(), 1);
                rels.insert(relations::REQUIRES.into(), 2);
            }
            MemoryCategory::Pinned => {
                // No relations for pinned memories
            }
        }
        rels
    }

    /// Convert a PointResultDto payload (JSON Values) to the byte-based payload Memory expects.
    fn convert_payload(
        payload: &std::collections::HashMap<String, serde_json::Value>,
    ) -> std::collections::HashMap<String, Vec<u8>> {
        payload
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::to_vec(v).unwrap_or_default()))
            .collect()
    }
}

#[async_trait]
impl MemoryStore for RemoteStore {
    async fn initialize(&mut self) -> Result<()> {
        for &category in MemoryCategory::all() {
            let url = self.collection_url(category);

            // Check if collection exists
            let resp = self.client.get(&url).send().await;
            if let Ok(r) = resp {
                if r.status().is_success() {
                    tracing::info!(
                        collection = category.collection_name(),
                        "Collection already exists"
                    );
                    continue;
                }
            }

            // Create collection
            let body = CreateCollectionBody {
                vectors: VectorsDto {
                    size: self.dimensions,
                    distance: "Cosine".into(),
                },
                hnsw_config: HnswDto {
                    m: 16,
                    m0: 32,
                    ef_construct: 200,
                },
                relations: Some(Self::build_relations(category)),
            };

            let resp = self
                .client
                .put(&url)
                .json(&body)
                .send()
                .await
                .map_err(|e| McpMemoryError::Store {
                    message: format!("Failed to create collection: {e}"),
                })?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(McpMemoryError::Store {
                    message: format!("Create collection failed ({status}): {body}"),
                });
            }

            tracing::info!(
                collection = category.collection_name(),
                "Created collection"
            );
        }
        Ok(())
    }

    async fn upsert_memory(&mut self, memory: &Memory, vector: Vec<f32>) -> Result<()> {
        let url = format!(
            "{}/points",
            self.collection_url(memory.category)
        );

        // Convert byte-based payload to JSON values for the REST API
        let json_payload: std::collections::HashMap<String, serde_json::Value> = memory
            .to_payload()
            .into_iter()
            .map(|(k, v)| {
                let val = serde_json::from_slice(&v).unwrap_or(serde_json::Value::Null);
                (k, val)
            })
            .collect();

        let body = UpsertBody {
            points: vec![PointDto {
                id: memory.id,
                vector,
                payload: json_payload,
            }],
        };

        let resp = self
            .client
            .put(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| McpMemoryError::Store {
                message: format!("Upsert failed: {e}"),
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(McpMemoryError::Store {
                message: format!("Upsert failed ({status}): {body}"),
            });
        }

        Ok(())
    }

    async fn vector_search(
        &self,
        category: MemoryCategory,
        query_vector: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchHit>> {
        let url = format!(
            "{}/points/search",
            self.collection_url(category)
        );

        let body = SearchBody {
            vector: query_vector.to_vec(),
            limit,
            with_payload: false,
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| McpMemoryError::Store {
                message: format!("Search failed: {e}"),
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(McpMemoryError::Store {
                message: format!("Search failed ({status}): {body}"),
            });
        }

        let parsed: SearchResponse = resp.json().await.map_err(|e| McpMemoryError::Store {
            message: format!("Failed to parse search response: {e}"),
        })?;

        Ok(parsed
            .result
            .into_iter()
            .map(|r| SearchHit {
                id: r.id,
                score: r.score,
                category,
            })
            .collect())
    }

    async fn get_memory(&self, category: MemoryCategory, id: u64) -> Result<Option<Memory>> {
        let url = format!(
            "{}/points",
            self.collection_url(category)
        );

        let body = serde_json::json!({ "ids": [id], "with_payload": true });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| McpMemoryError::Store {
                message: format!("Get point failed: {e}"),
            })?;

        if !resp.status().is_success() {
            return Ok(None);
        }

        let parsed: PointResponse = resp.json().await.map_err(|e| McpMemoryError::Store {
            message: format!("Failed to parse point response: {e}"),
        })?;

        match parsed.result.into_iter().next().flatten() {
            Some(p) => {
                let byte_payload = Self::convert_payload(&p.payload);
                let mem = Memory::from_payload(p.id, &byte_payload)?;
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
        let url = format!(
            "{}/graph/traverse",
            self.collection_url(category)
        );

        let body = TraverseBody {
            start_id,
            max_depth,
            relations: relations.map(|r| r.iter().map(|s| s.to_string()).collect()),
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| McpMemoryError::Store {
                message: format!("Traverse failed: {e}"),
            })?;

        if !resp.status().is_success() {
            return Ok(Vec::new());
        }

        let parsed: TraverseResponse = resp.json().await.map_err(|e| McpMemoryError::Store {
            message: format!("Failed to parse traverse response: {e}"),
        })?;

        Ok(parsed
            .result
            .edges
            .into_iter()
            .map(|e| TraversalHit {
                target_id: e.target_id,
                depth: e.depth,
                relation: e.relation,
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
        let url = format!(
            "{}/graph/edges",
            self.collection_url(category)
        );

        let body = EdgeBody {
            from_id,
            to_id,
            relation: relation.to_string(),
            weight,
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| McpMemoryError::Store {
                message: format!("Add edge failed: {e}"),
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(McpMemoryError::Store {
                message: format!("Add edge failed ({status}): {body}"),
            });
        }

        Ok(())
    }

    async fn remove_edge(
        &mut self,
        _category: MemoryCategory,
        _from_id: u64,
        _to_id: u64,
        _relation: Option<&str>,
    ) -> Result<bool> {
        // LatticeDB REST API doesn't have a direct remove_edge endpoint yet
        // This is a known limitation of the remote store
        Ok(false)
    }

    async fn delete_memory(&mut self, category: MemoryCategory, id: u64) -> Result<bool> {
        let url = format!(
            "{}/points/delete",
            self.collection_url(category)
        );

        let body = serde_json::json!({ "points": [id] });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| McpMemoryError::Store {
                message: format!("Delete failed: {e}"),
            })?;

        Ok(resp.status().is_success())
    }

    async fn list_all(&self, category: MemoryCategory) -> Result<Vec<Memory>> {
        // Use scroll API to retrieve all points
        let url = format!("{}/points/scroll", self.collection_url(category));
        let body = serde_json::json!({ "limit": 100, "with_payload": true });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| McpMemoryError::Store {
                message: format!("Scroll failed: {e}"),
            })?;

        if !resp.status().is_success() {
            return Ok(Vec::new());
        }

        let parsed: PointResponse = resp.json().await.map_err(|e| McpMemoryError::Store {
            message: format!("Failed to parse scroll response: {e}"),
        })?;

        let mut memories = Vec::new();
        for point in parsed.result.into_iter().flatten() {
            let byte_payload = Self::convert_payload(&point.payload);
            match Memory::from_payload(point.id, &byte_payload) {
                Ok(mem) => memories.push(mem),
                Err(e) => {
                    tracing::warn!(id = point.id, "Failed to deserialize point: {e}");
                }
            }
        }
        Ok(memories)
    }

    async fn count(&self, category: MemoryCategory) -> Result<usize> {
        let url = self.collection_url(category);

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| McpMemoryError::Store {
                message: format!("Get collection info failed: {e}"),
            })?;

        if !resp.status().is_success() {
            return Ok(0);
        }

        let parsed: CollectionInfoResponse =
            resp.json().await.map_err(|e| McpMemoryError::Store {
                message: format!("Failed to parse collection info: {e}"),
            })?;

        Ok(parsed.result.points_count)
    }

    async fn total_count(&self) -> Result<usize> {
        let mut total = 0;
        for &cat in MemoryCategory::all() {
            total += self.count(cat).await?;
        }
        Ok(total)
    }
}

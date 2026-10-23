//! Remote LatticeDB store — HTTP client to a running LatticeDB server.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::{McpMemoryError, Result};
use crate::memory::types::{relations, Memory, MemoryCategory, SearchHit, TraversalHit};

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

// Scroll envelope: server returns ApiResponse<ScrollResult>
// = { result: { points: [...], next_page_offset: Option<u64> } }.
#[derive(Deserialize)]
struct ScrollResponse {
    result: ScrollResultDto,
}

#[derive(Deserialize)]
struct ScrollResultDto {
    points: Vec<ScrollPointDto>,
    #[serde(default)]
    next_page_offset: Option<u64>,
}

#[derive(Deserialize)]
struct ScrollPointDto {
    id: u64,
    #[serde(default)]
    payload: Option<std::collections::HashMap<String, serde_json::Value>>,
}

#[derive(Deserialize)]
struct TraverseResultDto {
    #[serde(default)]
    edges: Vec<EdgeRecordDto>,
}

// Matches the server's traverse edge shape: { from_id, to_id, relation, weight }.
// `weight` is ignored (serde drops unknown fields). Per-edge depth is NOT
// returned; it is reconstructed by BFS over the edge set.
#[derive(Deserialize)]
struct EdgeRecordDto {
    from_id: u64,
    to_id: u64,
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

// For get_vectors: /points with with_vector:true returns { id, vector }.
#[derive(Deserialize)]
struct VectorResponse {
    result: Vec<Option<VectorPointDto>>,
}

#[derive(Deserialize)]
struct VectorPointDto {
    id: u64,
    #[serde(default)]
    vector: Option<Vec<f32>>,
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
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            base_url: base_url.trim_end_matches('/').to_string(),
            dimensions,
        }
    }

    fn collection_url(&self, category: MemoryCategory) -> String {
        format!(
            "{}/collections/{}",
            self.base_url,
            category.collection_name()
        )
    }

    fn build_relations(category: MemoryCategory) -> std::collections::HashMap<String, u16> {
        // Single source of truth: relations::REGISTRY (see memory/types.rs).
        relations::relations_for(category)
            .map(|(name, id)| (name.to_string(), id))
            .collect()
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
        for &category in MemoryCategory::all_storage() {
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
        let url = format!("{}/points", self.collection_url(memory.category));

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
        let url = format!("{}/points/search", self.collection_url(category));

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
        let url = format!("{}/points", self.collection_url(category));

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

    async fn get_memories(&self, category: MemoryCategory, ids: &[u64]) -> Result<Vec<Memory>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/points", self.collection_url(category));
        let body = serde_json::json!({ "ids": ids, "with_payload": true });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| McpMemoryError::Store {
                message: format!("Get points failed: {e}"),
            })?;

        if !resp.status().is_success() {
            return Ok(Vec::new());
        }

        let parsed: PointResponse = resp.json().await.map_err(|e| McpMemoryError::Store {
            message: format!("Failed to parse points response: {e}"),
        })?;

        let mut out = Vec::new();
        for p in parsed.result.into_iter().flatten() {
            let byte_payload = Self::convert_payload(&p.payload);
            match Memory::from_payload(p.id, &byte_payload) {
                Ok(m) => out.push(m),
                Err(e) => tracing::warn!(id = p.id, "Failed to deserialize point: {e}"),
            }
        }
        Ok(out)
    }

    async fn get_vectors(
        &self,
        category: MemoryCategory,
        ids: &[u64],
    ) -> Result<Vec<(u64, Vec<f32>)>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/points", self.collection_url(category));
        let body = serde_json::json!({ "ids": ids, "with_vector": true, "with_payload": false });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| McpMemoryError::Store {
                message: format!("Get vectors failed: {e}"),
            })?;

        if !resp.status().is_success() {
            return Ok(Vec::new());
        }

        let parsed: VectorResponse = resp.json().await.map_err(|e| McpMemoryError::Store {
            message: format!("Failed to parse vector response: {e}"),
        })?;

        Ok(parsed
            .result
            .into_iter()
            .flatten()
            .filter_map(|p| p.vector.map(|v| (p.id, v)))
            .collect())
    }

    async fn traverse(
        &self,
        category: MemoryCategory,
        start_id: u64,
        max_depth: usize,
        relations: Option<&[&str]>,
    ) -> Result<Vec<TraversalHit>> {
        let url = format!("{}/graph/traverse", self.collection_url(category));

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

        // The server returns a flat edge list ({from_id,to_id,relation}) without
        // per-edge depth. Reconstruct shortest-hop depth from `start_id` by
        // relaxing the edge set up to `max_depth` times (matches the embedded
        // store's path-depth semantics, used by graph-relevance scoring).
        let edges = parsed.result.edges;
        let mut depth_of: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
        depth_of.insert(start_id, 0);
        for _ in 0..max_depth.max(1) {
            for e in &edges {
                if let Some(&d) = depth_of.get(&e.from_id) {
                    let nd = d + 1;
                    depth_of
                        .entry(e.to_id)
                        .and_modify(|x| {
                            if nd < *x {
                                *x = nd
                            }
                        })
                        .or_insert(nd);
                }
            }
        }

        Ok(edges
            .into_iter()
            .map(|e| TraversalHit {
                target_id: e.to_id,
                depth: depth_of.get(&e.to_id).copied().unwrap_or(1),
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
        let url = format!("{}/graph/edges", self.collection_url(category));

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
        // The LatticeDB REST API has no edge-removal endpoint. Surface this as
        // an error rather than a silent `Ok(false)`, so callers don't believe
        // an edge was checked-and-absent when it actually couldn't be touched.
        Err(McpMemoryError::Store {
            message: "remove_edge is unsupported by the remote store (no REST endpoint); \
                      use the embedded store for edge mutation"
                .into(),
        })
    }

    async fn delete_memory(&mut self, category: MemoryCategory, id: u64) -> Result<bool> {
        let url = format!("{}/points/delete", self.collection_url(category));

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
        // Scroll the full collection with offset pagination until exhausted.
        // (A previous version hard-capped at the first 100 points AND parsed
        // the wrong envelope shape, silently truncating prune/PageRank.)
        const PAGE: usize = 512;
        let url = format!("{}/points/scroll", self.collection_url(category));
        let mut memories = Vec::new();
        let mut offset: Option<u64> = None;

        loop {
            let body = serde_json::json!({
                "limit": PAGE,
                "offset": offset,
                "with_payload": true
            });

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
                return Ok(memories);
            }

            let parsed: ScrollResponse = resp.json().await.map_err(|e| McpMemoryError::Store {
                message: format!("Failed to parse scroll response: {e}"),
            })?;

            let page_len = parsed.result.points.len();
            for point in parsed.result.points {
                if let Some(payload) = point.payload {
                    let byte_payload = Self::convert_payload(&payload);
                    match Memory::from_payload(point.id, &byte_payload) {
                        Ok(mem) => memories.push(mem),
                        Err(e) => {
                            tracing::warn!(id = point.id, "Failed to deserialize point: {e}");
                        }
                    }
                }
            }

            // Advance only on a cursor; stop on no-cursor or an empty page
            // (empty-page guard prevents an infinite loop if the server ever
            // returns a stale non-null cursor).
            match parsed.result.next_page_offset {
                Some(next) if page_len > 0 => offset = Some(next),
                _ => break,
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
        for &cat in MemoryCategory::all_storage() {
            total += self.count(cat).await?;
        }
        Ok(total)
    }
}

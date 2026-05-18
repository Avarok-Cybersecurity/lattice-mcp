//! Deterministic bag-of-words embedder for integration tests.
//!
//! No network, no API keys. Texts sharing keywords produce high cosine similarity.

use async_trait::async_trait;
use lattice_mcp::embedding::EmbeddingProvider;
use lattice_mcp::error::Result;

const DIM: usize = 64;

/// Fixed vocabulary: each keyword maps to a specific vector dimension.
const VOCAB: &[(&str, usize)] = &[
    ("auth", 0), ("authentication", 0), ("jwt", 1), ("token", 1),
    ("database", 2), ("schema", 2), ("table", 3), ("users", 3),
    ("password", 4), ("bcrypt", 4), ("hash", 5), ("hashed", 5),
    ("error", 6), ("bug", 6), ("fix", 7), ("fixed", 7),
    ("security", 8), ("https", 8), ("encrypt", 9), ("vulnerability", 9),
    ("api", 10), ("endpoint", 10), ("login", 11), ("register", 11),
    ("post", 12), ("get", 12), ("middleware", 13), ("header", 13),
    ("implement", 14), ("implementation", 14), ("design", 15), ("architecture", 15),
    ("decision", 16), ("chose", 16), ("requirement", 17), ("feature", 17),
    ("session", 18), ("started", 18), ("ended", 19), ("completed", 19),
    ("plan", 20), ("planning", 20), ("review", 21), ("test", 21),
    ("code", 22), ("module", 22), ("function", 23), ("struct", 23),
    ("config", 24), ("setting", 24), ("pattern", 25), ("procedural", 25),
    ("stateless", 26), ("library", 26), ("validation", 27), ("validate", 27),
    ("explicit", 28), ("unwrap", 28), ("production", 29), ("deploy", 29),
    ("project", 30), ("widget", 30), ("app", 31), ("application", 31),
];

pub struct MockEmbedder;

impl MockEmbedder {
    fn embed_text(text: &str) -> Vec<f32> {
        let mut vec = vec![0.0_f32; DIM];
        let lower = text.to_lowercase();

        for word in lower.split(|c: char| !c.is_alphanumeric()) {
            if word.is_empty() {
                continue;
            }
            for &(keyword, dim_idx) in VOCAB {
                if word == keyword {
                    vec[dim_idx] += 1.0;
                    break;
                }
            }
            // Hash remaining words to spread across unused dimensions
            let hash_dim = 32 + (word.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32)) as usize % 32);
            vec[hash_dim] += 0.3;
        }

        // L2 normalize
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        } else {
            // Zero vector fallback: set first dim to 1.0
            vec[0] = 1.0;
        }

        vec
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(Self::embed_text(text))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| Self::embed_text(t)).collect())
    }

    fn dimensions(&self) -> usize {
        DIM
    }
}

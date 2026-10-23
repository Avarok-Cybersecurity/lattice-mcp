//! Live integration tests against the LAN stack (NOT run by default).
//!
//! Endpoints come from the environment (a test whose variables are unset
//! prints a skip notice and passes):
//! - `LATTICE_MCP_LIVE_EMBED_URL`  embeddings URL (nvidia/llama-embed-nemotron-8b, 4096-d)
//! - `LATTICE_MCP_LIVE_RERANK_URL` rerank URL     (jinaai/jina-reranker-v2-base-multilingual)
//! - `LATTICE_MCP_LIVE_DB_URL`     LatticeDB URL  (remote store)
//!
//! No LLM/chat configured (consolidation = algorithmic).
//!
//! Run explicitly:
//!   cargo test --test live_lan -- --ignored --nocapture
//! or a single one:
//!   cargo test --test live_lan -- --ignored embedded_lan_full_stack --nocapture

mod common;

use lattice_mcp::config::{
    AvatarConfig, Config, DatabaseMode, EmbeddingConfig, LifecycleConfig, RerankerApiFormat,
    RerankerConfig,
};
use lattice_mcp::memory::types::MemoryCategory;
use lattice_mcp::memory::MemorySystem;
use lattice_mcp::store::embedded::EmbeddedStore;
use lattice_mcp::store::remote::RemoteStore;

use common::config::make_test_config;
use common::live_env::{endpoints, DB_URL_VAR, EMBED_URL_VAR, RERANK_URL_VAR};

const EMBED_MODEL: &str = "nvidia/llama-embed-nemotron-8b";
const EMBED_DIMS: usize = 4096;
const RERANK_MODEL: &str = "jinaai/jina-reranker-v2-base-multilingual";

fn lan_embedding(url: &str) -> EmbeddingConfig {
    EmbeddingConfig {
        endpoint: url.into(),
        model: EMBED_MODEL.into(),
        dimensions: EMBED_DIMS,
        api_key_env: None,
    }
}

fn lan_reranker(url: &str) -> RerankerConfig {
    RerankerConfig {
        endpoint: url.into(),
        model: RERANK_MODEL.into(),
        api_format: RerankerApiFormat::Jina,
        api_key_env: None,
        reranker_weight: 0.5,
        candidate_multiplier: 3,
    }
}

/// Build the real embedder/reranker providers the way main.rs does.
fn providers(
    cfg: &Config,
) -> (
    Box<dyn lattice_mcp::embedding::EmbeddingProvider>,
    Option<Box<dyn lattice_mcp::reranker::RerankerProvider>>,
) {
    use lattice_mcp::embedding::openai::OpenAiEmbedder;
    use lattice_mcp::reranker::http::HttpReranker;
    let embedder = Box::new(OpenAiEmbedder::new(&cfg.embedding).expect("embedder"));
    let reranker = cfg.reranker.as_ref().map(|rr| {
        Box::new(HttpReranker::new(rr).expect("reranker"))
            as Box<dyn lattice_mcp::reranker::RerankerProvider>
    });
    (embedder, reranker)
}

#[tokio::test]
#[ignore = "live: requires LATTICE_MCP_LIVE_EMBED_URL + LATTICE_MCP_LIVE_RERANK_URL"]
async fn embedded_lan_full_stack() {
    let Some([embed_url, rerank_url]) =
        endpoints("embedded_lan_full_stack", [EMBED_URL_VAR, RERANK_URL_VAR])
    else {
        return;
    };
    let tmp = tempfile::tempdir().unwrap();

    // Embedded store + REAL LAN embedder + REAL LAN reranker; avatar on; no chat.
    let mut cfg = make_test_config(tmp.path());
    cfg.embedding = lan_embedding(&embed_url);
    cfg.reranker = Some(lan_reranker(&rerank_url));
    cfg.avatar = Some(AvatarConfig {
        enabled: true,
        allow_freeform: false,
        redact_patterns: vec![],
        auto_derive: false,
        allow_remote: false,
    });
    cfg.lifecycle = Some(LifecycleConfig {
        enabled: true,
        budget_per_category: 100,
        allow_delete: false,
        delete_below: 0.2,
        prune_on_session_end: false,
    });
    cfg.validate().expect("config valid");

    let (embedder, reranker) = providers(&cfg);
    let store = EmbeddedStore::new(tmp.path().to_path_buf(), EMBED_DIMS);
    let mut sys = MemorySystem::new(Box::new(store), embedder, None, reranker, cfg);
    sys.initialize().await.expect("init");

    // --- project memory: write a few real-embedded memories ---
    sys.session_start(Some("lattice-mcp"), Some("wire up the LAN stack"))
        .await
        .expect("session_start");
    for (content, cat) in [
        (
            "Decided to use the embedded LatticeDB store for the avatar feature",
            MemoryCategory::Semantic,
        ),
        (
            "Fixed remote list_all to paginate via scroll offset instead of capping at 100",
            MemoryCategory::Episodic,
        ),
        (
            "The embedder nvidia/llama-embed-nemotron-8b returns 4096-dim vectors",
            MemoryCategory::Semantic,
        ),
    ] {
        sys.remember(content, cat, Some(0.8), Some("decision"), None, None)
            .await
            .expect("remember");
    }

    // --- recall (real embeddings + cross-encoder rerank) ---
    let hits = sys
        .recall("how are remote results paginated", &[], 5)
        .await
        .expect("recall");
    assert!(
        !hits.is_empty(),
        "recall returned nothing from a live embedder"
    );
    let top = &hits[0].memory.content;
    eprintln!(
        "[live] top recall: {top}  (score {:.3})",
        hits[0].final_score
    );
    assert!(
        hits.iter().any(|h| h.memory.content.contains("paginate")),
        "expected the pagination memory to surface; got: {:?}",
        hits.iter().map(|h| &h.memory.content).collect::<Vec<_>>()
    );

    // --- recall feedback ---
    let id = lattice_mcp::memory::types::Memory::content_id(
        "Fixed remote list_all to paginate via scroll offset instead of capping at 100",
        MemoryCategory::Episodic,
    );
    sys.rate_recall(MemoryCategory::Episodic, id, 5)
        .await
        .expect("rate_recall");

    // --- avatar roundtrip (real 4096-d embeddings) ---
    sys.avatar_store("default", "preferred_language", "Rust", &[])
        .await
        .expect("avatar_store");
    sys.avatar_store("default", "communication_tone", "concise and direct", &[])
        .await
        .expect("avatar_store");
    // reinforce -> sharpen
    let again = sys
        .avatar_store("default", "preferred_language", "Rust", &[])
        .await
        .expect("avatar_store reinforce");
    eprintln!(
        "[live] avatar preferred_language confidence after 2 obs: {:.3}",
        again.importance
    );

    let slots = sys.avatar_get("default").await.expect("avatar_get");
    assert_eq!(slots.len(), 2, "two distinct avatar slots");
    let asearch = sys
        .avatar_search("default", "what language does the user like", 5)
        .await
        .expect("avatar_search");
    assert!(
        asearch.iter().any(|s| s.memory.content.contains("Rust")),
        "avatar_search should surface preferred_language"
    );

    // --- consolidation (algorithmic; no LLM) + lifecycle prune ---
    let _ = sys
        .reflect(Some(MemoryCategory::Episodic), Some("pagination"))
        .await
        .expect("reflect");
    let report = sys.prune(MemoryCategory::Episodic).await.expect("prune");
    eprintln!(
        "[live] prune episodic: kept={} demoted={} evicted={}",
        report.kept, report.demoted, report.evicted
    );

    let stats = sys.stats().await.expect("stats");
    eprintln!(
        "[live] stats: total={} by_category={:?}",
        stats.total, stats.by_category
    );
    assert!(stats.total >= 3, "expected stored memories incl avatar");
    assert!(
        stats.by_category.get("avatar").copied().unwrap_or(0) >= 2,
        "avatar slots counted"
    );

    sys.session_end(Some("LAN embedder + reranker + avatar verified end-to-end"))
        .await
        .expect("session_end");
}

#[tokio::test]
#[ignore = "live: requires LATTICE_MCP_LIVE_EMBED_URL + LATTICE_MCP_LIVE_DB_URL"]
async fn remote_lan_roundtrip() {
    let Some([embed_url, db_url]) = endpoints("remote_lan_roundtrip", [EMBED_URL_VAR, DB_URL_VAR])
    else {
        return;
    };
    // Remote store + REAL LAN embedder. No avatar (avatar is embedded-only).
    let mut cfg = make_test_config(std::path::Path::new("/tmp/unused-remote"));
    cfg.database.mode = DatabaseMode::Remote;
    cfg.database.remote_url = Some(db_url.clone());
    cfg.embedding = lan_embedding(&embed_url);
    cfg.reranker = None;
    cfg.avatar = None;
    cfg.validate().expect("config valid");

    let (embedder, reranker) = providers(&cfg);
    let store = RemoteStore::new(db_url.clone(), EMBED_DIMS);
    let mut sys = MemorySystem::new(Box::new(store), embedder, None, reranker, cfg);
    sys.initialize()
        .await
        .expect("init (needs LatticeDB up at LATTICE_MCP_LIVE_DB_URL)");

    let a = sys
        .remember(
            "LIVE-TEST remote roundtrip marker alpha",
            MemoryCategory::Episodic,
            Some(0.5),
            Some("change"),
            None,
            None,
        )
        .await
        .expect("remember a");
    // Link b -> a so recall's per-hit graph traversal returns real edges. This
    // exercises RemoteStore::traverse with a non-empty edge set (regression for
    // the traverse-response shape: server returns {from_id,to_id,relation}).
    let b = sys
        .remember(
            "LIVE-TEST remote roundtrip marker beta",
            MemoryCategory::Episodic,
            Some(0.5),
            Some("change"),
            Some(&[a.id]),
            None,
        )
        .await
        .expect("remember b");

    let hits = sys
        .recall("remote roundtrip marker", &[], 5)
        .await
        .expect("recall (traverses edges)");
    assert!(
        hits.iter().any(|h| h.memory.content.contains("LIVE-TEST")),
        "should recall the markers"
    );

    let stats = sys.stats().await.expect("stats");
    eprintln!(
        "[live-remote] total={} by_category={:?}",
        stats.total, stats.by_category
    );

    // self-clean: remove the markers we wrote.
    sys.forget(MemoryCategory::Episodic, a.id)
        .await
        .expect("forget a");
    sys.forget(MemoryCategory::Episodic, b.id)
        .await
        .expect("forget b");
}

#[tokio::test]
#[ignore = "live: requires all three LATTICE_MCP_LIVE_* URLs and the inference-bible topic memories already stored"]
async fn inference_bible_recall_quality() {
    let Some([embed_url, rerank_url, db_url]) = endpoints(
        "inference_bible_recall_quality",
        [EMBED_URL_VAR, RERANK_URL_VAR, DB_URL_VAR],
    ) else {
        return;
    };
    // Paraphrased queries (no keyword overlap with stored text) -> expected topic.
    // Verifies SEMANTIC recall against the live remote DB + production reranker.
    let cases: &[(&str, &str)] = &[
        ("techniques for shrinking files without losing any information", "Entropy and source coding"),
        ("how should I revise what I believe as fresh evidence comes in", "Bayesian inference"),
        ("sending bits reliably across a link that garbles some of them", "Error-correcting codes"),
        ("deciding between a simple and a complicated explanation without overfitting", "Model comparison and Occam"),
        ("approximating an average when the probability distribution is too hard to integrate directly", "Monte Carlo methods"),
    ];

    let mut cfg = make_test_config(std::path::Path::new("/tmp/unused-bible-recall"));
    cfg.database.mode = DatabaseMode::Remote;
    cfg.database.remote_url = Some(db_url.clone());
    cfg.embedding = lan_embedding(&embed_url);
    cfg.reranker = Some(lan_reranker(&rerank_url)); // mirror production recall path
    cfg.avatar = None;
    cfg.validate().expect("config valid");

    let (embedder, reranker) = providers(&cfg);
    let store = RemoteStore::new(db_url.clone(), EMBED_DIMS);
    let mut sys = MemorySystem::new(Box::new(store), embedder, None, reranker, cfg);
    sys.initialize().await.expect("init");

    let mut top1 = 0;
    let mut top3 = 0;
    for (q, expected) in cases {
        let hits = sys
            .recall(q, &[MemoryCategory::Semantic], 3)
            .await
            .expect("recall");
        let rank = hits
            .iter()
            .position(|h| h.memory.content.contains(expected));
        let top = hits
            .first()
            .map(|h| h.memory.content.chars().take(48).collect::<String>())
            .unwrap_or_default();
        eprintln!(
            "[bible-recall] q={:?}\n   expected={:?} rank={:?} top={:?}",
            q, expected, rank, top
        );
        match rank {
            Some(0) => {
                top1 += 1;
                top3 += 1;
            }
            Some(_) => {
                top3 += 1;
            }
            None => {}
        }
    }
    eprintln!(
        "[bible-recall] top-1 {}/{}, top-3 {}/{}",
        top1,
        cases.len(),
        top3,
        cases.len()
    );
    assert_eq!(
        top3,
        cases.len(),
        "every paraphrased query should recall its topic within top-3"
    );
}

#[tokio::test]
#[ignore = "live: requires LATTICE_MCP_LIVE_EMBED_URL + LATTICE_MCP_LIVE_DB_URL"]
async fn remote_avatar_with_allow_remote() {
    let Some([embed_url, db_url]) = endpoints(
        "remote_avatar_with_allow_remote",
        [EMBED_URL_VAR, DB_URL_VAR],
    ) else {
        return;
    };
    // Avatar persisted to the REMOTE LAN DB via the allow_remote escape hatch.
    let mut cfg = make_test_config(std::path::Path::new("/tmp/unused-remote-avatar"));
    cfg.database.mode = DatabaseMode::Remote;
    cfg.database.remote_url = Some(db_url.clone());
    cfg.embedding = lan_embedding(&embed_url);
    cfg.reranker = None;
    cfg.avatar = Some(AvatarConfig {
        enabled: true,
        allow_freeform: false,
        redact_patterns: vec![],
        auto_derive: false,
        allow_remote: true, // opt out of strict local-first for the trusted LAN DB
    });
    cfg.validate().expect("config valid with allow_remote");

    let (embedder, reranker) = providers(&cfg);
    let store = RemoteStore::new(db_url.clone(), EMBED_DIMS);
    let mut sys = MemorySystem::new(Box::new(store), embedder, None, reranker, cfg);
    sys.initialize().await.expect("init");

    let m = sys
        .avatar_store("livetest", "preferred_language", "Rust", &[])
        .await
        .expect("avatar_store on remote");
    let got = sys
        .avatar_get("livetest")
        .await
        .expect("avatar_get on remote");
    assert!(
        got.iter().any(|s| s.content.contains("Rust")),
        "avatar slot should persist + read back from the remote DB"
    );
    eprintln!(
        "[live-remote-avatar] stored+read avatar slot from remote DB ({} slot(s))",
        got.len()
    );

    // self-clean: remove the avatar slot we wrote.
    sys.forget(MemoryCategory::Avatar, m.id)
        .await
        .expect("forget avatar cleanup");
}

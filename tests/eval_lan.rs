//! Real-embedder retrieval evaluation (NOT run by default).
//!
//! Uses an isolated embedded store (temp dir) + the real LAN embedder so the
//! numbers reflect true semantic recall without touching the shared remote DB.
//! No reranker — this isolates the quality of the *retrieval signals*
//! (vector + graph + recency + importance, and later the lexical signal).
//!
//! Run: cargo test --test eval_lan -- --ignored --nocapture
//! Requires `LATTICE_MCP_LIVE_EMBED_URL` (an OpenAI-compatible embeddings URL
//! serving the model below); unset ⇒ the test prints a skip notice and passes.

mod common;

use lattice_mcp::config::{Config, EmbeddingConfig, MultiSignalConfig};
use lattice_mcp::memory::MemorySystem;
use lattice_mcp::store::embedded::EmbeddedStore;

use common::config::make_test_config;
use common::eval::{print_report, run, seed, EvalReport};
use common::eval_corpus::locomo_style_corpus;
use common::live_env::{endpoints, EMBED_URL_VAR};

const EMBED_MODEL: &str = "nvidia/llama-embed-nemotron-8b";
const EMBED_DIMS: usize = 4096;

fn lan_config(dir: &std::path::Path, embed_url: &str, multi: Option<MultiSignalConfig>) -> Config {
    let mut cfg = make_test_config(dir);
    cfg.embedding = EmbeddingConfig {
        endpoint: embed_url.into(),
        model: EMBED_MODEL.into(),
        dimensions: EMBED_DIMS,
        api_key_env: None,
    };
    cfg.reranker = None; // isolate retrieval signals
    cfg.multi_signal = multi;
    cfg
}

async fn system(dir: &std::path::Path, cfg: Config) -> MemorySystem {
    use lattice_mcp::embedding::openai::OpenAiEmbedder;
    let embedder = Box::new(OpenAiEmbedder::new(&cfg.embedding).expect("embedder"));
    let store = EmbeddedStore::new(dir.to_path_buf(), EMBED_DIMS);
    let mut sys = MemorySystem::new(store_box(store), embedder, None, None, cfg);
    sys.initialize().await.expect("init");
    sys
}

fn store_box(s: EmbeddedStore) -> Box<dyn lattice_mcp::store::MemoryStore> {
    Box::new(s)
}

async fn eval_with(embed_url: &str, multi: Option<MultiSignalConfig>) -> EvalReport {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = lan_config(tmp.path(), embed_url, multi);
    let mut sys = system(tmp.path(), cfg).await;
    let corpus = locomo_style_corpus();
    let ids = seed(&mut sys, &corpus).await;
    run(&mut sys, &corpus, &ids, 5).await
}

#[tokio::test]
#[ignore = "live: requires LATTICE_MCP_LIVE_EMBED_URL"]
async fn eval_lan_baseline() {
    let Some([embed_url]) = endpoints("eval_lan_baseline", [EMBED_URL_VAR]) else {
        return;
    };
    let r = eval_with(&embed_url, None).await;
    print_report("baseline", &r);
    // Sanity floors — the harness must produce non-degenerate numbers.
    assert!(
        r.overall.recall_at_k > 0.4,
        "baseline recall too low: {:.3}",
        r.overall.recall_at_k
    );
    assert!(
        r.overall.mrr > 0.3,
        "baseline mrr too low: {:.3}",
        r.overall.mrr
    );
}

#[tokio::test]
#[ignore = "live: requires LATTICE_MCP_LIVE_EMBED_URL"]
async fn eval_lan_multisignal_compare() {
    use common::eval::QKind;
    let Some([embed_url]) = endpoints("eval_lan_multisignal_compare", [EMBED_URL_VAR]) else {
        return;
    };
    let off = eval_with(&embed_url, None).await;
    let on = eval_with(
        &embed_url,
        Some(MultiSignalConfig {
            enabled: true,
            lexical_weight: 0.35,
        }),
    )
    .await;
    print_report("multisignal-off", &off);
    print_report("multisignal-on", &on);

    let lex_off = off
        .by_kind
        .get(&QKind::Lexical)
        .map(|m| m.mrr)
        .unwrap_or(0.0);
    let lex_on = on
        .by_kind
        .get(&QKind::Lexical)
        .map(|m| m.mrr)
        .unwrap_or(0.0);
    eprintln!(
        "[multisignal] overall mrr {:.3}->{:.3}  ndcg {:.3}->{:.3}  | Lexical mrr {:.3}->{:.3}",
        off.overall.mrr,
        on.overall.mrr,
        off.overall.ndcg_at_k,
        on.overall.ndcg_at_k,
        lex_off,
        lex_on
    );
    // The win: lexical-category ranking improves and overall does not regress.
    assert!(
        lex_on >= lex_off + 1e-4,
        "lexical MRR should improve: {lex_off:.3} -> {lex_on:.3}"
    );
    assert!(
        on.overall.mrr >= off.overall.mrr - 1e-4,
        "overall MRR must not regress"
    );
    assert!(
        on.overall.recall_at_k >= off.overall.recall_at_k - 1e-4,
        "overall recall must not regress"
    );
}

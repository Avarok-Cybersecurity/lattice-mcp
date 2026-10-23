//! A small labelled LoCoMo-style corpus for retrieval evaluation.
//!
//! Generic software-project agent-memory facts spanning the four LoCoMo
//! question categories plus a Lexical category (exact identifiers / codes /
//! versions) where keyword overlap complements semantic embedding.
#![allow(dead_code)]

use super::eval::{EvalCorpus, EvalDoc, EvalQuery, QKind};
use lattice_mcp::memory::types::MemoryCategory::{Episodic, Procedural, Semantic};

pub fn locomo_style_corpus() -> EvalCorpus {
    let docs = vec![
        // --- single-hop / open-domain facts ---
        EvalDoc { label: "jwt", content: "The team chose JWT bearer tokens for stateless authentication.", category: Semantic, links: &[] },
        EvalDoc { label: "bcrypt", content: "User passwords are hashed with bcrypt at cost factor 12.", category: Semantic, links: &[] },
        EvalDoc { label: "pg", content: "The primary datastore is PostgreSQL 15 with read replicas.", category: Semantic, links: &[] },
        EvalDoc { label: "redis", content: "Redis is used for session caching and rate limiting.", category: Semantic, links: &[] },
        EvalDoc { label: "deploy", content: "Services are deployed on Kubernetes using Helm charts.", category: Semantic, links: &[] },
        EvalDoc { label: "lang", content: "The backend is written in Rust and the frontend in TypeScript.", category: Semantic, links: &[] },
        // --- lexical-favoring: exact identifiers / codes / versions ---
        EvalDoc { label: "errcode", content: "Error code E_AUTH_401 is returned when the bearer token has expired.", category: Episodic, links: &[] },
        EvalDoc { label: "fn", content: "The function validate_jwt() verifies the signature and expiry before granting access.", category: Procedural, links: &[] },
        EvalDoc { label: "ver", content: "We pinned the parity-check decoder library to version v2.3.1 to avoid a regression.", category: Episodic, links: &[] },
        EvalDoc { label: "endpoint", content: "The liveness probe hits GET /internal/healthz every ten seconds.", category: Procedural, links: &[] },
        // --- multi-hop (linked) ---
        EvalDoc { label: "outage", content: "The May 2026 outage was caused by connection-pool exhaustion in the auth service.", category: Episodic, links: &["fix"] },
        EvalDoc { label: "fix", content: "Connection-pool exhaustion was resolved by raising max_connections and adding backpressure.", category: Episodic, links: &["outage"] },
        EvalDoc { label: "dep", content: "The auth service depends on the token service for periodic key rotation.", category: Semantic, links: &["jwt"] },
        // --- temporal ---
        EvalDoc { label: "v1", content: "In January we shipped v1 with synchronous request handling.", category: Episodic, links: &[] },
        EvalDoc { label: "v2", content: "In April we migrated to asynchronous handlers, deprecating the v1 synchronous path.", category: Episodic, links: &["v1"] },
        // --- distractors (make precision meaningful) ---
        EvalDoc { label: "logging", content: "Structured JSON logging is shipped to the central ELK stack.", category: Semantic, links: &[] },
        EvalDoc { label: "metrics", content: "Prometheus scrapes metrics every 15 seconds and Grafana visualizes them.", category: Semantic, links: &[] },
        EvalDoc { label: "backup", content: "Database backups run nightly at 02:00 UTC and are retained for 30 days.", category: Semantic, links: &[] },
        EvalDoc { label: "oncall", content: "The on-call rotation is weekly and PagerDuty escalates after five minutes.", category: Semantic, links: &[] },
        EvalDoc { label: "cdn", content: "Static assets are served through a CDN with a 24-hour cache TTL.", category: Semantic, links: &[] },
    ];

    let queries = vec![
        // single-hop
        EvalQuery {
            text: "how are user passwords stored",
            relevant: &["bcrypt"],
            kind: QKind::SingleHop,
        },
        EvalQuery {
            text: "which database does the system use",
            relevant: &["pg"],
            kind: QKind::SingleHop,
        },
        EvalQuery {
            text: "what handles session caching",
            relevant: &["redis"],
            kind: QKind::SingleHop,
        },
        // open-domain
        EvalQuery {
            text: "how do we authenticate users",
            relevant: &["jwt"],
            kind: QKind::OpenDomain,
        },
        EvalQuery {
            text: "what programming languages is the system built in",
            relevant: &["lang"],
            kind: QKind::OpenDomain,
        },
        // lexical (exact tokens; embeddings often blur these)
        EvalQuery {
            text: "E_AUTH_401",
            relevant: &["errcode"],
            kind: QKind::Lexical,
        },
        EvalQuery {
            text: "validate_jwt",
            relevant: &["fn"],
            kind: QKind::Lexical,
        },
        EvalQuery {
            text: "v2.3.1",
            relevant: &["ver"],
            kind: QKind::Lexical,
        },
        EvalQuery {
            text: "/internal/healthz",
            relevant: &["endpoint"],
            kind: QKind::Lexical,
        },
        // multi-hop (graph link should surface the partner)
        EvalQuery {
            text: "how was the connection pool outage fixed",
            relevant: &["outage", "fix"],
            kind: QKind::MultiHop,
        },
        EvalQuery {
            text: "what does the auth service rely on for key rotation",
            relevant: &["dep"],
            kind: QKind::MultiHop,
        },
        // temporal
        EvalQuery {
            text: "what changed between the january and april releases",
            relevant: &["v1", "v2"],
            kind: QKind::Temporal,
        },
    ];

    EvalCorpus { docs, queries }
}

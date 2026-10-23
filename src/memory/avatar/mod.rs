//! Persistent user-avatar subsystem: a slot-keyed, cross-session user model.
//!
//! Pure cores (no I/O — SBIO):
//! - [`schema`]: curated slot vocabulary, validation, canonical rendering.
//! - [`merge`]: reinforce-vs-supersede decision + confidence (Bayesian SSOT).
//! - [`redact`]: pre-write PII redaction (strict local-first).
//!
//! The I/O orchestration (store/search/get/auto-derive) lives in
//! `memory::ops::avatar`, and the MCP surface in `tools::avatar`.

pub mod derive;
pub mod merge;
pub mod redact;
pub mod schema;

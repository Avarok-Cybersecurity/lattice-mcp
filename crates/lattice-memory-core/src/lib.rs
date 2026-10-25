//! The lattice-mcp agent-memory model as a pure library: memory records,
//! adaptive importance (Bayesian, FSRS, SM-2, PageRank, write-time signals),
//! contradiction and write-surprise signals, consolidation and distillation,
//! retrieval scoring (fusion, MMR diversification, lexical overlap),
//! utility-driven lifecycle and the user-avatar slot logic.
//!
//! No I/O. Storage, embedding, reranking and chat stay with the host; time
//! enters only as a `DateTime<Utc>` argument, which hosts read from a
//! [`clock::Clock`]. `no_std` + `alloc`; the `std` feature is for std hosts
//! (see `Cargo.toml`).

#![no_std]
#![forbid(unsafe_code)]
// Pre-existing test style carried over from lattice-mcp: build a Default
// state, then assign the fields under test.
#![cfg_attr(test, allow(clippy::field_reassign_with_default))]

extern crate alloc;
#[cfg(any(feature = "std", test))]
extern crate std;

pub mod avatar;
pub mod chat_types;
pub mod clock;
pub mod config;
pub mod consolidation;
pub mod contradiction;
pub mod error;
pub mod importance;
pub mod importance_state;
pub mod lifecycle;
mod math;
pub mod retrieval;
pub mod scoring;
pub mod slot_contradiction;
pub mod types;
pub mod write_surprise;

/// The `alloc` names every module uses, in place of the std prelude.
pub(crate) mod prelude {
    pub use alloc::format;
    pub use alloc::string::{String, ToString};
    pub use alloc::vec;
    pub use alloc::vec::Vec;
}

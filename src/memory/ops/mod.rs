//! Operation implementations extracted from `MemorySystem`.
//!
//! Each submodule holds the body of one orchestrator operation as a
//! `pub(crate) async fn` taking `&mut MemorySystem`. The methods on
//! `MemorySystem` are thin delegations. This keeps `memory/mod.rs`
//! within the file-size limit without changing behavior.

pub(crate) mod avatar;
pub(crate) mod contradiction;
pub(crate) mod feedback;
pub(crate) mod maintenance;
pub(crate) mod pin;
pub(crate) mod recall;
pub(crate) mod reflect;
pub(crate) mod remember;
pub(crate) mod session;

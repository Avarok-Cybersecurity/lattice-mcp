// Shared test module: each test binary compiles it but uses only the
// variables it needs.
#![allow(dead_code)]

//! Endpoints for the live (`#[ignore]`d) tests come from the environment so no
//! network address is committed. A test whose variables are unset prints why
//! and returns early: an unconfigured run is a skip, never a failure.

pub const EMBED_URL_VAR: &str = "LATTICE_MCP_LIVE_EMBED_URL";
pub const RERANK_URL_VAR: &str = "LATTICE_MCP_LIVE_RERANK_URL";
pub const DB_URL_VAR: &str = "LATTICE_MCP_LIVE_DB_URL";

/// The value of every variable in `vars`, or `None` (after printing the
/// missing names) when any is unset or empty.
pub fn endpoints<const N: usize>(test: &str, vars: [&str; N]) -> Option<[String; N]> {
    let values = vars.map(|v| std::env::var(v).ok().filter(|s| !s.is_empty()));
    let missing: Vec<&str> = vars
        .iter()
        .zip(values.iter())
        .filter(|(_, val)| val.is_none())
        .map(|(name, _)| *name)
        .collect();
    if !missing.is_empty() {
        eprintln!("[{test}] skipped: set {} to run it", missing.join(", "));
        return None;
    }
    Some(values.map(|v| v.expect("checked above")))
}

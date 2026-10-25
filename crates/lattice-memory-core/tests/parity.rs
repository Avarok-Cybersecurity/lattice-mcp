//! Behaviour parity with lattice-mcp before the extraction.
//!
//! `parity/golden.txt` is `parity/fingerprint.rs` compiled against the
//! lattice-mcp modules at f77c609 (the last commit before this crate
//! existed: `use lattice_mcp::memory as m; use lattice_mcp::config as cfgm;`)
//! and run six times with identical output. This test compiles the same
//! fingerprint against this crate. Any differing line is a behaviour change
//! of the moved code.

use lattice_memory_core as m;
use lattice_memory_core::config as cfgm;

mod fp {
    use super::{cfgm, m};
    include!("parity/fingerprint.rs");
}

const GOLDEN: &str = include_str!("parity/golden.txt");

/// Every line where `got` and `want` disagree, plus a length mismatch.
fn differences(got: &[String], want: &[&str]) -> Vec<String> {
    let mut d: Vec<String> = got
        .iter()
        .zip(want.iter())
        .enumerate()
        .filter(|(_, (g, w))| g.as_str() != **w)
        .map(|(i, (g, w))| format!("line {}:\n  want {w}\n  got  {g}", i + 1))
        .collect();
    if got.len() != want.len() {
        d.push(format!("{} lines, golden has {}", got.len(), want.len()));
    }
    d
}

#[test]
fn moved_model_reproduces_pre_extraction_outputs() {
    let want: Vec<&str> = GOLDEN.lines().collect();
    assert!(
        want.len() > 100,
        "golden looks truncated: {} lines",
        want.len()
    );
    let d = differences(&fp::fingerprint(), &want);
    assert!(d.is_empty(), "{} differences:\n{}", d.len(), d.join("\n"));
}

#[test]
fn comparison_catches_a_single_flipped_bit() {
    let want: Vec<&str> = GOLDEN.lines().collect();
    let mut got: Vec<String> = want.iter().map(|s| s.to_string()).collect();
    let i = got
        .iter()
        .position(|l| l.starts_with("fsrs.retention") && l.contains("3e"))
        .expect("a retention line");
    let flipped = got[i].replace("3e", "3f");
    assert_ne!(flipped, got[i], "the mutation must change the line");
    got[i] = flipped;
    assert_eq!(differences(&got, &want).len(), 1);
    got.pop();
    assert_eq!(
        differences(&got, &want).len(),
        2,
        "a missing line is also a difference"
    );
}

#[test]
fn ties_in_shared_terms_are_ordered_and_repeatable() {
    // Before the extraction this order came from HashMap iteration and
    // changed from run to run; it is now count-descending, then
    // alphabetical.
    let mk = |id: u64, c: &str| {
        let mut x = fp_mem(id, c);
        x.importance = 0.5 + id as f32 / 100.0;
        x
    };
    let ms = [
        mk(1, "zeta alpha mango"),
        mk(2, "zeta alpha mango"),
        mk(3, "other words"),
    ];
    let s = m::consolidation::algorithmic_summarize(&ms);
    assert!(s.contains("(key topics: alpha, mango, zeta)"), "{s}");
}

fn fp_mem(id: u64, c: &str) -> m::types::Memory {
    let t = chrono::DateTime::from_timestamp(1_780_000_000, 0).expect("valid timestamp");
    m::types::Memory {
        id,
        category: m::types::MemoryCategory::Episodic,
        content: c.into(),
        created_at: t,
        last_accessed: t,
        access_count: 0,
        importance: 0.5,
        session_id: None,
        metadata: m::types::MemoryMetadata::Episodic {
            event_type: "x".into(),
        },
        importance_state: Default::default(),
    }
}

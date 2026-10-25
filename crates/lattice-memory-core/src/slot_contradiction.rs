//! Keyed contradiction detection: facts that assert a value for the same
//! `(subject, slot)` conflict when their values differ and no settlement
//! decides between them.
//!
//! Cosine similarity barely separates a contradiction from a paraphrase
//! (AUROC 0.59, MemStrata 2606.26511), so this detector compares keys and
//! values instead. A [`Settlement`] records who decided which fact is in
//! force (TOKI 2606.06240: resolution must be auditable). Nothing here
//! extracts keys from text; callers supply them. lattice-mcp does not use it
//! yet (its `remember` carries no slot key); `contradiction` is unchanged.

use alloc::collections::BTreeSet;

use chrono::{DateTime, Utc};

use crate::error::{MemoryCoreError, Result};
use crate::prelude::*;

/// A `(subject, slot)` pair, compared after trimming, lowercasing and
/// collapsing internal whitespace.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SlotKey {
    subject: String,
    slot: String,
}

fn normalize(s: &str) -> String {
    s.split_whitespace()
        .map(|w| w.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

impl SlotKey {
    pub fn new(subject: &str, slot: &str) -> Self {
        Self {
            subject: normalize(subject),
            slot: normalize(slot),
        }
    }
    pub fn subject(&self) -> &str {
        &self.subject
    }
    pub fn slot(&self) -> &str {
        &self.slot
    }
}

/// One fact: memory `id` asserts `value` for `key`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotFact {
    pub id: u64,
    pub key: SlotKey,
    pub value: String,
}

/// Who decided a settlement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Adjudicator {
    User,
    Llm,
    /// A named deterministic rule (e.g. "newest-measurement-wins").
    Rule(String),
}

/// The record of deciding a conflict: `winner` stays in force,
/// `superseded` does not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settlement {
    pub key: SlotKey,
    pub winner: u64,
    pub superseded: u64,
    pub adjudicated_by: Adjudicator,
    pub at: DateTime<Utc>,
}

/// Two facts in force on one key with different values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotConflict {
    pub key: SlotKey,
    /// The two fact ids, smaller first.
    pub ids: (u64, u64),
}

/// Every open conflict among `facts`, ordered by key, then ids.
///
/// A fact named as `superseded` by a settlement on its key is out of force
/// and conflicts with nothing. Equal values (after the same normalization as
/// keys) reinforce each other and never conflict.
pub fn open_conflicts(facts: &[SlotFact], settlements: &[Settlement]) -> Vec<SlotConflict> {
    let retired: BTreeSet<(&SlotKey, u64)> =
        settlements.iter().map(|s| (&s.key, s.superseded)).collect();
    let mut live: Vec<&SlotFact> = facts
        .iter()
        .filter(|f| !retired.contains(&(&f.key, f.id)))
        .collect();
    live.sort_by(|a, b| a.key.cmp(&b.key).then(a.id.cmp(&b.id)));

    let mut out = Vec::new();
    for (i, a) in live.iter().enumerate() {
        for b in live[i + 1..].iter().take_while(|b| b.key == a.key) {
            if a.id != b.id && normalize(&a.value) != normalize(&b.value) {
                out.push(SlotConflict {
                    key: a.key.clone(),
                    ids: (a.id.min(b.id), a.id.max(b.id)),
                });
            }
        }
    }
    out
}

/// Settle `conflict` in favour of `winner`, which must be one of its ids.
pub fn settle(
    conflict: &SlotConflict,
    winner: u64,
    adjudicated_by: Adjudicator,
    at: DateTime<Utc>,
) -> Result<Settlement> {
    let (a, b) = conflict.ids;
    let superseded = match winner {
        w if w == a => b,
        w if w == b => a,
        _ => return Err(MemoryCoreError::SettlementWinnerNotInConflict { winner, a, b }),
    };
    Ok(Settlement {
        key: conflict.key.clone(),
        winner,
        superseded,
        adjudicated_by,
        at,
    })
}

#[cfg(test)]
#[path = "slot_contradiction_tests.rs"]
mod tests;

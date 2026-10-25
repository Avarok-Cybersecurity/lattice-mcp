//! Utility-driven memory lifecycle planning (pure, no I/O — SBIO).
//!
//! Under a per-category budget, the highest-utility memories are kept; the
//! overflow is *demoted* (importance lowered, recoverable) by default, and
//! only hard-*evicted* when explicitly allowed and clearly low value.
//! `content_id` is a content hash, so eviction is "soft" — re-`remember`ing
//! identical content restores it.

use crate::prelude::*;

use chrono::{DateTime, Utc};
use core::cmp::Ordering;
use serde::{Deserialize, Serialize};

/// Outcome of a `prune` pass (plain data — lives here, not in the I/O
/// `ops` module, so it can cross the public `MemorySystem::prune` boundary).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PruneReport {
    pub category: String,
    pub kept: usize,
    pub demoted: usize,
    pub evicted: usize,
}

/// Lifecycle inputs for one memory (utility already blended by the caller).
pub struct LifecycleItem {
    pub id: u64,
    /// Composite utility in [0, 1]: importance signal blended with recency.
    pub utility: f32,
    pub created_at: DateTime<Utc>,
}

/// The plan: which memories to demote vs hard-delete.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct EvictionPlan {
    pub evict: Vec<u64>,
    pub demote: Vec<u64>,
}

/// Keep the top `budget` items by utility (ties → newer first). Overflow is
/// demoted; if `allow_delete`, overflow with `utility < delete_below` is
/// hard-deleted instead. Empty plan when within budget (no-op, behavior
/// preserving).
pub fn eviction_plan(
    items: &[LifecycleItem],
    budget: usize,
    allow_delete: bool,
    delete_below: f32,
) -> EvictionPlan {
    if items.len() <= budget {
        return EvictionPlan::default();
    }

    let mut ranked: Vec<&LifecycleItem> = items.iter().collect();
    ranked.sort_by(|a, b| {
        b.utility
            .partial_cmp(&a.utility)
            .unwrap_or(Ordering::Equal)
            .then(b.created_at.cmp(&a.created_at))
    });

    let mut plan = EvictionPlan::default();
    for item in ranked.into_iter().skip(budget) {
        if allow_delete && item.utility < delete_below {
            plan.evict.push(item.id);
        } else {
            plan.demote.push(item.id);
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: u64, utility: f32, age_days: i64) -> LifecycleItem {
        LifecycleItem {
            id,
            utility,
            created_at: Utc::now() - chrono::Duration::days(age_days),
        }
    }

    #[test]
    fn within_budget_is_noop() {
        let items = vec![item(1, 0.9, 0), item(2, 0.1, 1)];
        assert_eq!(eviction_plan(&items, 5, true, 0.3), EvictionPlan::default());
    }

    #[test]
    fn keeps_highest_utility_demotes_overflow() {
        let items = vec![
            item(1, 0.9, 0),
            item(2, 0.8, 0),
            item(3, 0.2, 0),
            item(4, 0.1, 0),
        ];
        let plan = eviction_plan(&items, 2, false, 0.3);
        // top-2 (ids 1,2) kept; 3,4 demoted; nothing evicted (allow_delete=false)
        assert!(plan.evict.is_empty());
        assert_eq!(plan.demote.len(), 2);
        assert!(plan.demote.contains(&3) && plan.demote.contains(&4));
    }

    #[test]
    fn allow_delete_evicts_only_below_threshold() {
        let items = vec![
            item(1, 0.9, 0),
            item(2, 0.5, 0),  // overflow but >= delete_below ⇒ demote
            item(3, 0.05, 0), // overflow and < delete_below ⇒ evict
        ];
        let plan = eviction_plan(&items, 1, true, 0.3);
        assert_eq!(plan.demote, vec![2]);
        assert_eq!(plan.evict, vec![3]);
    }

    #[test]
    fn ties_break_by_recency_newer_kept() {
        let items = vec![item(1, 0.5, 10), item(2, 0.5, 0)];
        // equal utility, budget 1 → newer (id 2, age 0) kept, older demoted
        let plan = eviction_plan(&items, 1, false, 0.3);
        assert_eq!(plan.demote, vec![1]);
    }
}

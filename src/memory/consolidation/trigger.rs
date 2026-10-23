//! Auto-consolidation trigger predicate (pure, no I/O — SBIO).
//!
//! Fires only at the session boundary (`session_end`), never from inside
//! `remember`, to stay clear of the `reflect → recall → remember` recursion
//! hazard. The signal is the count of stored memories since the last
//! consolidation vs. the configured threshold.
//!
//! (A score-entropy input was considered and dropped: computing it needs an
//! extra scoring pass for marginal value — YAGNI. The predicate stays a
//! simple, testable threshold.)

/// Whether enough new memories have accumulated since the last consolidation
/// to warrant an automatic `reflect`. A `threshold` of 0 disables the
/// trigger (defensive — config validation also rejects 0).
pub fn should_consolidate(writes_since_last: usize, threshold: usize) -> bool {
    threshold > 0 && writes_since_last >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fires_at_or_above_threshold() {
        assert!(should_consolidate(5, 5));
        assert!(should_consolidate(9, 5));
    }

    #[test]
    fn silent_below_threshold() {
        assert!(!should_consolidate(4, 5));
        assert!(!should_consolidate(0, 5));
    }

    #[test]
    fn zero_threshold_never_fires() {
        assert!(!should_consolidate(100, 0));
    }
}

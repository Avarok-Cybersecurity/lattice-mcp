//! Avatar slot merge semantics (pure, no I/O — SBIO).
//!
//! Re-observing a slot either REINFORCES the current value (evidence and
//! confidence rise) or SUPERSEDES it with a new value (confidence-gated
//! last-writer-wins, the prior point id recorded for reversibility). Confidence
//! is NOT computed here from scratch — it reuses the Bayesian posterior mean
//! (SSOT in `importance::bayesian`).

use crate::importance::bayesian;
use crate::importance_state::ImportanceState;

/// What to do with an incoming observation for a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeDecision {
    /// No prior slot — create it.
    Insert,
    /// Same value as the prior — reinforce (bump evidence + confidence).
    Reinforce,
    /// Different value — supersede the prior (record it in `supersedes`).
    Supersede,
}

/// Decide the merge action given any prior value and the new value.
/// Values are compared case-insensitively after trimming.
pub fn decide(prior_value: Option<&str>, new_value: &str) -> MergeDecision {
    match prior_value {
        None => MergeDecision::Insert,
        Some(prev) if eq_value(prev, new_value) => MergeDecision::Reinforce,
        Some(_) => MergeDecision::Supersede,
    }
}

fn eq_value(a: &str, b: &str) -> bool {
    a.trim().eq_ignore_ascii_case(b.trim())
}

/// Belief strength of a slot in [0,1] = posterior mean of its Beta state.
/// Single source of truth: `importance::bayesian::posterior_mean`.
pub fn confidence(state: &ImportanceState) -> f32 {
    bayesian::posterior_mean(state.alpha, state.beta)
}

/// Apply an observation to a slot's adaptive state.
///
/// `Reinforce`/`Insert` count as positive evidence (alpha rises → confidence
/// rises). `Supersede` seeds a FRESH Beta state for the new value (the old
/// belief does not transfer) plus one positive observation, so a newly
/// changed value starts at modest—not inflated—confidence.
pub fn apply(decision: MergeDecision, state: &mut ImportanceState) {
    match decision {
        MergeDecision::Insert | MergeDecision::Reinforce => {
            // Positive evidence with no temporal decay (avatar traits are stable).
            bayesian::update(state, true, 0.0);
        }
        MergeDecision::Supersede => {
            *state = ImportanceState::default();
            bayesian::update(state, true, 0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_matrix() {
        assert_eq!(decide(None, "Rust"), MergeDecision::Insert);
        assert_eq!(decide(Some("Rust"), "rust"), MergeDecision::Reinforce);
        assert_eq!(decide(Some("Rust"), " RUST "), MergeDecision::Reinforce);
        assert_eq!(decide(Some("Rust"), "Go"), MergeDecision::Supersede);
    }

    #[test]
    fn reinforcement_raises_confidence() {
        let mut s = ImportanceState::default();
        let c0 = confidence(&s);
        apply(MergeDecision::Insert, &mut s);
        let c1 = confidence(&s);
        apply(MergeDecision::Reinforce, &mut s);
        let c2 = confidence(&s);
        assert!(c1 > c0 && c2 > c1, "confidence should rise: {c0} {c1} {c2}");
    }

    #[test]
    fn supersede_resets_to_modest_confidence() {
        let mut s = ImportanceState::default();
        for _ in 0..10 {
            apply(MergeDecision::Reinforce, &mut s);
        }
        let high = confidence(&s);
        apply(MergeDecision::Supersede, &mut s);
        let after = confidence(&s);
        assert!(after < high, "supersede must not inherit old confidence");
        // Fresh state + 1 positive observation => alpha=2, beta=1 => 2/3.
        assert!((after - 2.0 / 3.0).abs() < 1e-5, "got {after}");
    }
}

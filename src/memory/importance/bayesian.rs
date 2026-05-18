//! Bayesian importance updating via Beta-Binomial model.
//!
//! Each memory tracks alpha (successes) and beta (failures).
//! The posterior mean `alpha / (alpha + beta)` gives the importance estimate.
//! Starts at Beta(1,1) = uniform prior, converges with evidence.

use super::super::importance_state::ImportanceState;

/// Posterior mean of Beta(alpha, beta) distribution.
///
/// Returns importance estimate in [0.0, 1.0].
/// With default priors (1.0, 1.0), returns 0.5 ("we know nothing").
pub fn posterior_mean(alpha: f32, beta: f32) -> f32 {
    let denom = alpha + beta;
    if denom > 0.0 {
        alpha / denom
    } else {
        0.5
    }
}

/// Posterior mean with exploration bonus (Feel-Good Thompson Sampling).
///
/// Adds `c * sqrt(variance)` so uncertain memories get surfaced.
/// Beta variance = a*b / ((a+b)^2 * (a+b+1)). As evidence accumulates,
/// variance shrinks and this converges to plain posterior_mean.
pub fn posterior_with_exploration(alpha: f32, beta: f32, exploration_c: f32) -> f32 {
    let total = alpha + beta;
    let mean = if total > 0.0 { alpha / total } else { 0.5 };
    let variance = if total > 0.0 {
        (alpha * beta) / (total * total * (total + 1.0))
    } else {
        0.25
    };
    (mean + exploration_c * variance.sqrt()).min(1.0)
}

/// Update Bayesian state after retrieval feedback.
///
/// Applies temporal decay to existing evidence before adding new observation.
/// Decay preserves the Beta(1,1) uniform prior — only accumulated evidence fades.
/// This prevents stale high-alpha memories from dominating forever (KABB, ICML 2025).
pub fn update(state: &mut ImportanceState, relevant: bool, elapsed_days: f64) {
    // Decay existing evidence toward prior, not toward zero.
    // evidence = alpha - 1.0 (strip prior contribution).
    // kappa=0.005/day: half-life ≈ 139 days. Gentle enough for active memories,
    // but ensures dormant memories lose certainty over months.
    const KAPPA: f64 = 0.005;
    let decay = (-KAPPA * elapsed_days).exp() as f32;

    let evidence_alpha = (state.alpha - 1.0).max(0.0);
    let evidence_beta = (state.beta - 1.0).max(0.0);
    state.alpha = 1.0 + decay * evidence_alpha;
    state.beta = 1.0 + decay * evidence_beta;

    if relevant {
        state.alpha += 1.0;
    } else {
        state.beta += 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_prior_gives_half() {
        assert_eq!(posterior_mean(1.0, 1.0), 0.5);
    }

    #[test]
    fn all_successes_approaches_one() {
        assert!(posterior_mean(100.0, 1.0) > 0.98);
    }

    #[test]
    fn all_failures_approaches_zero() {
        assert!(posterior_mean(1.0, 100.0) < 0.02);
    }

    #[test]
    fn zero_denom_returns_half() {
        assert_eq!(posterior_mean(0.0, 0.0), 0.5);
    }

    #[test]
    fn positive_feedback_increases_posterior() {
        let mut state = ImportanceState::default();
        let before = posterior_mean(state.alpha, state.beta);
        update(&mut state, true, 0.0);
        let after = posterior_mean(state.alpha, state.beta);
        assert!(after > before);
    }

    #[test]
    fn negative_feedback_decreases_posterior() {
        let mut state = ImportanceState::default();
        let before = posterior_mean(state.alpha, state.beta);
        update(&mut state, false, 0.0);
        let after = posterior_mean(state.alpha, state.beta);
        assert!(after < before);
    }

    #[test]
    fn many_positives_converge() {
        let mut state = ImportanceState::default();
        for _ in 0..50 {
            update(&mut state, true, 0.0);
        }
        let score = posterior_mean(state.alpha, state.beta);
        assert!(score > 0.95, "50 positives should converge near 1.0, got {score}");
    }

    #[test]
    fn temporal_decay_fades_old_evidence() {
        let mut state = ImportanceState::default();
        // Build up strong positive evidence
        for _ in 0..20 {
            update(&mut state, true, 0.0);
        }
        let strong = posterior_mean(state.alpha, state.beta);
        assert!(strong > 0.9);

        // After 200 days with no feedback, evidence should decay
        update(&mut state, true, 200.0);
        let after_decay = posterior_mean(state.alpha, state.beta);
        // Still positive (just added true), but accumulated evidence is weaker
        assert!(
            state.alpha < 20.0,
            "decay should reduce alpha below 20, got {}",
            state.alpha
        );
    }

    #[test]
    fn temporal_decay_preserves_prior() {
        let mut state = ImportanceState::default();
        // With no evidence, decay should have no effect
        update(&mut state, true, 1000.0);
        // Should just be prior (1,1) + one observation = (2,1)
        assert!((state.alpha - 2.0).abs() < 0.01);
        assert!((state.beta - 1.0).abs() < 0.01);
    }

    #[test]
    fn exploration_bonus_helps_uncertain_memories() {
        // Uncertain: Beta(1,1) — high variance
        let uncertain = posterior_with_exploration(1.0, 1.0, 1.5);
        let certain = posterior_with_exploration(50.0, 10.0, 1.5);

        let uncertain_base = posterior_mean(1.0, 1.0);
        let certain_base = posterior_mean(50.0, 10.0);

        let uncertain_bonus = uncertain - uncertain_base;
        let certain_bonus = certain - certain_base;

        assert!(
            uncertain_bonus > certain_bonus,
            "uncertain should get larger bonus: {uncertain_bonus} vs {certain_bonus}"
        );
    }

    #[test]
    fn exploration_bonus_capped_at_one() {
        let score = posterior_with_exploration(1.0, 1.0, 10.0);
        assert!(score <= 1.0, "exploration bonus should not exceed 1.0, got {score}");
    }
}

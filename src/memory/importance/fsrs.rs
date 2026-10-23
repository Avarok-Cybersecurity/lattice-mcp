//! FSRS (Free Spaced Repetition Scheduler) forgetting curves.
//!
//! Each memory tracks `stability` (days until ~37% retention) and
//! `difficulty` [0.0, 1.0]. Retention at recall-time replaces the
//! generic Ebbinghaus decay when adaptive importance is enabled.
//!
//! Update rules follow a simplified FSRS-4.5 model:
//! - Successful recall → stability grows, difficulty decreases
//! - Failed recall → stability shrinks, difficulty increases

use crate::config::FsrsConfig;
use crate::memory::importance_state::ImportanceState;
use crate::memory::types::MemoryCategory;

/// Retention probability: `e^(-elapsed_days / stability)`.
///
/// Returns [0.0, 1.0] where 1.0 = just reviewed.
/// At `elapsed_days == stability`, retention ≈ 0.37 (1/e).
pub fn retention(stability: f32, elapsed_days: f64) -> f32 {
    let s = stability.max(0.01) as f64;
    (-elapsed_days / s).exp() as f32
}

/// Update FSRS state after recall feedback.
///
/// `quality` uses the 0–5 scale (SM-2 compatible):
/// - 0–2: failure (memory was not recalled correctly)
/// - 3–5: success (memory was recalled, higher = easier)
///
/// `elapsed_days`: time since last review (used to scale stability growth).
pub fn update_state(state: &mut ImportanceState, quality: u8, elapsed_days: f64) {
    let q = quality.min(5);
    let success = q >= 3;

    if success {
        // Stability growth factor: harder memories grow slower.
        // Base growth scaled by (1 - difficulty) and retrieval strength.
        let retrievability = retention(state.fsrs_stability, elapsed_days);
        let growth = 1.0 + (1.0 - state.fsrs_difficulty) * (1.0 - retrievability) * 0.5;
        state.fsrs_stability *= growth;

        // Difficulty decreases on success (easier to retain).
        let d_delta = (q as f32 - 3.0) * 0.02; // q=3→0, q=4→0.02, q=5→0.04
        state.fsrs_difficulty = (state.fsrs_difficulty - 0.05 - d_delta).clamp(0.0, 1.0);
    } else {
        // Stability shrinks on failure.
        let penalty = 0.3 + state.fsrs_difficulty * 0.2; // harder → harsher penalty
        state.fsrs_stability *= penalty;
        state.fsrs_stability = state.fsrs_stability.max(0.1); // floor

        // Difficulty increases on failure.
        let d_delta = (3.0 - q as f32) * 0.03; // q=0→0.09, q=1→0.06, q=2→0.03
        state.fsrs_difficulty = (state.fsrs_difficulty + 0.1 + d_delta).clamp(0.0, 1.0);
    }
}

/// Per-category initial stability from config (PCND: no hidden defaults).
pub fn initial_stability(category: MemoryCategory, config: &FsrsConfig) -> f32 {
    match category {
        MemoryCategory::Episodic => config.episodic_initial_stability,
        MemoryCategory::Semantic => config.semantic_initial_stability,
        MemoryCategory::Procedural => config.procedural_initial_stability,
        // Pinned and Avatar are stable: ~100 years, effectively no decay.
        MemoryCategory::Pinned | MemoryCategory::Avatar => 36500.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retention_at_zero_is_one() {
        let r = retention(1.0, 0.0);
        assert!(
            (r - 1.0).abs() < 0.001,
            "retention at t=0 should be 1.0, got {r}"
        );
    }

    #[test]
    fn retention_at_stability_is_e_inv() {
        let r = retention(7.0, 7.0);
        let expected = (-1.0_f64).exp() as f32; // ~0.368
        assert!(
            (r - expected).abs() < 0.01,
            "retention at t=S should be ~0.37, got {r}"
        );
    }

    #[test]
    fn retention_decreases_with_time() {
        let r1 = retention(5.0, 1.0);
        let r2 = retention(5.0, 10.0);
        assert!(r1 > r2, "retention should decrease: {r1} vs {r2}");
    }

    #[test]
    fn higher_stability_retains_longer() {
        let r_low = retention(1.0, 5.0);
        let r_high = retention(10.0, 5.0);
        assert!(
            r_high > r_low,
            "higher stability should retain better: {r_high} vs {r_low}"
        );
    }

    #[test]
    fn success_increases_stability() {
        let mut state = ImportanceState::default();
        let before = state.fsrs_stability;
        update_state(&mut state, 4, 1.0);
        assert!(
            state.fsrs_stability > before,
            "success should increase stability"
        );
    }

    #[test]
    fn failure_decreases_stability() {
        let mut state = ImportanceState::default();
        state.fsrs_stability = 5.0;
        let before = state.fsrs_stability;
        update_state(&mut state, 1, 1.0);
        assert!(
            state.fsrs_stability < before,
            "failure should decrease stability"
        );
    }

    #[test]
    fn stability_has_floor() {
        let mut state = ImportanceState::default();
        state.fsrs_stability = 0.2;
        state.fsrs_difficulty = 0.9;
        for _ in 0..20 {
            update_state(&mut state, 0, 1.0);
        }
        assert!(
            state.fsrs_stability >= 0.1,
            "stability should not go below floor"
        );
    }

    #[test]
    fn success_decreases_difficulty() {
        let mut state = ImportanceState::default();
        state.fsrs_difficulty = 0.5;
        let before = state.fsrs_difficulty;
        update_state(&mut state, 4, 1.0);
        assert!(
            state.fsrs_difficulty < before,
            "success should decrease difficulty"
        );
    }

    #[test]
    fn failure_increases_difficulty() {
        let mut state = ImportanceState::default();
        let before = state.fsrs_difficulty;
        update_state(&mut state, 1, 1.0);
        assert!(
            state.fsrs_difficulty > before,
            "failure should increase difficulty"
        );
    }

    #[test]
    fn difficulty_clamped() {
        let mut state = ImportanceState::default();
        state.fsrs_difficulty = 0.95;
        update_state(&mut state, 0, 1.0);
        assert!(
            state.fsrs_difficulty <= 1.0,
            "difficulty should clamp at 1.0"
        );

        state.fsrs_difficulty = 0.02;
        update_state(&mut state, 5, 1.0);
        assert!(
            state.fsrs_difficulty >= 0.0,
            "difficulty should clamp at 0.0"
        );
    }

    #[test]
    fn initial_stability_per_category() {
        let cfg = FsrsConfig::default();
        assert_eq!(initial_stability(MemoryCategory::Episodic, &cfg), 1.0);
        assert_eq!(initial_stability(MemoryCategory::Semantic, &cfg), 7.0);
        assert_eq!(initial_stability(MemoryCategory::Procedural, &cfg), 3.0);
        assert_eq!(initial_stability(MemoryCategory::Pinned, &cfg), 36500.0);
    }

    #[test]
    fn harder_memory_grows_slower() {
        let mut easy = ImportanceState::default();
        easy.fsrs_difficulty = 0.1;
        easy.fsrs_stability = 5.0;

        let mut hard = ImportanceState::default();
        hard.fsrs_difficulty = 0.9;
        hard.fsrs_stability = 5.0;

        update_state(&mut easy, 4, 3.0);
        update_state(&mut hard, 4, 3.0);

        assert!(
            easy.fsrs_stability > hard.fsrs_stability,
            "easy memory should grow faster: {} vs {}",
            easy.fsrs_stability,
            hard.fsrs_stability
        );
    }
}

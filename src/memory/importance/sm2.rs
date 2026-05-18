//! SM-2 spaced repetition algorithm.
//!
//! Tracks easiness factor, interval, and repetition count per memory.
//! Priority is inversely proportional to interval×easiness — memories
//! that are "due" for review score higher.
//!
//! Based on Piotr Wozniak's SM-2 algorithm (1987), adapted for
//! importance scoring rather than flashcard scheduling.

use crate::memory::importance_state::ImportanceState;

/// Review priority: higher when memory is "due" (short interval).
///
/// Formula: `1 / (1 + interval × easiness × 0.1)`
/// Returns [0.0, 1.0] where 1.0 = maximally due for review.
pub fn priority(interval: f32, easiness: f32) -> f32 {
    1.0 / (1.0 + interval * easiness * 0.1)
}

/// Mastery signal: higher when memory is well-learned (long interval, high easiness).
///
/// Complement of priority: `1.0 - priority`.
/// Returns [0.0, 1.0] where 1.0 = maximally mastered.
/// Used in composite_importance so that well-recalled memories score higher,
/// aligning with Bayesian and FSRS feedback direction.
pub fn mastery(interval: f32, easiness: f32) -> f32 {
    1.0 - priority(interval, easiness)
}

/// Update SM-2 state after recall feedback.
///
/// `quality` uses the standard 0–5 scale:
/// - 0: complete blackout
/// - 1: incorrect, but remembered upon seeing answer
/// - 2: incorrect, but answer seemed easy to recall
/// - 3: correct with serious difficulty
/// - 4: correct after hesitation
/// - 5: perfect recall
///
/// Quality ≥ 3 counts as successful recall.
pub fn update(state: &mut ImportanceState, quality: u8) {
    let q = quality.min(5) as f32;

    // Update easiness factor (EF).
    // SM-2 formula: EF' = EF + (0.1 - (5 - q) × (0.08 + (5 - q) × 0.02))
    let ef = state.sm2_easiness + (0.1 - (5.0 - q) * (0.08 + (5.0 - q) * 0.02));
    state.sm2_easiness = ef.max(1.3); // SM-2 floor

    if quality >= 3 {
        state.sm2_repetitions += 1;
        state.sm2_interval = match state.sm2_repetitions {
            1 => 1.0,
            2 => 6.0,
            _ => state.sm2_interval * state.sm2_easiness,
        };
    } else {
        // Failed recall → reset to beginning
        state.sm2_repetitions = 0;
        state.sm2_interval = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_at_default_is_moderate() {
        let state = ImportanceState::default();
        let p = priority(state.sm2_interval, state.sm2_easiness);
        // 1/(1 + 1.0 * 2.5 * 0.1) = 1/1.25 = 0.8
        assert!((p - 0.8).abs() < 0.01, "default priority should be ~0.8, got {p}");
    }

    #[test]
    fn priority_decreases_as_interval_grows() {
        let p1 = priority(1.0, 2.5);
        let p2 = priority(30.0, 2.5);
        assert!(p1 > p2, "shorter interval = higher priority: {p1} vs {p2}");
    }

    #[test]
    fn priority_at_zero_interval_is_one() {
        let p = priority(0.0, 2.5);
        assert!((p - 1.0).abs() < 0.001);
    }

    #[test]
    fn success_extends_interval() {
        let mut state = ImportanceState::default();
        update(&mut state, 4); // first success
        assert_eq!(state.sm2_repetitions, 1);
        assert_eq!(state.sm2_interval, 1.0);

        update(&mut state, 4); // second success
        assert_eq!(state.sm2_repetitions, 2);
        assert_eq!(state.sm2_interval, 6.0);

        update(&mut state, 4); // third → interval grows by EF
        assert_eq!(state.sm2_repetitions, 3);
        assert!(state.sm2_interval > 6.0, "interval should grow: {}", state.sm2_interval);
    }

    #[test]
    fn failure_resets_repetitions() {
        let mut state = ImportanceState::default();
        // Build up some reps
        for _ in 0..5 {
            update(&mut state, 4);
        }
        assert!(state.sm2_repetitions >= 5);
        let prev_interval = state.sm2_interval;
        assert!(prev_interval > 6.0);

        // Fail
        update(&mut state, 2);
        assert_eq!(state.sm2_repetitions, 0);
        assert_eq!(state.sm2_interval, 1.0);
    }

    #[test]
    fn easiness_clamps_at_floor() {
        let mut state = ImportanceState::default();
        // Repeated failures should drive EF down but not below 1.3
        for _ in 0..20 {
            update(&mut state, 0);
        }
        assert!(
            state.sm2_easiness >= 1.3,
            "easiness should not drop below 1.3, got {}",
            state.sm2_easiness
        );
    }

    #[test]
    fn perfect_recall_increases_easiness() {
        let mut state = ImportanceState::default();
        let before = state.sm2_easiness;
        update(&mut state, 5);
        assert!(
            state.sm2_easiness > before,
            "perfect recall should increase easiness: {} vs {}",
            state.sm2_easiness,
            before
        );
    }

    #[test]
    fn quality_three_barely_passes() {
        let mut state = ImportanceState::default();
        update(&mut state, 3);
        assert_eq!(state.sm2_repetitions, 1, "quality=3 should count as success");
        // EF should decrease slightly for quality=3
        // SM-2: 0.1 - (5-3)(0.08 + (5-3)*0.02) = 0.1 - 2*(0.08+0.04) = 0.1 - 0.24 = -0.14
        assert!(
            state.sm2_easiness < 2.5,
            "quality=3 should decrease easiness: {}",
            state.sm2_easiness
        );
    }

    #[test]
    fn interval_uses_current_easiness() {
        let mut state = ImportanceState::default();
        // Get to 3+ repetitions
        for _ in 0..3 {
            update(&mut state, 5);
        }
        let interval_before = state.sm2_interval;
        update(&mut state, 5);
        // SM-2 updates EF first, then computes interval = old_interval × new_EF.
        // For quality=5: EF' = EF + 0.1, so interval grows by the updated EF.
        let new_ef = state.sm2_easiness;
        let expected = interval_before * new_ef;
        assert!(
            (state.sm2_interval - expected).abs() < 0.01,
            "interval should equal old_interval × new_EF: got {} expected {}",
            state.sm2_interval,
            expected
        );
    }
}

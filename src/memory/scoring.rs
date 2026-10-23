//! Pure scoring functions — no I/O, no side effects.
//!
//! Implements Ebbinghaus forgetting curves, importance scoring,
//! and recency calculations for the HybridRAG retrieval pipeline.

use chrono::{DateTime, Utc};

/// Temporal decay using Ebbinghaus forgetting curve.
///
/// Returns a factor in [0.0, 1.0] where 1.0 = just created.
/// Formula: `decay = 0.5^(elapsed_days / half_life_days)`
pub fn temporal_decay(created_at: DateTime<Utc>, now: DateTime<Utc>, half_life_days: f64) -> f32 {
    let elapsed_secs = (now - created_at).num_seconds().max(0) as f64;
    let elapsed_days = elapsed_secs / 86400.0;
    0.5_f64.powf(elapsed_days / half_life_days) as f32
}

/// Recency boost based on last access time.
///
/// Returns [0.0, 1.0] where 1.0 = accessed just now.
pub fn recency_score(last_accessed: DateTime<Utc>, now: DateTime<Utc>, half_life_days: f64) -> f32 {
    temporal_decay(last_accessed, now, half_life_days)
}

/// Adjust importance based on access frequency.
///
/// More accesses raise the importance floor via a logarithmic curve.
/// Returns [0.0, 1.0].
pub fn access_adjusted_importance(base_importance: f32, access_count: u64) -> f32 {
    let access_boost = 1.0 - (1.0 / (1.0 + access_count as f32 * 0.1));
    (base_importance + access_boost * 0.3).min(1.0)
}

/// Auto-classify importance based on content heuristics.
///
/// Returns [0.0, 1.0]. Used when the caller doesn't supply an explicit importance.
pub fn auto_importance(content: &str, event_type: &str) -> f32 {
    let lower = content.to_lowercase();
    let mut score = 0.5_f32;

    // Error fixes and bug reports are high importance
    if event_type == "error" || lower.contains("fix") || lower.contains("bug") {
        score += 0.2;
    }
    // Architecture and design decisions are high importance
    if event_type == "decision" || lower.contains("architecture") || lower.contains("design") {
        score += 0.2;
    }
    // User preferences are high importance
    if lower.contains("prefer") || lower.contains("always") || lower.contains("never") {
        score += 0.15;
    }
    // Security-related content is critical
    if lower.contains("security") || lower.contains("vulnerability") || lower.contains("auth") {
        score += 0.15;
    }

    score.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn decay_at_zero_is_one() {
        let now = Utc::now();
        let score = temporal_decay(now, now, 30.0);
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn decay_at_half_life_is_half() {
        let now = Utc::now();
        let created = now - Duration::days(30);
        let score = temporal_decay(created, now, 30.0);
        assert!((score - 0.5).abs() < 0.01);
    }

    #[test]
    fn decay_at_double_half_life_is_quarter() {
        let now = Utc::now();
        let created = now - Duration::days(60);
        let score = temporal_decay(created, now, 30.0);
        assert!((score - 0.25).abs() < 0.01);
    }

    #[test]
    fn access_adjusted_increases_with_count() {
        let base = 0.5;
        let low = access_adjusted_importance(base, 1);
        let high = access_adjusted_importance(base, 100);
        assert!(high > low);
    }

    #[test]
    fn access_adjusted_capped_at_one() {
        let score = access_adjusted_importance(0.9, 1000);
        assert!(score <= 1.0);
    }

    #[test]
    fn auto_importance_baseline() {
        let score = auto_importance("some regular content", "conversation");
        assert!((score - 0.5).abs() < 0.001);
    }

    #[test]
    fn auto_importance_error_boost() {
        let score = auto_importance("Fixed the login bug", "error");
        assert!(score >= 0.7);
    }

    #[test]
    fn auto_importance_decision_boost() {
        let score = auto_importance("Architecture decision: use microservices", "decision");
        assert!(score >= 0.7);
    }

    #[test]
    fn auto_importance_capped_at_one() {
        let score = auto_importance(
            "Fix security vulnerability in auth architecture, always prefer",
            "error",
        );
        assert!(score <= 1.0);
    }
}

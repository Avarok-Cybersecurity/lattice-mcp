//! Write-time contradiction detection (pure, no I/O — SBIO).
//!
//! Experimental. The algorithmic path is deliberately *conservative*: a
//! generic low token-overlap is noisy, so it keys on a **negation-polarity
//! flip** between two semantically near-identical statements (e.g. "auth
//! uses bcrypt" vs "auth must not use bcrypt"). The LLM path
//! (`ChatProvider::entailment`) is the high-precision route and the only one
//! allowed to drive `SUPERSEDES`.

/// Negation / contradiction cue words.
const NEG_CUES: &[&str] = &[
    "not",
    "no",
    "never",
    "cannot",
    "can't",
    "won't",
    "don't",
    "doesn't",
    "isn't",
    "aren't",
    "without",
    "fails",
    "failed",
    "broken",
    "wrong",
    "incorrect",
    "deprecated",
    "removed",
    "avoid",
    "stop",
    "instead",
];

fn tokens(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

fn has_negation(toks: &[String]) -> bool {
    toks.iter().any(|t| NEG_CUES.contains(&t.as_str()))
}

fn jaccard(a: &[String], b: &[String]) -> f32 {
    use std::collections::HashSet;
    let sa: HashSet<&String> = a.iter().collect();
    let sb: HashSet<&String> = b.iter().collect();
    let inter = sa.intersection(&sb).count() as f32;
    let union = sa.union(&sb).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Disagreement score in `[0, 1]`. A negation-polarity flip contributes the
/// dominant 0.6; lexical divergence adds up to 0.4. Polarity flip alone
/// (0.6) clears the 0.5 contradiction bar; generic divergence alone (≤0.4)
/// does not — keeping the algorithmic path conservative.
pub fn lexical_disagreement(a: &str, b: &str) -> f32 {
    let (ta, tb) = (tokens(a), tokens(b));
    let polarity_flip = has_negation(&ta) != has_negation(&tb);
    let divergence = 1.0 - jaccard(&ta, &tb);
    let score = if polarity_flip { 0.6 } else { 0.0 } + 0.4 * divergence;
    score.clamp(0.0, 1.0)
}

/// Conservative algorithmic contradiction test: the candidate is
/// semantically near a neighbor (`embed_sim ≥ min_similarity`) yet
/// disagrees with it (`disagreement ≥ 0.5`, i.e. a polarity flip).
pub fn is_likely_contradiction(embed_sim: f32, disagreement: f32, min_similarity: f32) -> bool {
    embed_sim >= min_similarity && disagreement >= 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polarity_flip_is_high_disagreement() {
        let d = lexical_disagreement(
            "auth must use bcrypt for password hashing",
            "auth must not use bcrypt for password hashing",
        );
        assert!(d >= 0.5, "negation flip should clear the bar, got {d}");
    }

    #[test]
    fn same_polarity_paraphrase_is_low_disagreement() {
        let d = lexical_disagreement(
            "use bcrypt for password hashing",
            "use bcrypt to hash passwords",
        );
        assert!(d < 0.5, "no polarity flip ⇒ below bar, got {d}");
    }

    #[test]
    fn unrelated_text_without_flip_stays_conservative() {
        // Very divergent but no negation flip → divergence-only ≤ 0.4 < 0.5.
        let d = lexical_disagreement("alpha beta gamma", "delta epsilon zeta");
        assert!(
            d < 0.5,
            "generic divergence must not trip contradiction, got {d}"
        );
    }

    #[test]
    fn contradiction_needs_both_similarity_and_disagreement() {
        assert!(is_likely_contradiction(0.95, 0.6, 0.9));
        assert!(!is_likely_contradiction(0.80, 0.6, 0.9)); // not similar enough
        assert!(!is_likely_contradiction(0.95, 0.3, 0.9)); // not disagreeing enough
    }
}

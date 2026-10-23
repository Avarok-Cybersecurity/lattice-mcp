//! Query transformation before embedding (pure, no I/O — SBIO).
//!
//! Conservative by design: the algorithmic transform strips stopwords and
//! de-duplicates terms so the embedded query emphasizes content words.
//! HyDE / hypothetical-document expansion is deliberately **deferred** — it
//! doubles hot-path latency for marginal gain on short coding-memory
//! entries (see plan). The LLM path is `ChatProvider::transform_query`;
//! this is the deterministic fallback (PCND).

use crate::memory::consolidation::stop_words;

/// Stopword-filtered, de-duplicated, order-preserving rewrite. Never returns
/// an empty string — if filtering removes everything (e.g. an all-stopword
/// query) the original is kept so the embedder always gets real input.
pub fn keyword_expand(query: &str) -> String {
    let stop = stop_words();
    let mut seen = std::collections::HashSet::new();
    let kept: Vec<String> = query
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| w.len() > 2 && !stop.contains(w.as_str()))
        .filter(|w| seen.insert(w.clone()))
        .collect();
    if kept.is_empty() {
        query.to_string()
    } else {
        kept.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_stopwords_and_dedups() {
        let out = keyword_expand("the JWT token and the JWT auth flow");
        // stopwords (the, and) dropped; duplicate "jwt" collapsed; order kept.
        assert_eq!(out, "jwt token auth flow");
    }

    #[test]
    fn all_stopwords_falls_back_to_original() {
        let q = "is it the a an";
        assert_eq!(keyword_expand(q), q, "never embed an empty query");
    }

    #[test]
    fn punctuation_is_normalized() {
        assert_eq!(keyword_expand("bcrypt, hashing!"), "bcrypt hashing");
    }
}

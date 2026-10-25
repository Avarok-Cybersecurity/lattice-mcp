//! Lexical (keyword-overlap) retrieval signal (pure, no I/O — SBIO).
//!
//! Complements the semantic embedding: exact/rare tokens (error codes,
//! identifiers, versions like `E_AUTH_401`, `validate_jwt`, `v2.3.1`) are
//! often blurred by a dense embedder but matched exactly by keyword overlap
//! (Mem0-style multi-signal retrieval). Reuses the shared stopword set.
//!
//! Score = fraction of the query's content terms that appear in the document
//! ("query coverage"), in [0,1]. This rewards documents that literally
//! contain the rare query tokens without diluting on document length.

use crate::prelude::*;

use alloc::collections::BTreeSet;

use crate::consolidation::stop_words;

/// Tokenize into lowercased content terms: split on whitespace, trim edge
/// punctuation (keeping interior `_ . /` so `e_auth_401`, `v2.3.1`,
/// `internal/healthz` survive), drop stopwords and tokens of length ≤ 2.
fn content_terms(text: &str) -> BTreeSet<String> {
    let stop = stop_words();
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|t| t.len() > 2 && !stop.contains(t.as_str()))
        .collect()
}

/// Query-coverage lexical overlap in [0,1]: |query_terms ∩ doc_terms| /
/// |query_terms|. Returns 0.0 when the query has no content terms.
pub fn lexical_overlap(query: &str, content: &str) -> f32 {
    let q = content_terms(query);
    if q.is_empty() {
        return 0.0;
    }
    let d = content_terms(content);
    let hits = q.iter().filter(|t| d.contains(*t)).count();
    hits as f32 / q.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_identifier_matches_strongly() {
        let doc = "Error code E_AUTH_401 is returned when the bearer token expired.";
        assert!((lexical_overlap("E_AUTH_401", doc) - 1.0).abs() < 1e-6);
        // an unrelated doc scores 0 for that rare token
        assert_eq!(
            lexical_overlap("E_AUTH_401", "Redis is used for caching."),
            0.0
        );
    }

    #[test]
    fn interior_punctuation_tokens_survive() {
        assert!(lexical_overlap("v2.3.1", "pinned the decoder to version v2.3.1 today") > 0.99);
        assert!(
            lexical_overlap("/internal/healthz", "probe hits GET /internal/healthz now") > 0.99
        );
        assert!(
            lexical_overlap("validate_jwt", "the function validate_jwt() checks expiry") > 0.99
        );
    }

    #[test]
    fn partial_coverage_is_fractional() {
        // 2 of 3 content terms (user, passwords; "stored" absent) -> ~0.667
        let s = lexical_overlap(
            "how are user passwords stored",
            "user passwords are hashed with bcrypt",
        );
        assert!((s - 2.0 / 3.0).abs() < 1e-6, "got {s}");
    }

    #[test]
    fn empty_or_stopword_query_is_zero() {
        assert_eq!(lexical_overlap("", "anything"), 0.0);
        assert_eq!(lexical_overlap("is it the a", "anything here"), 0.0);
    }
}

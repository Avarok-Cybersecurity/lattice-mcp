//! High-precision explicit-statement slot extraction (pure, no I/O — SBIO).
//!
//! Used by the opt-in session-end auto-derive capture layer WITHOUT an LLM.
//! Deliberately high-precision / low-recall: it only fires on unambiguous
//! "`<slot> is <value>`" / "`<slot>: <value>`" templates generated from the
//! curated vocabulary (SSOT: triggers derive from `schema::CURATED_SLOTS`),
//! so auto-derive sharpens the avatar rather than polluting it with guesses.
//! An LLM path (via `ChatProvider`) can add higher-recall extraction.

use crate::prelude::*;

use super::schema::{value_kind_for, CURATED_SLOTS};
use crate::types::ValueKind;

const MAX_VALUE_LEN: usize = 120;

/// Extract `(slot_key, value)` pairs from one text using explicit templates.
pub fn extract_explicit(text: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for (slot, _) in CURATED_SLOTS {
        let spaced = slot.replace('_', " ");
        for trigger in [format!("{spaced} is "), format!("{spaced}: ")] {
            if let Some(idx) = find_ci_ascii(text, &trigger) {
                let start = idx + trigger.len();
                let value = capture_value(&text[start..], value_kind_for(slot));
                if !value.is_empty() && !out.iter().any(|(k, _)| k == slot) {
                    out.push((slot.to_string(), value));
                }
            }
        }
    }
    out
}

/// Capture the value after a trigger, scoped by the slot's value kind:
/// Scalar/Enum take the first token (precise); List/FreeText take the phrase
/// up to the first sentence terminator. Trimmed and length-capped.
fn capture_value(rest: &str, kind: ValueKind) -> String {
    let phrase_end = rest
        .find(['.', ',', ';', '\n', '!', '?'])
        .unwrap_or(rest.len());
    let phrase = &rest[..phrase_end];
    let slice = match kind {
        ValueKind::Scalar | ValueKind::Enum => phrase.split_whitespace().next().unwrap_or(""),
        ValueKind::List | ValueKind::FreeText => phrase.trim(),
    };
    slice
        .chars()
        .take(MAX_VALUE_LEN)
        .collect::<String>()
        .trim()
        .to_string()
}

/// ASCII case-insensitive substring search returning a byte index valid in
/// `haystack` (ASCII lowercasing is 1:1 byte-preserving, so offsets are safe).
fn find_ci_ascii(haystack: &str, needle: &str) -> Option<usize> {
    let (hb, nb) = (haystack.as_bytes(), needle.as_bytes());
    if nb.is_empty() || nb.len() > hb.len() {
        return None;
    }
    (0..=hb.len() - nb.len()).find(|&i| {
        hb[i..i + nb.len()]
            .iter()
            .zip(nb)
            .all(|(a, b)| a.eq_ignore_ascii_case(b))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_explicit_slot_value_preserving_case() {
        let got = extract_explicit("My preferred language is Rust for systems work.");
        assert_eq!(
            got,
            vec![("preferred_language".to_string(), "Rust".to_string())]
        );
    }

    #[test]
    fn colon_form_and_multiple_slots() {
        let got = extract_explicit("timezone: America/New_York; editor is Helix");
        assert!(got.contains(&("timezone".into(), "America/New_York".into())));
        assert!(got.contains(&("editor".into(), "Helix".into())));
    }

    #[test]
    fn no_explicit_statement_yields_nothing() {
        // Vague preference language must NOT fire (precision over recall).
        assert!(extract_explicit("I kind of like fast tools sometimes").is_empty());
    }

    #[test]
    fn case_insensitive_trigger() {
        let got = extract_explicit("PREFERRED LANGUAGE IS Go");
        assert_eq!(
            got,
            vec![("preferred_language".to_string(), "Go".to_string())]
        );
    }

    #[test]
    fn value_is_length_capped() {
        let long = "x".repeat(300);
        let got = extract_explicit(&format!("coding style is {long}"));
        assert_eq!(got.len(), 1);
        assert!(got[0].1.len() <= MAX_VALUE_LEN);
    }
}

//! Pre-write PII redaction for avatar values (pure, no I/O — SBIO; CWE-312).
//!
//! Strict local-first: a persistent model of a real person must not silently
//! absorb secrets. Built-in heuristics (email, long digit runs) always apply;
//! `extra_patterns` adds user-declared literal substrings. No regex crate
//! dependency — deliberately conservative, simple, and auditable.

/// Result of a redaction pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedactOutcome {
    pub text: String,
    pub redacted: bool,
    pub kinds: Vec<&'static str>,
}

/// Redact likely-PII from `value`. `extra_patterns` are case-insensitive
/// literal substrings the user wants masked (e.g. a real name, an org).
pub fn redact(value: &str, extra_patterns: &[String]) -> RedactOutcome {
    let mut kinds: Vec<&'static str> = Vec::new();
    let mut text = value.to_string();

    // 1) User-declared literal patterns (case-insensitive).
    for pat in extra_patterns.iter().filter(|p| !p.trim().is_empty()) {
        if let Some(replaced) = replace_ci(&text, pat, "<redacted:custom>") {
            text = replaced;
            push_unique(&mut kinds, "custom");
        }
    }

    // 2) Built-in token heuristics (email, long digit runs).
    let mut out_tokens = Vec::new();
    for tok in text.split_whitespace() {
        if looks_like_email(tok) {
            out_tokens.push("<redacted:email>".to_string());
            push_unique(&mut kinds, "email");
        } else if max_digit_run(tok) >= 7 {
            out_tokens.push("<redacted:number>".to_string());
            push_unique(&mut kinds, "number");
        } else {
            out_tokens.push(tok.to_string());
        }
    }
    text = out_tokens.join(" ");

    RedactOutcome {
        redacted: !kinds.is_empty(),
        text,
        kinds,
    }
}

fn push_unique(v: &mut Vec<&'static str>, k: &'static str) {
    if !v.contains(&k) {
        v.push(k);
    }
}

/// Case-insensitive literal replace-all. Returns Some(new) if any match.
fn replace_ci(haystack: &str, needle: &str, with: &str) -> Option<String> {
    let (hl, nl) = (haystack.to_lowercase(), needle.to_lowercase());
    if nl.is_empty() || !hl.contains(&nl) {
        return None;
    }
    let mut out = String::with_capacity(haystack.len());
    let mut i = 0;
    while i < haystack.len() {
        if hl[i..].starts_with(&nl) {
            out.push_str(with);
            i += nl.len();
        } else {
            // Advance one char (respect UTF-8 boundaries).
            let ch = haystack[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    Some(out)
}

fn looks_like_email(tok: &str) -> bool {
    let t = tok.trim_matches(|c: char| !c.is_alphanumeric() && c != '@' && c != '.' && c != '_');
    match t.find('@') {
        Some(at) => at > 0 && t[at + 1..].contains('.') && !t[at + 1..].ends_with('.'),
        None => false,
    }
}

fn max_digit_run(tok: &str) -> usize {
    let (mut max, mut cur) = (0usize, 0usize);
    for c in tok.chars() {
        if c.is_ascii_digit() {
            cur += 1;
            max = max.max(cur);
        } else {
            cur = 0;
        }
    }
    max
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_pii_is_untouched() {
        let r = redact("prefers concise explanations", &[]);
        assert!(!r.redacted);
        assert_eq!(r.text, "prefers concise explanations");
    }

    #[test]
    fn email_is_redacted() {
        let r = redact("contact me at jane.doe@example.com please", &[]);
        assert!(r.redacted);
        assert!(r.text.contains("<redacted:email>"));
        assert!(!r.text.contains("example.com"));
        assert!(r.kinds.contains(&"email"));
    }

    #[test]
    fn long_digit_runs_redacted_short_ones_kept() {
        let r = redact("call 5551234567 not 42", &[]);
        assert!(r.text.contains("<redacted:number>"));
        assert!(r.text.contains("42"), "short numbers preserved");
    }

    #[test]
    fn custom_literal_pattern_case_insensitive() {
        let r = redact("works at AcmeCorp on auth", &["acmecorp".into()]);
        assert!(r.redacted);
        assert!(r.text.contains("<redacted:custom>"));
        assert!(!r.text.to_lowercase().contains("acmecorp"));
    }

    #[test]
    fn empty_patterns_ignored() {
        let r = redact("hello world", &["".into(), "   ".into()]);
        assert!(!r.redacted);
        assert_eq!(r.text, "hello world");
    }
}

//! Avatar slot vocabulary + validation + canonical rendering (pure, SBIO).
//!
//! A curated slot vocabulary gives the user model precision and dedupe; the
//! `allow_freeform` flag (opt-in) permits arbitrary slots when richer, less
//! structured signals are wanted.

use crate::prelude::*;

use crate::types::ValueKind;

/// Curated slots: (slot_key, value_kind). The canonical user-model vocabulary.
pub const CURATED_SLOTS: &[(&str, ValueKind)] = &[
    ("preferred_language", ValueKind::Scalar),
    ("coding_style", ValueKind::FreeText),
    ("communication_tone", ValueKind::Enum),
    ("tooling", ValueKind::List),
    ("timezone", ValueKind::Scalar),
    ("expertise", ValueKind::List),
    ("domain_focus", ValueKind::List),
    ("workflow_preference", ValueKind::FreeText),
    ("naming_convention", ValueKind::FreeText),
    ("testing_preference", ValueKind::FreeText),
    ("editor", ValueKind::Scalar),
    ("os_platform", ValueKind::Scalar),
    ("review_preference", ValueKind::FreeText),
    ("goal", ValueKind::FreeText),
];

/// Normalize a raw slot key: trim, lowercase, and map spaces/hyphens to `_`.
pub fn normalize_slot(raw: &str) -> String {
    raw.trim()
        .to_lowercase()
        .chars()
        .map(|c| if c == ' ' || c == '-' { '_' } else { c })
        .collect()
}

/// The `ValueKind` for a (normalized) slot — curated kind, else `FreeText`.
pub fn value_kind_for(slot_key: &str) -> ValueKind {
    CURATED_SLOTS
        .iter()
        .find(|(k, _)| *k == slot_key)
        .map(|(_, vk)| *vk)
        .unwrap_or(ValueKind::FreeText)
}

/// Validate a normalized slot key against the policy.
/// Curated slots always pass; unknown slots pass only when `allow_freeform`.
pub fn validate_slot(slot_key: &str, allow_freeform: bool) -> Result<(), String> {
    if slot_key.is_empty() {
        return Err("slot_key must not be empty".into());
    }
    let curated = CURATED_SLOTS.iter().any(|(k, _)| *k == slot_key);
    if curated || allow_freeform {
        Ok(())
    } else {
        Err(format!(
            "unknown slot_key '{slot_key}'; use a curated slot or enable avatar.allow_freeform"
        ))
    }
}

/// Canonical searchable rendering stored as the memory `content`.
/// Keeps the slot first so vector search keys on the trait name + value.
pub fn canonical_render(slot_key: &str, value: &str) -> String {
    format!("{slot_key}: {value}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_maps_spaces_and_case() {
        assert_eq!(
            normalize_slot("  Preferred Language "),
            "preferred_language"
        );
        assert_eq!(normalize_slot("communication-tone"), "communication_tone");
    }

    #[test]
    fn curated_slot_kinds() {
        assert_eq!(value_kind_for("tooling"), ValueKind::List);
        assert_eq!(value_kind_for("timezone"), ValueKind::Scalar);
        // Unknown -> FreeText.
        assert_eq!(value_kind_for("favorite_color"), ValueKind::FreeText);
    }

    #[test]
    fn validation_respects_freeform_flag() {
        assert!(validate_slot("preferred_language", false).is_ok());
        assert!(validate_slot("favorite_color", false).is_err());
        assert!(validate_slot("favorite_color", true).is_ok());
        assert!(validate_slot("", true).is_err());
    }

    #[test]
    fn render_is_slot_first() {
        assert_eq!(
            canonical_render("preferred_language", "Rust"),
            "preferred_language: Rust"
        );
    }
}

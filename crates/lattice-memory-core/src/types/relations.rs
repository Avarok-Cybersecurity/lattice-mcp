//! Relation constants — each maps to a u16 ID in LatticeDB's relation registry.

use super::MemoryCategory;

// Episodic relations
pub const PRECEDED_BY: &str = "PRECEDED_BY";
pub const CAUSED: &str = "CAUSED";
pub const RESOLVED_BY: &str = "RESOLVED_BY";

// Semantic relations
pub const RELATES_TO: &str = "RELATES_TO";
pub const DEPENDS_ON: &str = "DEPENDS_ON";
pub const CONTRADICTS: &str = "CONTRADICTS";
pub const SUPERSEDES: &str = "SUPERSEDES";

// Procedural relations
pub const APPLIES_TO: &str = "APPLIES_TO";
pub const ALTERNATIVE_TO: &str = "ALTERNATIVE_TO";
pub const REQUIRES: &str = "REQUIRES";

// Cross-category (stored in the target collection)
pub const LEARNED_FROM: &str = "LEARNED_FROM";
pub const EVIDENCED_BY: &str = "EVIDENCED_BY";
pub const USES_KNOWLEDGE: &str = "USES_KNOWLEDGE";

/// Single source of truth for the (category, relation, id) registry.
/// Both the embedded and remote stores consume this for collection
/// registration and reverse name lookup — no per-store duplication.
pub const REGISTRY: &[(MemoryCategory, &str, u16)] = &[
    (MemoryCategory::Episodic, PRECEDED_BY, 0),
    (MemoryCategory::Episodic, CAUSED, 1),
    (MemoryCategory::Episodic, RESOLVED_BY, 2),
    (MemoryCategory::Episodic, LEARNED_FROM, 3),
    (MemoryCategory::Episodic, EVIDENCED_BY, 4),
    (MemoryCategory::Semantic, RELATES_TO, 0),
    (MemoryCategory::Semantic, DEPENDS_ON, 1),
    (MemoryCategory::Semantic, CONTRADICTS, 2),
    (MemoryCategory::Semantic, SUPERSEDES, 3),
    (MemoryCategory::Semantic, USES_KNOWLEDGE, 4),
    (MemoryCategory::Procedural, APPLIES_TO, 0),
    (MemoryCategory::Procedural, ALTERNATIVE_TO, 1),
    (MemoryCategory::Procedural, REQUIRES, 2),
    // Avatar: a slot revision chains via SUPERSEDES; related slots link
    // via RELATES_TO.
    (MemoryCategory::Avatar, SUPERSEDES, 0),
    (MemoryCategory::Avatar, RELATES_TO, 1),
];

/// Reverse-map a (category, relation_id) to its relation name.
pub fn relation_name(category: MemoryCategory, id: u16) -> &'static str {
    REGISTRY
        .iter()
        .find(|(c, _, rid)| *c == category && *rid == id)
        .map(|(_, name, _)| *name)
        .unwrap_or("UNKNOWN")
}

/// Iterate the (relation_name, id) pairs registered for a category.
pub fn relations_for(category: MemoryCategory) -> impl Iterator<Item = (&'static str, u16)> {
    REGISTRY
        .iter()
        .filter(move |(c, _, _)| *c == category)
        .map(|(_, name, id)| (*name, *id))
}

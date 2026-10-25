//! Structured 4-field distillation of a memory cluster (pure, no I/O — SBIO).
//!
//! Replaces the single-string summary with `{core, context, theme,
//! provenance}` so consolidated knowledge stays searchable AND traceable to
//! its sources. The LLM path is `ChatProvider::distill` (Step 2A); this is
//! the deterministic fallback (PCND — explicit, no hidden behavior).

use crate::prelude::*;

use crate::chat_types::DistilledKnowledge;
use crate::types::Memory;

/// Deterministic distillation: reuse the algorithmic summary as `core`,
/// the dominant shared term as `theme`, a category/count line as `context`,
/// and the source IDs as `provenance` (the provenance-loss fix).
pub fn algorithmic_distill(memories: &[Memory]) -> DistilledKnowledge {
    let core = super::algorithmic_summarize(memories);
    let theme = super::top_shared_term(memories).unwrap_or_else(|| "general".to_string());

    let mut cats: Vec<&str> = memories
        .iter()
        .map(|m| m.category.collection_name())
        .collect();
    cats.sort_unstable();
    cats.dedup();
    let context = format!(
        "Distilled from {} memories ({})",
        memories.len(),
        cats.join(", ")
    );

    DistilledKnowledge {
        core,
        context,
        theme,
        provenance: memories.iter().map(|m| m.id).collect(),
    }
}

/// Render a `DistilledKnowledge` into the stored semantic content string.
/// Keeps `core` first (what existing recall keys on) and appends the
/// disambiguating context + theme so the text stays self-describing.
pub fn render(dk: &DistilledKnowledge) -> String {
    format!("{} [theme: {}] ({})", dk.core, dk.theme, dk.context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Memory, MemoryCategory};

    fn mem(content: &str) -> Memory {
        Memory {
            id: Memory::content_id(content, MemoryCategory::Episodic),
            category: MemoryCategory::Episodic,
            content: content.to_string(),
            created_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
            access_count: 0,
            importance: 0.7,
            session_id: None,
            metadata: crate::types::MemoryMetadata::Episodic {
                event_type: "test".into(),
            },
            importance_state: Default::default(),
        }
    }

    #[test]
    fn distill_captures_provenance_and_theme() {
        let cluster = vec![
            mem("authentication bug fixed in login module"),
            mem("authentication token validation fixed"),
        ];
        let dk = algorithmic_distill(&cluster);
        assert_eq!(dk.provenance.len(), 2, "every source id recorded");
        assert_eq!(dk.provenance[0], cluster[0].id);
        assert_eq!(dk.theme, "authentication", "dominant shared term");
        assert!(!dk.core.is_empty());
        assert!(dk.context.contains("2 memories"));
    }

    #[test]
    fn render_keeps_core_first() {
        let dk = DistilledKnowledge {
            core: "Core insight".into(),
            context: "Distilled from 3 memories (episodic)".into(),
            theme: "auth".into(),
            provenance: vec![1, 2, 3],
        };
        let s = render(&dk);
        assert!(s.starts_with("Core insight"));
        assert!(s.contains("[theme: auth]"));
    }

    #[test]
    fn empty_cluster_is_safe() {
        let dk = algorithmic_distill(&[]);
        assert!(dk.provenance.is_empty());
        assert_eq!(dk.theme, "general");
    }
}

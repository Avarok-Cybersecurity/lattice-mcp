//! Avatar I/O orchestration: store / search / get / session-end auto-derive.
//!
//! Pure logic lives in `memory::avatar::{schema,merge,redact,derive}`; this
//! module is the SBIO seam that talks to the store/embedder. Writes reuse
//! `remember::commit_upsert` (single upsert; the slot-keyed point id means a
//! re-observation UPSERTs in place and the avatar sharpens).

use tracing::info;

use crate::config::AvatarConfig;
use crate::error::{McpMemoryError, Result};
use crate::memory::avatar::{derive, merge, redact, schema};
use crate::memory::importance_state::ImportanceState;
use crate::memory::types::*;
use crate::memory::MemorySystem;

/// Default avatar id when the caller does not specify one.
pub const DEFAULT_AVATAR: &str = "default";

fn enabled_cfg(sys: &MemorySystem) -> Result<AvatarConfig> {
    match sys.config.avatar.as_ref() {
        Some(a) if a.enabled => Ok(a.clone()),
        _ => Err(McpMemoryError::Config {
            message: "avatar feature is disabled; set [avatar] enabled = true".into(),
        }),
    }
}

fn avatar_value(m: &Memory) -> Option<String> {
    match &m.metadata {
        MemoryMetadata::Avatar { value, .. } => Some(value.clone()),
        _ => None,
    }
}
fn avatar_evidence(m: &Memory) -> u32 {
    match &m.metadata {
        MemoryMetadata::Avatar { evidence_count, .. } => *evidence_count,
        _ => 0,
    }
}
fn avatar_provenance(m: &Memory) -> Vec<u64> {
    match &m.metadata {
        MemoryMetadata::Avatar { provenance, .. } => provenance.clone(),
        _ => Vec::new(),
    }
}
fn belongs(m: &Memory, avatar_id: &str) -> bool {
    matches!(&m.metadata, MemoryMetadata::Avatar { avatar_id: a, .. } if a == avatar_id)
}

/// Store (or reinforce/supersede) an avatar slot. Returns the resulting slot.
#[tracing::instrument(skip_all, fields(avatar_id))]
pub(crate) async fn store(
    sys: &mut MemorySystem,
    avatar_id: &str,
    slot_raw: &str,
    value_raw: &str,
    provenance: &[u64],
) -> Result<Memory> {
    let cfg = enabled_cfg(sys)?;
    let slot_key = schema::normalize_slot(slot_raw);
    schema::validate_slot(&slot_key, cfg.allow_freeform)
        .map_err(|message| McpMemoryError::Config { message })?;

    // Strict local-first: redact PII before anything is persisted.
    let red = redact::redact(value_raw, &cfg.redact_patterns);
    if red.redacted {
        info!(slot = %slot_key, kinds = ?red.kinds, "Avatar value redacted pre-write");
    }
    let value = red.text;

    let id = Memory::avatar_point_id(avatar_id, &slot_key);
    let prior = sys.store.get_memory(MemoryCategory::Avatar, id).await?;
    let prior_value = prior.as_ref().and_then(avatar_value);
    let decision = merge::decide(prior_value.as_deref(), &value);

    // Reinforce continues the prior Beta state; insert/supersede start fresh.
    let mut state = match (&prior, decision) {
        (Some(p), merge::MergeDecision::Reinforce) => p.importance_state.clone(),
        _ => ImportanceState::default(),
    };
    merge::apply(decision, &mut state);
    let confidence = merge::confidence(&state);

    let evidence_count = match decision {
        merge::MergeDecision::Reinforce => prior.as_ref().map(avatar_evidence).unwrap_or(0) + 1,
        _ => 1,
    };
    let prior_value_field = match decision {
        merge::MergeDecision::Supersede => prior_value,
        _ => None,
    };

    let mut merged_prov = prior.as_ref().map(avatar_provenance).unwrap_or_default();
    for p in provenance {
        if !merged_prov.contains(p) {
            merged_prov.push(*p);
        }
    }

    let now = sys.clock.now();
    let memory = Memory {
        id,
        category: MemoryCategory::Avatar,
        content: schema::canonical_render(&slot_key, &value),
        created_at: prior.as_ref().map(|p| p.created_at).unwrap_or(now),
        last_accessed: now,
        access_count: prior.as_ref().map(|p| p.access_count + 1).unwrap_or(0),
        importance: confidence, // surface belief strength for ranking
        session_id: sys.current_session_id.clone(),
        metadata: MemoryMetadata::Avatar {
            avatar_id: avatar_id.to_string(),
            slot_key: slot_key.clone(),
            value,
            value_kind: schema::value_kind_for(&slot_key),
            confidence,
            provenance: merged_prov,
            evidence_count,
            prior_value: prior_value_field,
        },
        importance_state: state,
    };

    let vector = sys.embedder.embed(&memory.content).await?;
    super::remember::commit_upsert(sys, &memory, vector).await?;
    // NOTE: deliberately does NOT bump `writes_since_consolidation` (the
    // episodic auto-consolidation counter), matching `pin`. Avatar is a
    // special namespace with its own `session_end` derivation; feeding it
    // into the episodic `reflect` trigger would consolidate the wrong store.
    info!(id, slot = %slot_key, ?decision, confidence, "Avatar slot stored");
    Ok(memory)
}

/// Semantic search within one avatar. Reuses the recall pipeline scoped to
/// the Avatar collection, then filters to `avatar_id` (the collection is
/// shared across avatars). Over-fetches to offset the post-filter.
///
/// A user model is small (tens of slots), so we fetch `max(limit*4, 64)`
/// candidates: ≥64 comfortably covers a single avatar's slots even when the
/// collection holds many avatars, so the `avatar_id` filter rarely starves
/// the result. For exhaustive (non-ranked) retrieval, use `get`.
#[tracing::instrument(skip_all, fields(avatar_id))]
pub(crate) async fn search(
    sys: &mut MemorySystem,
    avatar_id: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<ScoredMemory>> {
    enabled_cfg(sys)?;
    let overfetch = (limit * 4).max(64);
    let scored =
        super::recall::run(sys, query, &[MemoryCategory::Avatar], overfetch, None, None).await?;
    Ok(scored
        .into_iter()
        .filter(|s| belongs(&s.memory, avatar_id))
        .take(limit)
        .collect())
}

/// All slots for one avatar (no query), most-confident first.
#[tracing::instrument(skip_all, fields(avatar_id))]
pub(crate) async fn get(sys: &MemorySystem, avatar_id: &str) -> Result<Vec<Memory>> {
    enabled_cfg(sys)?;
    let mut slots: Vec<Memory> = sys
        .store
        .list_all(MemoryCategory::Avatar)
        .await?
        .into_iter()
        .filter(|m| belongs(m, avatar_id))
        .collect();
    slots.sort_by(|a, b| {
        b.importance
            .partial_cmp(&a.importance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(slots)
}

/// Session-end auto-derive: extract explicit slot statements from this
/// session's episodic memories into the avatar (high-precision, no LLM).
/// No-op unless `avatar.auto_derive`. Returns the number of slots written.
#[tracing::instrument(skip_all, fields(avatar_id))]
pub(crate) async fn auto_derive(sys: &mut MemorySystem, avatar_id: &str) -> Result<usize> {
    match sys.config.avatar.as_ref() {
        Some(a) if a.enabled && a.auto_derive => {}
        _ => return Ok(0),
    }

    let sid = sys.current_session_id.clone();
    let episodics = sys.store.list_all(MemoryCategory::Episodic).await?;
    let mut pending: Vec<(String, String, u64)> = Vec::new();
    for m in &episodics {
        if sid.is_some() && m.session_id != sid {
            continue;
        }
        for (slot, value) in derive::extract_explicit(&m.content) {
            pending.push((slot, value, m.id));
        }
    }

    let mut count = 0;
    for (slot, value, src) in pending {
        if store(sys, avatar_id, &slot, &value, &[src]).await.is_ok() {
            count += 1;
        }
    }
    if count > 0 {
        info!(count, "Avatar slots auto-derived at session end");
    }
    Ok(count)
}

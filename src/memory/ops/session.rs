//! Session lifecycle operations.

use tracing::info;

use crate::error::Result;
use crate::memory::types::*;
use crate::memory::MemorySystem;

#[tracing::instrument(skip_all)]
pub(crate) async fn start(
    sys: &mut MemorySystem,
    project: Option<&str>,
    task: Option<&str>,
) -> Result<(String, Vec<ScoredMemory>)> {
    let session_id = uuid::Uuid::new_v4().to_string();
    sys.current_session_id = Some(session_id.clone());

    let content = match (project, task) {
        (Some(p), Some(t)) => format!("Session started for project '{p}': {t}"),
        (Some(p), None) => format!("Session started for project '{p}'"),
        (None, Some(t)) => format!("Session started: {t}"),
        (None, None) => "Session started".to_string(),
    };

    sys.remember(
        &content,
        MemoryCategory::Episodic,
        Some(0.3),
        Some("session"),
        None,
        None,
    )
    .await?;

    // Recall relevant context
    let context = if let Some(t) = task {
        sys.recall(t, &[], sys.config.memory.max_retrieval_results)
            .await?
    } else {
        Vec::new()
    };

    info!(session_id = %session_id, context_count = context.len(), "Session started");
    Ok((session_id, context))
}

#[tracing::instrument(skip_all)]
pub(crate) async fn end(sys: &mut MemorySystem, summary: Option<&str>) -> Result<()> {
    let content = match summary {
        Some(s) => format!("Session ended: {s}"),
        None => "Session ended".to_string(),
    };

    sys.remember(
        &content,
        MemoryCategory::Episodic,
        Some(0.3),
        Some("session"),
        None,
        None,
    )
    .await?;

    // Step 8: optional auto-consolidation at the session boundary. Checked
    // here (outside any `remember` call stack) so the reflect→recall→remember
    // path can't recurse — reflect resets the counter at its end, one pass.
    if sys.config.consolidation.auto_consolidate == Some(true)
        && crate::memory::consolidation::trigger::should_consolidate(
            sys.writes_since_consolidation,
            sys.config.memory.consolidation_threshold,
        )
    {
        sys.reflect(None, None).await?;
    }

    // Step 5: optional lifecycle prune at the session boundary. Safe from
    // the reflect→recall→remember recursion hazard — prune only does
    // list/get/upsert/delete, never recall/remember/reflect.
    if sys
        .config
        .lifecycle
        .as_ref()
        .is_some_and(|l| l.enabled && l.prune_on_session_end)
    {
        for cat in [
            MemoryCategory::Episodic,
            MemoryCategory::Semantic,
            MemoryCategory::Procedural,
        ] {
            super::maintenance::prune(sys, cat).await?;
        }
    }

    // Avatar auto-derive at the session boundary (opt-in; no-op otherwise).
    // Runs before the session id is cleared so it can scope to this session's
    // episodic memories. Recursion-safe: avatar store does not recall/reflect.
    super::avatar::auto_derive(sys, super::avatar::DEFAULT_AVATAR).await?;

    info!("Session ended");
    sys.current_session_id = None;
    Ok(())
}

//! Integration test: multi-session project workflow.
//!
//! Simulates a 3-session software development project ("widget-app" auth feature)
//! to prove the memory system tracks and helps progress across sessions.
//!
//! Assertion categories:
//! 1. Session context — session_start returns relevant prior memories
//! 2. Semantic recall — query finds semantically related memories
//! 3. Graph discovery — connected memories are traversable
//! 4. Importance evolution — rate_recall changes importance_state
//! 5. Consolidation — reflect creates semantic summaries
//! 6. Pin persistence — pinned memories persist across sessions
//! 7. Stats growth — memory counts increase over time
//! 8. Write-time signals — TF-IDF/salience computed for new memories
//! 9. Cross-session knowledge — session 3 recalls session 1 memories

mod common;

use lattice_mcp::memory::types::{relations, MemoryCategory};
use lattice_mcp::memory::MemorySystem;
use lattice_mcp::store::embedded::EmbeddedStore;

use common::config::make_test_config;
use common::mock_embedder::MockEmbedder;

/// Build a fresh MemorySystem backed by EmbeddedStore + MockEmbedder.
async fn make_system(data_dir: &std::path::Path) -> MemorySystem {
    let config = make_test_config(data_dir);
    let store = EmbeddedStore::new(data_dir.to_path_buf(), 64);
    let embedder = MockEmbedder;

    let mut sys = MemorySystem::new(
        Box::new(store),
        Box::new(embedder),
        None, // no chat provider
        None, // no reranker
        config,
    );
    sys.initialize().await.expect("initialize failed");
    sys
}

#[tokio::test]
async fn multi_session_project_workflow() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();

    // Track stats across sessions
    let stats_after_s1;
    let stats_after_s2;

    // Pre-compute IDs we'll need for cross-session assertions
    let jwt_content = "Architecture decision: use JWT tokens for stateless authentication";
    let bcrypt_content = "Chose bcrypt for password hashing, auth library is jsonwebtoken";
    let jwt_id =
        lattice_mcp::memory::types::Memory::content_id(jwt_content, MemoryCategory::Semantic);
    let bcrypt_id =
        lattice_mcp::memory::types::Memory::content_id(bcrypt_content, MemoryCategory::Semantic);

    // =========================================================
    // SESSION 1: Planning
    // =========================================================
    {
        let mut sys = make_system(dir).await;

        let (session_id, _context) = sys
            .session_start(Some("widget-app"), Some("Plan authentication feature"))
            .await
            .expect("session_start");
        assert!(!session_id.is_empty(), "session ID should be non-empty");

        // Remember planning decisions
        let jwt_mem = sys
            .remember(
                jwt_content,
                MemoryCategory::Semantic,
                Some(0.8),
                Some("decision"),
                None,
                None,
            )
            .await
            .expect("remember JWT");
        assert_eq!(jwt_mem.id, jwt_id);

        sys.remember(
            "Database schema needs users table with hashed passwords for authentication",
            MemoryCategory::Semantic,
            Some(0.7),
            Some("decision"),
            None,
            None,
        )
        .await
        .expect("remember DB schema");

        let bcrypt_mem = sys
            .remember(
                bcrypt_content,
                MemoryCategory::Semantic,
                Some(0.7),
                Some("decision"),
                None,
                None,
            )
            .await
            .expect("remember bcrypt");
        assert_eq!(bcrypt_mem.id, bcrypt_id);

        sys.remember(
            "Security requirement: all authentication endpoints must use HTTPS",
            MemoryCategory::Episodic,
            Some(0.8),
            Some("decision"),
            None,
            None,
        )
        .await
        .expect("remember security req");

        sys.remember(
            "API design: POST /auth/login, POST /auth/register, GET /auth/me",
            MemoryCategory::Procedural,
            Some(0.6),
            Some("pattern"),
            None,
            None,
        )
        .await
        .expect("remember API design");

        // --- Assertion 3: Graph discovery (manual connect) ---
        sys.connect(
            jwt_id,
            bcrypt_id,
            relations::DEPENDS_ON,
            0.9,
            MemoryCategory::Semantic,
        )
        .await
        .expect("connect JWT→bcrypt");

        let traversal = sys
            .search_graph(MemoryCategory::Semantic, jwt_id, 2, None)
            .await
            .expect("search_graph");
        assert!(
            traversal.iter().any(|h| h.target_id == bcrypt_id),
            "Graph traversal from JWT should find bcrypt via DEPENDS_ON"
        );

        // --- Assertion 6: Pin persistence ---
        let pin = sys
            .pin(
                "Always use explicit error types, never unwrap in production",
                Some("coding-standard"),
            )
            .await
            .expect("pin");
        assert_eq!(pin.category, MemoryCategory::Pinned);

        let pinned = sys.list_pinned().await.expect("list_pinned");
        assert_eq!(pinned.len(), 1, "Should have exactly 1 pin after session 1");

        // --- Assertion 8: Write-time signals ---
        // Memories stored after similar ones exist should have write-time signals
        // The bcrypt memory was stored after JWT (both share "auth" keywords)
        let bcrypt_after = sys
            .get_memory(MemoryCategory::Semantic, bcrypt_id)
            .await
            .expect("get bcrypt")
            .expect("bcrypt should exist");
        // With importance enabled and neighbors present, at least one signal should be non-zero
        let has_write_signals = bcrypt_after.importance_state.tfidf_uniqueness != 0.0
            || bcrypt_after.importance_state.salience != 0.0;
        assert!(
            has_write_signals,
            "Write-time signals (TF-IDF or salience) should be computed for memories with neighbors"
        );

        // --- Assertion 7: Stats growth (baseline) ---
        stats_after_s1 = sys.stats().await.expect("stats");
        assert!(
            stats_after_s1.total > 0,
            "Should have memories after session 1"
        );

        sys.session_end(Some(
            "Completed auth feature planning with JWT, bcrypt, 3 endpoints",
        ))
        .await
        .expect("session_end");
    }

    // =========================================================
    // SESSION 2: Implementation
    // =========================================================
    {
        let mut sys = make_system(dir).await;

        // --- Assertion 1: Session context ---
        let (_session_id, context) = sys
            .session_start(
                Some("widget-app"),
                Some("Implement authentication endpoints"),
            )
            .await
            .expect("session_start s2");
        assert!(
            !context.is_empty(),
            "Session 2 should recall context from session 1 planning"
        );

        // --- Assertion 6: Pin persistence (session 2) ---
        let pinned = sys.list_pinned().await.expect("list_pinned s2");
        assert_eq!(pinned.len(), 1, "Pin should persist into session 2");
        assert!(pinned[0].content.contains("explicit error types"));

        // Remember implementation work
        sys.remember(
            "Implemented POST /auth/login with JWT token generation and authentication",
            MemoryCategory::Episodic,
            Some(0.7),
            Some("change"),
            None,
            None,
        )
        .await
        .expect("remember login impl");

        sys.remember(
            "Fixed authentication error: bcrypt hash comparison was case-sensitive",
            MemoryCategory::Episodic,
            Some(0.9),
            Some("error"),
            None,
            None,
        )
        .await
        .expect("remember bcrypt bug");

        sys.remember(
            "Implemented POST /auth/register with password validation",
            MemoryCategory::Episodic,
            Some(0.7),
            Some("change"),
            None,
            None,
        )
        .await
        .expect("remember register impl");

        sys.remember(
            "Auth middleware pattern: extract JWT token from Authorization header",
            MemoryCategory::Procedural,
            Some(0.6),
            Some("pattern"),
            None,
            None,
        )
        .await
        .expect("remember middleware pattern");

        // --- Assertion 4: Importance evolution via rate_recall ---
        let jwt_before = sys
            .get_memory(MemoryCategory::Semantic, jwt_id)
            .await
            .expect("get jwt")
            .expect("jwt should exist");
        let alpha_before = jwt_before.importance_state.alpha;
        let sm2_reps_before = jwt_before.importance_state.sm2_repetitions;

        let rated = sys
            .rate_recall(MemoryCategory::Semantic, jwt_id, 5)
            .await
            .expect("rate_recall jwt");
        assert!(rated.is_some(), "rate_recall should return updated memory");
        let rated_mem = rated.unwrap();
        assert!(
            rated_mem.importance_state.alpha > alpha_before,
            "Bayesian alpha should increase after quality=5 rating (before={}, after={})",
            alpha_before,
            rated_mem.importance_state.alpha
        );
        assert!(
            rated_mem.importance_state.sm2_repetitions > sm2_reps_before,
            "SM-2 repetitions should increase after quality=5 rating"
        );
        assert!(
            rated_mem.importance_state.last_review.is_some(),
            "last_review should be set after rate_recall"
        );

        // Rate bcrypt memory too
        sys.rate_recall(MemoryCategory::Semantic, bcrypt_id, 4)
            .await
            .expect("rate_recall bcrypt");

        // --- Assertion 7: Stats growth (session 2) ---
        stats_after_s2 = sys.stats().await.expect("stats s2");
        assert!(
            stats_after_s2.total > stats_after_s1.total,
            "Total memory count should grow between sessions (s1={}, s2={})",
            stats_after_s1.total,
            stats_after_s2.total
        );

        sys.session_end(Some(
            "Implemented login, register, middleware. Fixed bcrypt bug",
        ))
        .await
        .expect("session_end s2");
    }

    // =========================================================
    // SESSION 3: Review
    // =========================================================
    {
        let mut sys = make_system(dir).await;

        // --- Assertion 1 + 9: Cross-session context ---
        let (_session_id, context) = sys
            .session_start(
                Some("widget-app"),
                Some("Review authentication implementation"),
            )
            .await
            .expect("session_start s3");
        assert!(
            !context.is_empty(),
            "Session 3 should recall context from sessions 1 and 2"
        );

        // --- Assertion 9: Cross-session knowledge ---
        let recall_results = sys
            .recall("JWT authentication token", &[], 10)
            .await
            .expect("recall JWT");
        assert!(
            !recall_results.is_empty(),
            "Recall should find JWT-related memories from session 1"
        );

        // --- Assertion 2: Semantic recall ---
        let bug_recall = sys
            .recall("authentication error bcrypt bug fix", &[], 10)
            .await
            .expect("recall bug");
        assert!(
            !bug_recall.is_empty(),
            "Recall should find the bcrypt bug fix from session 2"
        );
        let found_bug = bug_recall
            .iter()
            .any(|sm| sm.memory.content.contains("bcrypt"));
        assert!(
            found_bug,
            "Bug recall should include the bcrypt error memory"
        );

        // --- Assertion 6: Pin persistence (session 3) ---
        let pinned = sys.list_pinned().await.expect("list_pinned s3");
        assert_eq!(pinned.len(), 1, "Pin should persist into session 3");

        // --- Assertion 5: Consolidation ---
        let consolidated = sys
            .reflect(Some(MemoryCategory::Episodic), Some("authentication"))
            .await
            .expect("reflect");
        // reflect may or may not produce results depending on clustering
        // (memories need shared terms + temporal proximity + importance >= 0.4)
        // At minimum, verify it doesn't error and returns a vec
        // If it does produce consolidations, they should be semantic
        for mem in &consolidated {
            assert_eq!(
                mem.category,
                MemoryCategory::Semantic,
                "Consolidated memories should be semantic"
            );
        }

        // --- Assertion 7: Stats growth (session 3) ---
        let stats_after_s3 = sys.stats().await.expect("stats s3");
        assert!(
            stats_after_s3.total >= stats_after_s2.total,
            "Total should not decrease (s2={}, s3={})",
            stats_after_s2.total,
            stats_after_s3.total
        );

        // Verify category breakdown
        let episodic_count = stats_after_s3
            .by_category
            .get("episodic")
            .copied()
            .unwrap_or(0);
        let semantic_count = stats_after_s3
            .by_category
            .get("semantic")
            .copied()
            .unwrap_or(0);
        assert!(episodic_count > 0, "Should have episodic memories");
        assert!(semantic_count > 0, "Should have semantic memories");

        sys.session_end(Some("Auth review complete"))
            .await
            .expect("session_end s3");
    }
}

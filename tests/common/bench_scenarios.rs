//! Canonical benchmark trajectory (split from bench_fixtures for size).
#![allow(dead_code)]

use super::bench_fixtures::Step;
use lattice_mcp::memory::types::MemoryCategory;

/// Canonical software-project trajectory over the `MockEmbedder` vocabulary
/// (auth / jwt / bcrypt / database / api / security).
///
/// Deliberately seeded with keyword-colliding **distractors** and
/// **near-duplicates**, queried at a tight `k=3`, so the baseline does NOT
/// saturate recall@k/MRR — leaving headroom for ranking/policy features to
/// demonstrate measurable improvement (the OFF-vs-ON gate is meaningless if
/// OFF already scores 1.0).
pub fn auth_project_scenario() -> Vec<Step> {
    use MemoryCategory::*;
    vec![
        // --- Target memories (the "relevant" answers) ---
        Step::Write {
            content: "Architecture decision: use JWT tokens for stateless authentication",
            category: Semantic,
            importance: 0.8,
            event_type: "decision",
        },
        Step::Write {
            content: "Chose bcrypt for password hashing in the auth library",
            category: Semantic,
            importance: 0.7,
            event_type: "decision",
        },
        Step::Write {
            content: "Database schema needs a users table with hashed passwords",
            category: Semantic,
            importance: 0.7,
            event_type: "decision",
        },
        Step::Write {
            content: "Fixed authentication error: bcrypt hash comparison was case sensitive",
            category: Episodic,
            importance: 0.9,
            event_type: "error",
        },
        Step::Write {
            content: "Security requirement: all authentication endpoints must use HTTPS",
            category: Episodic,
            importance: 0.8,
            event_type: "decision",
        },
        // --- Near-duplicate (novelty/dedup features should notice) ---
        Step::Write {
            content: "Fixed authentication bug where bcrypt hash comparison was case sensitive",
            category: Episodic,
            importance: 0.6,
            event_type: "error",
        },
        // --- Keyword-colliding distractors (compete for top-k slots) ---
        Step::Write {
            content: "JWT token refresh rotation strategy is still undecided for the api",
            category: Episodic,
            importance: 0.3,
            event_type: "conversation",
        },
        Step::Write {
            content: "Authentication rate limiting via token bucket on login endpoint",
            category: Episodic,
            importance: 0.3,
            event_type: "conversation",
        },
        Step::Write {
            content: "Discussed bcrypt cost factor tuning for password hashing performance",
            category: Episodic,
            importance: 0.3,
            event_type: "conversation",
        },
        Step::Write {
            content: "Database connection pool sizing for the users service api",
            category: Episodic,
            importance: 0.3,
            event_type: "conversation",
        },
        Step::Write {
            content: "Generic security review checklist mentions https and authentication",
            category: Episodic,
            importance: 0.3,
            event_type: "conversation",
        },
        Step::Write {
            content: "Implemented POST /auth/login endpoint with JWT token generation",
            category: Episodic,
            importance: 0.5,
            event_type: "change",
        },
        Step::Write {
            content: "Auth middleware pattern: extract JWT token from Authorization header",
            category: Procedural,
            importance: 0.6,
            event_type: "pattern",
        },
        // --- Feedback signals (importance/recency features react to these) ---
        Step::Rate {
            content: "Architecture decision: use JWT tokens for stateless authentication",
            category: Semantic,
            quality: 5,
        },
        Step::Rate {
            content: "Fixed authentication error: bcrypt hash comparison was case sensitive",
            category: Episodic,
            quality: 4,
        },
        // --- Queries at tight k=3 amid distractors (headroom on both metrics) ---
        Step::Recall {
            query: "JWT authentication token decision",
            k: 3,
            relevant: &["JWT tokens for stateless authentication"],
        },
        Step::Recall {
            query: "bcrypt password hashing error fix",
            k: 3,
            relevant: &[
                "bcrypt hash comparison was case sensitive",
                "bcrypt for password hashing",
            ],
        },
        Step::Recall {
            query: "database users table schema passwords",
            k: 3,
            relevant: &["users table with hashed passwords"],
        },
        Step::Recall {
            query: "security https endpoint requirement",
            k: 3,
            relevant: &["authentication endpoints must use HTTPS"],
        },
        // Broad query: many memories share these keywords, so distractors
        // compete for the 3 slots. precision@k stays < 1.0 here ⇒ headroom
        // for fusion/MMR/eviction to push relevant items above distractors.
        Step::Recall {
            query: "authentication security decision",
            k: 3,
            relevant: &[
                "JWT tokens for stateless authentication",
                "bcrypt for password hashing",
                "authentication endpoints must use HTTPS",
            ],
        },
    ]
}

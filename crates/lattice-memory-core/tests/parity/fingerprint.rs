// Behaviour fingerprint of the memory model: every public pure function run
// over a fixed input corpus, each result rendered exactly (floats as their
// bit patterns). The same source is compiled against the pre-extraction
// lattice-mcp modules (to produce `golden.txt`) and against this crate, so a
// line that differs is a behaviour change. Callers bring `m` (the model
// root) and `cfgm` (its config types) into scope before `include!`.

use chrono::{DateTime, Duration, Utc};

fn b32(x: f32) -> String {
    format!("{:08x}", x.to_bits())
}
fn bv(xs: &[f32]) -> String {
    xs.iter().map(|x| b32(*x)).collect::<Vec<_>>().join(",")
}
fn t0() -> DateTime<Utc> {
    DateTime::from_timestamp(1_780_000_000, 0).expect("valid timestamp")
}

fn mem(id: u64, content: &str, importance: f32, age_days: i64) -> m::types::Memory {
    m::types::Memory {
        id,
        category: m::types::MemoryCategory::Episodic,
        content: content.into(),
        created_at: t0() - Duration::days(age_days),
        last_accessed: t0() - Duration::hours(age_days * 3),
        access_count: id % 7,
        importance,
        session_id: Some(format!("s{}", id % 3)),
        metadata: m::types::MemoryMetadata::Episodic {
            event_type: "observation".into(),
        },
        importance_state: m::importance_state::ImportanceState::default(),
    }
}

#[allow(clippy::field_reassign_with_default)]
fn states() -> Vec<m::importance_state::ImportanceState> {
    let mut out = vec![m::importance_state::ImportanceState::default()];
    for (i, &(a, b)) in [(5.0_f32, 2.0_f32), (1.0, 9.0), (30.0, 30.0), (0.5, 0.25)]
        .iter()
        .enumerate()
    {
        let mut s = m::importance_state::ImportanceState::default();
        s.alpha = a;
        s.beta = b;
        s.fsrs_stability = 0.5 + i as f32 * 3.7;
        s.fsrs_difficulty = 0.1 + i as f32 * 0.2;
        s.sm2_easiness = 1.3 + i as f32 * 0.4;
        s.sm2_interval = 1.0 + i as f32 * 6.0;
        s.sm2_repetitions = i as u32 * 2;
        s.tfidf_uniqueness = 0.2 * i as f32;
        s.salience = 0.9 - 0.15 * i as f32;
        s.pagerank = 0.25 * i as f32;
        s.novelty = 0.1 + 0.2 * i as f32;
        s.last_review = Some(t0() - Duration::hours(17 * i as i64));
        out.push(s);
    }
    out
}

fn state_line(s: &m::importance_state::ImportanceState) -> String {
    format!(
        "{} {} {} {} {} {} {}",
        b32(s.alpha),
        b32(s.beta),
        b32(s.fsrs_stability),
        b32(s.fsrs_difficulty),
        b32(s.sm2_easiness),
        b32(s.sm2_interval),
        s.sm2_repetitions
    )
}

fn scored(id: u64, v: f32, g: f32, r: f32, i: f32, f: f32) -> m::types::ScoredMemory {
    let mut memory = mem(id, &format!("scored memory {id}"), i, id as i64);
    memory.importance_state.alpha = 1.0 + id as f32;
    memory.importance_state.beta = 1.0 + (id % 3) as f32;
    m::types::ScoredMemory {
        memory,
        vector_similarity: v,
        graph_relevance: g,
        temporal_recency: r,
        importance_score: i,
        final_score: f,
        related: vec![],
    }
}

const TEXTS: &[&str] = &[
    "Fixed the auth bug: JWT validation rejected tokens with a leading space",
    "Architecture decision: always route storage through the MemoryStore trait",
    "I prefer tabs; never commit generated files",
    "Security review found a vulnerability in the login rate limiter",
    "Deployed v2.3.1 to staging; healthcheck at internal/healthz passed",
    "error code E_AUTH_401 is returned when the bearer token has expired",
    "",
    "the and of",
];

pub fn fingerprint() -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let now = t0();

    // scoring
    for &(age_h, hl) in &[(0_i64, 30.0_f64), (24, 30.0), (24 * 45, 30.0), (24 * 400, 365.0), (-5, 7.0)] {
        let then = now - Duration::hours(age_h);
        out.push(format!(
            "scoring.decay {age_h} {hl} {} {}",
            b32(m::scoring::temporal_decay(then, now, hl)),
            b32(m::scoring::recency_score(then, now, hl))
        ));
    }
    for &(base, n) in &[(0.0_f32, 0_u64), (0.5, 3), (0.9, 50), (1.0, 1000)] {
        out.push(format!("scoring.access {} {n} {}", b32(base), b32(m::scoring::access_adjusted_importance(base, n))));
    }
    for (t, ev) in TEXTS.iter().zip(["error", "decision", "note", "x", "deploy", "error", "", "decision"]) {
        out.push(format!("scoring.auto {}", b32(m::scoring::auto_importance(t, ev))));
    }

    // importance: fsrs / sm2 / bayesian / composite
    let fsrs_cfg = cfgm::FsrsConfig::default();
    for c in m::types::MemoryCategory::all_storage() {
        out.push(format!("fsrs.init {:?} {}", c, b32(m::importance::fsrs::initial_stability(*c, &fsrs_cfg))));
    }
    for &(s, d) in &[(1.0_f32, 0.0_f64), (1.0, 1.0), (7.0, 3.5), (0.001, 2.0), (90.0, 400.0)] {
        out.push(format!("fsrs.retention {} {d} {}", b32(s), b32(m::importance::fsrs::retention(s, d))));
    }
    for &(iv, ef) in &[(1.0_f32, 2.5_f32), (6.0, 1.3), (30.0, 2.8), (0.0, 1.3)] {
        out.push(format!(
            "sm2.prio {} {} {} {}",
            b32(iv),
            b32(ef),
            b32(m::importance::sm2::priority(iv, ef)),
            b32(m::importance::sm2::mastery(iv, ef))
        ));
    }
    let weights = cfgm::ImportanceWeights::default();
    for (i, st) in states().iter().enumerate() {
        out.push(format!(
            "bayes {i} {} {} {}",
            b32(m::importance::bayesian::posterior_mean(st.alpha, st.beta)),
            b32(m::importance::bayesian::posterior_with_exploration(st.alpha, st.beta, 0.5)),
            b32(m::importance::composite_importance(st, now, &weights))
        ));
        for q in 0..=5_u8 {
            let mut f = st.clone();
            m::importance::fsrs::update_state(&mut f, q, 2.5 * i as f64);
            let mut s2 = st.clone();
            m::importance::sm2::update(&mut s2, q);
            let mut by = st.clone();
            m::importance::bayesian::update(&mut by, q >= 3, 10.0 * i as f64);
            out.push(format!("upd {i} {q} | {} | {} | {}", state_line(&f), state_line(&s2), state_line(&by)));
        }
    }

    // pagerank (graphs chosen so no node has two in-links: summation order
    // is then fixed, and the old HashMap iteration order cannot matter)
    for (gi, g) in [
        vec![(1_u64, vec![2_u64]), (2, vec![3]), (3, vec![1])],
        vec![(10, vec![11, 12, 13]), (11, vec![]), (12, vec![14])],
        vec![(5, vec![6]), (7, vec![8]), (9, vec![])],
    ]
    .iter()
    .enumerate()
    {
        let pr = m::importance::pagerank::compute_pagerank(g, 0.85, 20);
        let mut v: Vec<(u64, f32)> = pr.iter().map(|(k, s)| (*k, *s)).collect();
        v.sort_by_key(|x| x.0);
        out.push(format!(
            "pagerank {gi} {}",
            v.iter().map(|(k, s)| format!("{k}:{}", b32(*s))).collect::<Vec<_>>().join(",")
        ));
    }

    // write-time signals
    for sims in [vec![], vec![0.1_f32], vec![0.95, 0.4, 0.2], vec![0.99, 0.98], vec![-0.2, 0.3]] {
        out.push(format!(
            "write {} | {} {} {} {}",
            bv(&sims),
            b32(m::importance::write_time::tfidf_uniqueness(&sims)),
            b32(m::importance::write_time::salience(&sims)),
            b32(m::write_surprise::novelty(&sims)),
            m::write_surprise::is_near_dup(m::write_surprise::novelty(&sims), 0.05)
        ));
    }

    // contradiction
    for (a, b) in [
        ("auth must use bcrypt for password hashing", "auth must not use bcrypt for password hashing"),
        ("use bcrypt for password hashing", "use bcrypt to hash passwords"),
        ("alpha beta gamma", "delta epsilon zeta"),
        ("", "never"),
    ] {
        let d = m::contradiction::lexical_disagreement(a, b);
        out.push(format!(
            "contra {} {} {}",
            b32(d),
            m::contradiction::is_likely_contradiction(0.95, d, 0.9),
            m::contradiction::is_likely_contradiction(0.85, d, 0.9)
        ));
    }

    // consolidation (distinct shared-term counts: no tie for the old
    // HashMap order to break)
    let cluster = vec![
        mem(1, "deploy pipeline staging broke after the cache migration", 0.6, 1),
        mem(2, "deploy pipeline staging recovered", 0.8, 2),
        mem(3, "deploy pipeline flush step", 0.5, 3),
        mem(4, "deploy notes for friday", 0.45, 40),
    ];
    out.push(format!("consol.summary {}", m::consolidation::algorithmic_summarize(&cluster)));
    out.push(format!("consol.summary1 {}", m::consolidation::algorithmic_summarize(&cluster[..1])));
    out.push(format!("consol.summary0 [{}]", m::consolidation::algorithmic_summarize(&[])));
    for (mo, mi, hl) in [(2_usize, 0.4_f32, 7.0_f64), (1, 0.5, 1.0), (3, 0.0, 30.0)] {
        let p = m::consolidation::ClusterParams {
            min_overlap: mo,
            min_importance: mi,
            temporal_half_life_days: hl,
        };
        out.push(format!("consol.clusters {mo} {:?}", m::consolidation::detect_clusters(&cluster, p)));
    }
    let dk = m::consolidation::distill::algorithmic_distill(&cluster);
    out.push(format!(
        "distill {:?} {:?} {:?} {:?} | {}",
        dk.core,
        dk.context,
        dk.theme,
        dk.provenance,
        m::consolidation::distill::render(&dk)
    ));
    for w in [0_usize, 3, 10, 11] {
        out.push(format!("trigger {w} {}", m::consolidation::trigger::should_consolidate(w, 10)));
    }

    // retrieval
    let rc = cfgm::RetrievalConfig {
        vector_weight: 0.4,
        graph_weight: 0.25,
        recency_weight: 0.2,
        importance_weight: 0.15,
        graph_hop_depth: 2,
    };
    for &(v, g, r, i) in &[(0.9_f32, 0.0_f32, 1.0_f32, 0.5_f32), (0.3, 0.7, 0.2, 0.9), (0.0, 0.0, 0.0, 0.0)] {
        let h = m::retrieval::hybrid_score(v, g, r, i, &rc);
        out.push(format!("hybrid {} {}", b32(h), b32(m::retrieval::blend_reranker_score(h, 0.77, 0.5))));
    }
    let neighbors = [(1_u64, 1_usize), (2, 2), (9, 1), (3, 3)];
    for ids in [vec![1_u64, 2, 3], vec![9], vec![], vec![4, 5]] {
        out.push(format!(
            "graphrel {:?} {}",
            ids,
            b32(m::retrieval::graph_relevance_from_traversal(&neighbors, &ids.iter().copied().collect()))
        ));
    }
    let cands = vec![
        scored(1, 0.9, 0.1, 0.5, 0.4, 0.61),
        scored(2, 0.7, 0.6, 0.2, 0.9, 0.70),
        scored(3, 0.8, 0.0, 0.9, 0.1, 0.55),
        scored(2, 0.1, 0.1, 0.1, 0.1, 0.10),
        scored(4, 0.2, 0.9, 0.3, 0.6, 0.48),
    ];
    let ranked: Vec<u64> = m::retrieval::rank_and_dedup(cands.clone(), 3).iter().map(|s| s.memory.id).collect();
    out.push(format!("rank {ranked:?}"));
    for c in [0.0_f32, 0.5, 0.98] {
        out.push(format!("fusion.w {}", bv(&m::retrieval::fusion::confidence_weights(c, &rc))));
    }
    let sig: Vec<[f32; 4]> = cands
        .iter()
        .map(|s| [s.vector_similarity, s.graph_relevance, s.temporal_recency, s.importance_score])
        .collect();
    out.push(format!("fusion.rrf {}", bv(&m::retrieval::fusion::reciprocal_rank_fusion(&sig, 60.0))));
    for mode in [cfgm::FusionMode::ConfidenceWeighted, cfgm::FusionMode::Rrf] {
        let fc = cfgm::FusionConfig { enabled: true, mode, rrf_k: 60.0 };
        let mut c = cands.clone();
        m::retrieval::fusion::apply(&fc, &rc, &mut c);
        out.push(format!("fusion.apply {:?} {}", mode, bv(&c.iter().map(|s| s.final_score).collect::<Vec<_>>())));
    }
    let vecs = vec![vec![1.0_f32, 0.0, 0.0], vec![0.99, 0.1, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0], vec![0.5, 0.5, 0.7]];
    let rel = [0.9_f32, 0.88, 0.6, 0.5, 0.4];
    for lambda in [1.0_f32, 0.7, 0.3, 0.0] {
        out.push(format!("mmr {} {:?}", b32(lambda), m::retrieval::diversify::mmr(&vecs, &rel, lambda, 4)));
    }
    for q in ["what is E_AUTH_401", "how are user passwords stored", "the of and", "v2.3.1 internal/healthz"] {
        out.push(format!(
            "lexical {:?} | {}",
            m::retrieval::query::keyword_expand(q),
            TEXTS.iter().map(|t| b32(m::retrieval::lexical::lexical_overlap(q, t))).collect::<Vec<_>>().join(",")
        ));
    }

    // lifecycle
    let items: Vec<m::lifecycle::LifecycleItem> = (0..8_u64)
        .map(|i| m::lifecycle::LifecycleItem {
            id: i,
            utility: [0.9_f32, 0.1, 0.5, 0.5, 0.05, 0.7, 0.2, 0.5][i as usize],
            created_at: now - Duration::days(i as i64),
        })
        .collect();
    for (budget, del, below) in [(10_usize, true, 0.3_f32), (4, false, 0.3), (4, true, 0.3), (0, true, 1.0)] {
        let p = m::lifecycle::eviction_plan(&items, budget, del, below);
        out.push(format!("evict {budget} {del} {:?} {:?}", p.evict, p.demote));
    }

    // avatar
    for (prior, new) in [(None, "rust"), (Some("Rust "), "rust"), (Some("rust"), "go")] {
        let d = m::avatar::merge::decide(prior, new);
        let mut st = m::importance_state::ImportanceState::default();
        m::avatar::merge::apply(d, &mut st);
        m::avatar::merge::apply(d, &mut st);
        out.push(format!("avatar.merge {:?} {} {}", d, state_line(&st), b32(m::avatar::merge::confidence(&st))));
    }
    for t in [
        "My preferred language is Rust. Editor: helix, and timezone is UTC+2",
        "tooling: cargo, nextest, and just; os platform is linux",
        "nothing here",
    ] {
        out.push(format!("avatar.derive {:?}", m::avatar::derive::extract_explicit(t)));
    }
    for v in ["mail me at jane.doe@example.com or +1 555 123 4567", "Acme Corp builds it", "plain"] {
        let r = m::avatar::redact::redact(v, &["acme".to_string()]);
        out.push(format!("avatar.redact {:?} {} {:?}", r.text, r.redacted, r.kinds));
    }
    for s in ["Preferred Language", "tooling", "favourite-colour", ""] {
        let n = m::avatar::schema::normalize_slot(s);
        out.push(format!(
            "avatar.schema {:?} {:?} {:?} {:?} {:?}",
            n,
            m::avatar::schema::value_kind_for(&n),
            m::avatar::schema::validate_slot(&n, false),
            m::avatar::schema::validate_slot(&n, true),
            m::avatar::schema::canonical_render(&n, "x")
        ));
    }

    // types: ids, payload bytes, relations
    for c in m::types::MemoryCategory::all_storage() {
        out.push(format!(
            "id {} {} {}",
            c.collection_name(),
            m::types::Memory::content_id("same text", *c),
            m::types::Memory::avatar_point_id("default", c.collection_name())
        ));
        let rels: Vec<String> = m::types::relations::relations_for(*c).map(|(n, i)| format!("{n}={i}")).collect();
        out.push(format!("relations {} {}", c.collection_name(), rels.join(",")));
    }
    let mut avatar = mem(77, "preferred_language: rust", 0.7, 2);
    avatar.category = m::types::MemoryCategory::Avatar;
    avatar.session_id = None;
    avatar.metadata = m::types::MemoryMetadata::Avatar {
        avatar_id: "default".into(),
        slot_key: "preferred_language".into(),
        value: "rust".into(),
        value_kind: m::types::ValueKind::Scalar,
        confidence: 0.75,
        provenance: vec![3, 1],
        evidence_count: 2,
        prior_value: Some("go".into()),
    };
    avatar.importance_state = states()[2].clone();
    for mm in [mem(5, "payload \"quoted\" ünïcode", 0.33, 9), avatar] {
        let p = mm.to_payload();
        let mut keys: Vec<&String> = p.keys().collect();
        keys.sort();
        for k in keys {
            out.push(format!("payload {} {k} {}", mm.id, String::from_utf8_lossy(&p[k])));
        }
        let back = m::types::Memory::from_payload(mm.id, &p).expect("roundtrip");
        out.push(format!("payload.back {}", serde_json::to_string(&back).expect("json")));
    }
    out
}

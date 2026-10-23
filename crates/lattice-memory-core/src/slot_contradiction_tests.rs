use super::*;

fn fact(id: u64, subject: &str, slot: &str, value: &str) -> SlotFact {
    SlotFact {
        id,
        key: SlotKey::new(subject, slot),
        value: value.into(),
    }
}

fn at() -> DateTime<Utc> {
    DateTime::from_timestamp(1_780_000_000, 0).expect("valid timestamp")
}

#[test]
fn different_values_on_one_key_conflict() {
    let facts = [
        fact(7, "auth", "hash", "bcrypt"),
        fact(3, "auth", "hash", "argon2"),
    ];
    let c = open_conflicts(&facts, &[]);
    assert_eq!(
        c,
        vec![SlotConflict {
            key: SlotKey::new("auth", "hash"),
            ids: (3, 7)
        }]
    );
}

#[test]
fn same_value_after_normalization_reinforces() {
    let facts = [
        fact(1, "auth", "hash", "bcrypt"),
        fact(2, " Auth ", "HASH", "  BCrypt "),
    ];
    assert!(open_conflicts(&facts, &[]).is_empty());
}

#[test]
fn different_subject_or_slot_never_conflicts() {
    // Embedding-near statements about different keys: a similarity gate
    // would pair these; the key rule must not.
    let facts = [
        fact(1, "auth", "hash", "bcrypt"),
        fact(2, "auth", "cipher", "aes"),
        fact(3, "billing", "hash", "sha256"),
    ];
    assert!(open_conflicts(&facts, &[]).is_empty());
}

#[test]
fn key_normalization_joins_spacing_and_case() {
    assert_eq!(
        SlotKey::new("Preferred  Language", " x"),
        SlotKey::new("preferred language", "X")
    );
    assert_ne!(SlotKey::new("a", "bc"), SlotKey::new("ab", "c"));
}

#[test]
fn a_settlement_closes_the_conflict_and_retires_the_loser() {
    let mut facts = vec![
        fact(1, "gpu", "clock", "2400"),
        fact(2, "gpu", "clock", "513"),
    ];
    let c = open_conflicts(&facts, &[]);
    let s = settle(&c[0], 1, Adjudicator::User, at()).expect("winner is in the pair");
    assert_eq!((s.winner, s.superseded), (1, 2));
    assert!(open_conflicts(&facts, core::slice::from_ref(&s)).is_empty());

    // A new value conflicts with the winner, not with the retired loser.
    facts.push(fact(9, "gpu", "clock", "2000"));
    assert_eq!(
        open_conflicts(&facts, &[s])
            .into_iter()
            .map(|c| c.ids)
            .collect::<Vec<_>>(),
        vec![(1, 9)]
    );
}

#[test]
fn a_settlement_on_another_key_does_not_retire_the_id() {
    let facts = [
        fact(1, "gpu", "clock", "2400"),
        fact(2, "gpu", "clock", "513"),
    ];
    let other = Settlement {
        key: SlotKey::new("gpu", "power"),
        winner: 1,
        superseded: 2,
        adjudicated_by: Adjudicator::Rule("newest-wins".into()),
        at: at(),
    };
    assert_eq!(open_conflicts(&facts, &[other]).len(), 1);
}

#[test]
fn three_way_conflict_lists_every_pair_in_order() {
    let facts = [
        fact(5, "s", "k", "a"),
        fact(4, "s", "k", "b"),
        fact(6, "s", "k", "c"),
    ];
    let ids: Vec<_> = open_conflicts(&facts, &[])
        .into_iter()
        .map(|c| c.ids)
        .collect();
    assert_eq!(ids, vec![(4, 5), (4, 6), (5, 6)]);
}

#[test]
fn settle_refuses_a_winner_outside_the_pair() {
    let c = SlotConflict {
        key: SlotKey::new("s", "k"),
        ids: (1, 2),
    };
    let e = settle(&c, 3, Adjudicator::Llm, at()).unwrap_err();
    assert!(matches!(
        e,
        MemoryCoreError::SettlementWinnerNotInConflict {
            winner: 3,
            a: 1,
            b: 2
        }
    ));
}

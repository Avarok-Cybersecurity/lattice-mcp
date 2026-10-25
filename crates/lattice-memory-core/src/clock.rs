//! Clock port. The core never reads the wall clock; anything that needs
//! "now" takes it as an argument, and hosts obtain it from a [`Clock`].
//! chrono's `clock` feature is not enabled for this crate, so `Utc::now()`
//! does not compile here — that is the gate.

use chrono::{DateTime, Utc};

pub trait Clock {
    fn now(&self) -> DateTime<Utc>;
}

/// A clock that always returns the instant it was built with (tests, replays).
#[derive(Debug, Clone, Copy)]
pub struct FixedClock(pub DateTime<Utc>);

impl Clock for FixedClock {
    fn now(&self) -> DateTime<Utc> {
        self.0
    }
}

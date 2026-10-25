//! The host's wall clock, behind the core's `Clock` port.

use chrono::{DateTime, Utc};
use lattice_memory_core::clock::Clock;

#[derive(Debug, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

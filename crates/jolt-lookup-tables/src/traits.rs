//! Lookup-table-related traits.

use jolt_field::JoltField;
use std::fmt::Debug;

use crate::challenge_ops::{ChallengeOps, FieldOps};

/// Materialize and MLE-evaluate a single lookup table.
pub trait LookupTable: Clone + Debug + Send + Sync {
    fn materialize_entry(&self, index: u128) -> u64;

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: JoltField + FieldOps<C>;
}

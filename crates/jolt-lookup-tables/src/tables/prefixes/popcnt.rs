use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// Population count of the left operand's bound bits.
pub enum PopcntPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for PopcntPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        let (x, _) = b.uninterleave();
        checkpoints[Prefixes::Popcnt] + F::from_u32(u64::from(x).count_ones())
    }
}

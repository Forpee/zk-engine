use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// Trailing zeros of the left operand's bound bits (their count when all are
/// zero). Bits are bound MSB-first: a non-zero chunk resets the count to its
/// own trailing zeros; an all-zero chunk adds its width.
pub enum CtzPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for CtzPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        let (x, _) = b.uninterleave();
        let v = u64::from(x);
        if v == 0 {
            checkpoints[Prefixes::Ctz] + F::from_u64(x.len() as u64)
        } else {
            F::from_u32(v.trailing_zeros())
        }
    }
}

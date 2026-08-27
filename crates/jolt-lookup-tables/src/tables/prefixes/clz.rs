use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

/// Leading zeros of the left operand's bound bits. Bits are bound MSB-first,
/// so a chunk contributes its own leading zeros only while every previously
/// bound bit is zero — the [`Prefixes::LeftOperandIsZero`] checkpoint.
pub enum ClzPrefix {}

/// Leading zeros of `x` within an `n`-bit window.
pub(crate) fn leading_zeros(x: LookupBits) -> u64 {
    let n = x.len() as u64;
    let v = u64::from(x);
    if v == 0 {
        n
    } else {
        n - u64::from(u64::BITS - v.leading_zeros())
    }
}

impl<F: JoltField> SparseDensePrefix<F> for ClzPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        let (x, _) = b.uninterleave();
        checkpoints[Prefixes::Clz]
            + checkpoints[Prefixes::LeftOperandIsZero] * F::from_u64(leading_zeros(x))
    }
}

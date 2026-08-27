//! Suffix polynomial evaluations for the sparse-dense decomposition.
//!
//! Each suffix computes a function over the "unbound" low-order bits of a
//! lookup index during the sumcheck protocol. Suffixes evaluate to `u64`
//! values (not field elements), making them cheap to compute and
//! field-independent.
//!
//! The decomposition works as: `table_mle(r) = Σ prefix_i(r_high) · suffix_i(b_low)`,
//! where `b_low` ranges over the Boolean hypercube.

use crate::lookup_bits::LookupBits;

mod and;
mod andnot;
mod clz_low;
mod ctz_low;
mod eq;
mod left_is_zero;
mod left_shift;
mod lower_half_word;
mod lower_word;
mod lt;
mod one;
mod or;
mod overflow_bits_zero;
mod popcnt;
mod pow2;
mod right_is_zero;
mod right_operand;
mod right_shift;
mod right_shift_helper;
mod sign_extension;
mod sign_extension_upper_half;
mod xor;

use and::AndSuffix;
use andnot::AndNotSuffix;
use clz_low::ClzLowSuffix;
use ctz_low::CtzLowSuffix;
use eq::EqSuffix;
use left_is_zero::LeftOperandIsZeroSuffix;
use left_shift::LeftShiftSuffix;
use lower_half_word::LowerHalfWordSuffix;
use lower_word::LowerWordSuffix;
use lt::LessThanSuffix;
use one::OneSuffix;
use or::OrSuffix;
use overflow_bits_zero::OverflowBitsZeroSuffix;
use popcnt::PopcntSuffix;
// Shared bit-manipulation helpers: single source for the pext packing and
// the window-sign convention, reused by the corresponding tables/prefixes.
use pow2::Pow2Suffix;
use right_is_zero::RightOperandIsZeroSuffix;
use right_operand::RightOperandSuffix;
use right_shift::RightShiftSuffix;
use right_shift_helper::RightShiftHelperSuffix;
use sign_extension::SignExtensionSuffix;
use sign_extension_upper_half::SignExtensionUpperHalfSuffix;
use xor::XorSuffix;

use jolt_field::JoltField;

/// A suffix polynomial: evaluates on unbound Boolean variables during sumcheck.
///
/// Suffixes return `u64` values (not field elements) to avoid unnecessary
/// field arithmetic when the result is a small integer.
pub trait SparseDenseSuffix: 'static + Sync {
    /// Evaluate this suffix's MLE on bitvector `b`, where `b.len()` variables
    /// are set to Boolean values.
    fn suffix_mle(b: LookupBits) -> u64;
}

/// Type alias for suffix evaluations promoted to field elements.
pub type SuffixEval<F> = F;

/// All suffix types used by Jolt's lookup tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum::EnumCount)]
#[repr(u8)]
pub enum Suffixes {
    One,
    And,
    AndNot,
    Xor,
    Or,
    RightOperand,
    LowerWord,
    LowerHalfWord,
    LessThan,
    Eq,
    LeftOperandIsZero,
    RightOperandIsZero,
    Pow2,
    RightShift,
    RightShiftHelper,
    SignExtension,
    LeftShift,
    SignExtensionUpperHalf,
    OverflowBitsZero,
    /// Population count of the left operand's suffix bits.
    Popcnt,
    /// Leading zeros of the left operand's suffix bits.
    ClzLow,
    /// Trailing zeros of the left operand's suffix bits.
    CtzLow,
}

/// Total number of suffix variants.
pub const NUM_SUFFIXES: usize = <Suffixes as strum::EnumCount>::COUNT;

impl Suffixes {
    /// Returns `true` if this suffix's output is guaranteed to be in {0, 1}.
    ///
    /// This enables micro-optimizations in the sumcheck prover that avoid
    /// multiplying by 1 (directly adding the unreduced field element instead).
    #[inline(always)]
    pub fn is_01_valued(&self) -> bool {
        matches!(
            self,
            Suffixes::One
                | Suffixes::Eq
                | Suffixes::LessThan
                | Suffixes::LeftOperandIsZero
                | Suffixes::RightOperandIsZero
                | Suffixes::OverflowBitsZero
        )
    }

    /// Evaluate this suffix's MLE on bitvector `b`.
    pub fn suffix_mle(&self, b: LookupBits) -> u64 {
        match self {
            Suffixes::One => OneSuffix::suffix_mle(b),
            Suffixes::And => AndSuffix::suffix_mle(b),
            Suffixes::AndNot => AndNotSuffix::suffix_mle(b),
            Suffixes::Or => OrSuffix::suffix_mle(b),
            Suffixes::Xor => XorSuffix::suffix_mle(b),
            Suffixes::RightOperand => RightOperandSuffix::suffix_mle(b),
            Suffixes::LowerWord => LowerWordSuffix::suffix_mle(b),
            Suffixes::LowerHalfWord => LowerHalfWordSuffix::suffix_mle(b),
            Suffixes::LessThan => LessThanSuffix::suffix_mle(b),
            Suffixes::Eq => EqSuffix::suffix_mle(b),
            Suffixes::LeftOperandIsZero => LeftOperandIsZeroSuffix::suffix_mle(b),
            Suffixes::RightOperandIsZero => RightOperandIsZeroSuffix::suffix_mle(b),
            Suffixes::Pow2 => Pow2Suffix::suffix_mle(b),
            Suffixes::RightShift => RightShiftSuffix::suffix_mle(b),
            Suffixes::RightShiftHelper => RightShiftHelperSuffix::suffix_mle(b),
            Suffixes::SignExtension => SignExtensionSuffix::suffix_mle(b),
            Suffixes::LeftShift => LeftShiftSuffix::suffix_mle(b),
            Suffixes::SignExtensionUpperHalf => SignExtensionUpperHalfSuffix::suffix_mle(b),
            Suffixes::OverflowBitsZero => OverflowBitsZeroSuffix::suffix_mle(b),
            Suffixes::Popcnt => PopcntSuffix::suffix_mle(b),
            Suffixes::ClzLow => ClzLowSuffix::suffix_mle(b),
            Suffixes::CtzLow => CtzLowSuffix::suffix_mle(b),
        }
    }

    /// Evaluate and promote to a field element.
    #[inline]
    pub fn evaluate<F: JoltField>(&self, b: LookupBits) -> SuffixEval<F> {
        F::from_u64(self.suffix_mle(b))
    }
}

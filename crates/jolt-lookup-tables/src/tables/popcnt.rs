//! `popcnt(x)`: population count of the left operand (the right operand is
//! ignored). Used by the WebAssembly catalog; not part of the RV64
//! the RV64 table catalog.

use jolt_field::JoltField;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PopcntTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for PopcntTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, _) = uninterleave_bits(index);
        u64::from(x.count_ones())
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: JoltField + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut result = F::zero();
        for i in 0..XLEN {
            result += r[2 * i].into();
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for PopcntTable<XLEN> {
    fn prefixes(&self) -> &'static [Prefixes] {
        &[Prefixes::Popcnt]
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::One, Suffixes::Popcnt]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [one, popcnt] = suffixes.try_into().unwrap();
        prefixes[Prefixes::Popcnt] * one + popcnt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_full_hypercube_test, mle_random_test, prefix_suffix_test};
    use crate::XLEN;
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, PopcntTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, PopcntTable<8>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, PopcntTable<XLEN>>();
    }
}

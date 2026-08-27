//! `ctz(x)`: trailing zeros of the left operand (`XLEN` when it is zero); the
//! right operand is ignored. Used by the WebAssembly catalog; not part of the
//! the RV64 table catalog.
//!
//! MLE: `Σ_k k · x_k · Π_{j<k} (1 − x_j) + XLEN · Π_j (1 − x_j)` with `x_k`
//! the bit of weight `2^k`.

use jolt_field::JoltField;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct CtzTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for CtzTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, _) = uninterleave_bits(index);
        if x == 0 {
            XLEN as u64
        } else {
            u64::from(x.trailing_zeros())
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: JoltField + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut result = F::zero();
        let mut all_zero_so_far = F::one();
        for k in 0..XLEN {
            let x_k = r[2 * (XLEN - 1 - k)];
            result += all_zero_so_far * x_k * F::from_u64(k as u64);
            all_zero_so_far *= F::one() - x_k;
        }
        result + all_zero_so_far * F::from_u64(XLEN as u64)
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for CtzTable<XLEN> {
    fn prefixes(&self) -> &'static [Prefixes] {
        &[Prefixes::Ctz]
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::CtzLow, Suffixes::LeftOperandIsZero]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [ctz_low, left_is_zero] = suffixes.try_into().unwrap();
        ctz_low + prefixes[Prefixes::Ctz] * left_is_zero
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{
        mle_full_hypercube_test, mle_random_test, prefix_suffix_materialization_test,
        prefix_suffix_test,
    };
    use crate::XLEN;
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, CtzTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, CtzTable<8>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, CtzTable<XLEN>>();
        prefix_suffix_materialization_test::<XLEN, Fr, CtzTable<XLEN>>(2, 3);
    }
}

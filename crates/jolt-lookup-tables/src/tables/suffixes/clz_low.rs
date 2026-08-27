use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::tables::prefixes::clz::leading_zeros;

/// Leading zeros of the left operand's suffix bits (their count when zero).
pub enum ClzLowSuffix {}

impl SparseDenseSuffix for ClzLowSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, _) = b.uninterleave();
        leading_zeros(x)
    }
}

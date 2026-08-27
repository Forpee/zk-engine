use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// Trailing zeros of the left operand's suffix bits (their count when zero).
pub enum CtzLowSuffix {}

impl SparseDenseSuffix for CtzLowSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, _) = b.uninterleave();
        let v = u64::from(x);
        if v == 0 {
            x.len() as u64
        } else {
            u64::from(v.trailing_zeros())
        }
    }
}

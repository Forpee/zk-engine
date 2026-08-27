use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// Population count of the left operand's suffix bits.
pub enum PopcntSuffix {}

impl SparseDenseSuffix for PopcntSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, _) = b.uninterleave();
        u64::from(u64::from(x).count_ones())
    }
}

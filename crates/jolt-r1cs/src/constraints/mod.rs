//! ISA-specific R1CS constraint definitions.

pub mod jolt;
pub mod wasm;

use jolt_field::Field;

use crate::constraint::SparseRow;

/// Sparse row from `(column, coefficient)` pairs with `i64`-range
/// coefficients (zero entries dropped).
#[expect(
    clippy::expect_used,
    reason = "compile-time constant table; silent i128→i64 truncation would be a correctness bug"
)]
pub(super) fn row<F: Field>(entries: &[(usize, i128)]) -> SparseRow<F> {
    entries
        .iter()
        .filter(|(_, c)| *c != 0)
        .map(|&(idx, c)| {
            let narrow = i64::try_from(c).expect("coefficient out of i64 range; use row_wide");
            (idx, F::from_i64(narrow))
        })
        .collect()
}

/// Sparse row from `i128` coefficients, for constants that do not fit `i64`
/// (e.g. the 2^64 two's-complement bias).
pub(super) fn row_wide<F: Field>(entries: &[(usize, i128)]) -> SparseRow<F> {
    entries
        .iter()
        .filter(|(_, c)| *c != 0)
        .map(|&(idx, c)| (idx, F::from_i128(c)))
        .collect()
}

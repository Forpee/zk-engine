//! Lookup table definitions for the instruction lookup argument. Concrete
//! table implementations (catalogued per frontend, e.g. `jolt-wasm-tables`)
//! provide [`materialize_entry`](crate::LookupTable::materialize_entry) for
//! preprocessing and [`evaluate_mle`](crate::LookupTable::evaluate_mle) for
//! the sumcheck verifier.
//!
//! All tables are generic over `const XLEN: usize`. The supported word sizes
//! are `XLEN = 64` (production) and `XLEN = 8` (full-hypercube tests).

use jolt_field::JoltField;

pub mod align_addr;
pub mod and;
pub mod andn;
pub mod clz;
pub mod ctz;
pub mod equal;
pub mod halfword_alignment;
pub mod lower_half_word;
pub mod mulu_no_overflow;
pub mod not_equal;
pub mod or;
pub mod pext;
pub mod pext_signed;
pub mod popcnt;
pub mod pow2;
pub mod pow2_w;
pub mod prefixes;
pub mod range_check;
pub mod range_check_aligned;
pub mod shift_right_bitmask;
pub mod shift_right_bitmask_w;
pub mod sign_extend_word;
pub mod sign_mask;
pub mod signed_greater_than_equal;
pub mod signed_less_than;
pub mod suffixes;
pub mod unsigned_greater_than_equal;
pub mod unsigned_less_than;
pub mod unsigned_less_than_equal;
pub mod upper_word;
pub mod valid_div0;
pub mod valid_unsigned_remainder;
pub mod virtual_negate_if;
pub mod virtual_rev8w;
pub mod virtual_rotr;
pub mod virtual_rotrw;
pub mod virtual_sra;
pub mod virtual_sraw;
pub mod virtual_srl;
pub mod virtual_srlw;
pub mod virtual_xor_rot;
pub mod virtual_xor_rotw;
pub mod window_mask_b;
pub mod window_mask_h;
pub mod window_mask_w;
pub mod word_alignment;
pub mod xor;

pub use prefixes::{PrefixEval, Prefixes};
pub use suffixes::{SuffixEval, Suffixes};

/// Identifies a lookup table type at a given word size.
///
/// Each variant carries the corresponding zero-sized table marker. Instructions
/// declare which table they use via
/// [`InstructionLookupTable::lookup_table`](crate::InstructionLookupTable::lookup_table).
///
/// Variant indices match `jolt-prover-legacy::LookupTables` so lookup-table flags in
/// core-produced proofs can be interpreted without an adapter.
/// Prefix/suffix decomposition for sub-linear MLE evaluation.
///
/// Each lookup table decomposes its MLE as:
/// ```text
/// table_mle(r) = Σ_i prefix_i(r_high) · suffix_i(r_low)
/// ```
///
/// where the sum is over a small number of prefix-suffix pairs.
/// This enables the sumcheck prover to avoid materializing the entire table.
pub trait PrefixSuffixDecomposition<const XLEN: usize>: crate::LookupTable + Default {
    /// The prefix types used in this table's decomposition.
    fn prefixes(&self) -> &'static [Prefixes];

    /// The suffix types used in this table's decomposition.
    fn suffixes(&self) -> &'static [Suffixes];

    /// Recombine evaluated prefix and suffix values into the table's MLE evaluation.
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F;

    /// Generate a random lookup index inside the table's valid input domain,
    /// for testing.
    ///
    /// The default returns a uniform random `u128` masked to `2 * XLEN` bits.
    /// Tables with constrained input domains (e.g., shift/rotate tables that
    /// expect bitmask-shaped right operands) override this; off-domain
    /// indices are unreachable in real traces, and the prefix-suffix
    /// decomposition only matches `materialize_entry` on the valid domain.
    #[cfg(test)]
    fn random_lookup_index(rng: &mut rand::rngs::StdRng) -> u128 {
        let raw: u128 = rand::Rng::gen(rng);
        if XLEN == 64 {
            raw
        } else {
            raw & ((1u128 << (2 * XLEN)) - 1)
        }
    }
}

#[cfg(test)]
pub(crate) mod test_utils;

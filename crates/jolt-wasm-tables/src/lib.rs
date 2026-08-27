//! The WebAssembly lookup-table catalog.
//!
//! [`WasmTable`] realizes every [`AluOp`] as one prefix–suffix-decomposable
//! Jolt lookup table: the RISC-V-neutral tables of `jolt-lookup-tables` plus
//! the WebAssembly-only `Clz`/`Ctz`/`Popcnt` tables. This is the WASM
//! analogue of the RV64 `LookupTableKind` — the enum the instruction lookup
//! argument keys on — kept separate so the WASM catalog is its own
//! append-only universe (and stays within the kernels' 6-bit table-id cap).
//!
//! [`lookup_index`] forms the 128-bit lookup index from a row's instruction
//! inputs according to the op's [`OperandMode`], and each table's
//! `materialize_entry` at that index equals [`AluOp::evaluate`] — the
//! property `tests/catalog.rs` checks on random inputs and on real traces.

#![forbid(unsafe_code)]

use jolt_field::JoltField;
use jolt_lookup_tables::tables::and::AndTable;
use jolt_lookup_tables::tables::andn::AndnTable;
use jolt_lookup_tables::tables::clz::ClzTable;
use jolt_lookup_tables::tables::ctz::CtzTable;
use jolt_lookup_tables::tables::equal::EqualTable;
use jolt_lookup_tables::tables::lower_half_word::LowerHalfWordTable;
use jolt_lookup_tables::tables::mulu_no_overflow::MulUNoOverflowTable;
use jolt_lookup_tables::tables::not_equal::NotEqualTable;
use jolt_lookup_tables::tables::or::OrTable;
use jolt_lookup_tables::tables::popcnt::PopcntTable;
use jolt_lookup_tables::tables::pow2::Pow2Table;
use jolt_lookup_tables::tables::range_check::RangeCheckTable;
use jolt_lookup_tables::tables::shift_right_bitmask::ShiftRightBitmaskTable;
use jolt_lookup_tables::tables::sign_extend_word::SignExtendWordTable;
use jolt_lookup_tables::tables::signed_greater_than_equal::SignedGreaterThanEqualTable;
use jolt_lookup_tables::tables::signed_less_than::SignedLessThanTable;
use jolt_lookup_tables::tables::unsigned_greater_than_equal::UnsignedGreaterThanEqualTable;
use jolt_lookup_tables::tables::unsigned_less_than::UnsignedLessThanTable;
use jolt_lookup_tables::tables::unsigned_less_than_equal::UnsignedLessThanEqualTable;
use jolt_lookup_tables::tables::virtual_negate_if::VirtualNegateIfTable;
use jolt_lookup_tables::tables::virtual_rotr::VirtualROTRTable;
use jolt_lookup_tables::tables::virtual_sra::VirtualSRATable;
use jolt_lookup_tables::tables::virtual_srl::VirtualSRLTable;
use jolt_lookup_tables::tables::xor::XorTable;
use jolt_lookup_tables::tables::{PrefixEval, Prefixes, SuffixEval, Suffixes};
use jolt_lookup_tables::{
    interleave_bits, ChallengeOps, FieldOps, LookupTable, PrefixSuffixDecomposition,
};
use jolt_wasm_ir::{AluOp, OperandMode, Width};
use serde::{Deserialize, Serialize};

/// Word size of the WebAssembly catalog.
pub const XLEN: usize = 64;

/// The WebAssembly lookup-table catalog. Append-only.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    strum::EnumCount,
    strum::EnumIter,
)]
#[repr(u8)]
pub enum WasmTable {
    RangeCheck(RangeCheckTable<XLEN>),
    LowerHalfWord(LowerHalfWordTable<XLEN>),
    And(AndTable<XLEN>),
    Andn(AndnTable<XLEN>),
    Or(OrTable<XLEN>),
    Xor(XorTable<XLEN>),
    Equal(EqualTable<XLEN>),
    NotEqual(NotEqualTable<XLEN>),
    UnsignedLessThan(UnsignedLessThanTable<XLEN>),
    SignedLessThan(SignedLessThanTable<XLEN>),
    UnsignedGreaterThanEqual(UnsignedGreaterThanEqualTable<XLEN>),
    SignedGreaterThanEqual(SignedGreaterThanEqualTable<XLEN>),
    UnsignedLessThanEqual(UnsignedLessThanEqualTable<XLEN>),
    Srl(VirtualSRLTable<XLEN>),
    Sra(VirtualSRATable<XLEN>),
    Rotr(VirtualROTRTable<XLEN>),
    NegateIf(VirtualNegateIfTable<XLEN>),
    MulUNoOverflow(MulUNoOverflowTable<XLEN>),
    Pow2(Pow2Table<XLEN>),
    ShiftRightBitmask(ShiftRightBitmaskTable<XLEN>),
    SignExtendWord(SignExtendWordTable<XLEN>),
    Clz(ClzTable<XLEN>),
    Ctz(CtzTable<XLEN>),
    Popcnt(PopcntTable<XLEN>),
}

macro_rules! dispatch {
    ($self:expr, $t:ident => $expr:expr) => {
        match $self {
            Self::RangeCheck($t) => $expr,
            Self::LowerHalfWord($t) => $expr,
            Self::And($t) => $expr,
            Self::Andn($t) => $expr,
            Self::Or($t) => $expr,
            Self::Xor($t) => $expr,
            Self::Equal($t) => $expr,
            Self::NotEqual($t) => $expr,
            Self::UnsignedLessThan($t) => $expr,
            Self::SignedLessThan($t) => $expr,
            Self::UnsignedGreaterThanEqual($t) => $expr,
            Self::SignedGreaterThanEqual($t) => $expr,
            Self::UnsignedLessThanEqual($t) => $expr,
            Self::Srl($t) => $expr,
            Self::Sra($t) => $expr,
            Self::Rotr($t) => $expr,
            Self::NegateIf($t) => $expr,
            Self::MulUNoOverflow($t) => $expr,
            Self::Pow2($t) => $expr,
            Self::ShiftRightBitmask($t) => $expr,
            Self::SignExtendWord($t) => $expr,
            Self::Clz($t) => $expr,
            Self::Ctz($t) => $expr,
            Self::Popcnt($t) => $expr,
        }
    };
}

/// The kernels pack a table id in 6 bits (`jolt-kernels`'s
/// `PACKED_TABLE_BITS`); the catalog must stay below that.
const _: () = assert!(WasmTable::COUNT < 64);

impl WasmTable {
    pub const COUNT: usize = <Self as strum::EnumCount>::COUNT;

    /// The table that realizes `op`.
    pub fn of(op: AluOp) -> Self {
        match op {
            AluOp::Add(Width::W64) | AluOp::Sub(Width::W64) | AluOp::Mul(Width::W64) => {
                Self::RangeCheck(RangeCheckTable)
            }
            AluOp::Add(Width::W32)
            | AluOp::Sub(Width::W32)
            | AluOp::Mul(Width::W32)
            | AluOp::LowerHalfWord => Self::LowerHalfWord(LowerHalfWordTable),
            AluOp::And => Self::And(AndTable),
            AluOp::Andn => Self::Andn(AndnTable),
            AluOp::Or => Self::Or(OrTable),
            AluOp::Xor => Self::Xor(XorTable),
            AluOp::Eq => Self::Equal(EqualTable),
            AluOp::Ne => Self::NotEqual(NotEqualTable),
            AluOp::LtU => Self::UnsignedLessThan(UnsignedLessThanTable),
            AluOp::LtS => Self::SignedLessThan(SignedLessThanTable),
            AluOp::GeU => Self::UnsignedGreaterThanEqual(UnsignedGreaterThanEqualTable),
            AluOp::GeS => Self::SignedGreaterThanEqual(SignedGreaterThanEqualTable),
            AluOp::LeU => Self::UnsignedLessThanEqual(UnsignedLessThanEqualTable),
            AluOp::Srl => Self::Srl(VirtualSRLTable),
            AluOp::Sra => Self::Sra(VirtualSRATable),
            AluOp::Rotr => Self::Rotr(VirtualROTRTable),
            AluOp::NegateIf => Self::NegateIf(VirtualNegateIfTable),
            AluOp::MulUNoOverflow => Self::MulUNoOverflow(MulUNoOverflowTable),
            AluOp::Pow2 => Self::Pow2(Pow2Table),
            AluOp::ShiftRightBitmask => Self::ShiftRightBitmask(ShiftRightBitmaskTable),
            AluOp::SignExtendWord => Self::SignExtendWord(SignExtendWordTable),
            AluOp::Clz => Self::Clz(ClzTable),
            AluOp::Ctz => Self::Ctz(CtzTable),
            AluOp::Popcnt => Self::Popcnt(PopcntTable),
        }
    }

    /// The variant's position in declaration order, suitable for array
    /// indexing (the append-only table id).
    #[inline]
    pub fn index(&self) -> usize {
        let this = std::mem::discriminant(self);
        <Self as strum::IntoEnumIterator>::iter()
            .position(|table| std::mem::discriminant(&table) == this)
            .unwrap_or(usize::MAX)
    }

    pub fn materialize_entry(&self, index: u128) -> u64 {
        dispatch!(self, t => t.materialize_entry(index))
    }

    pub fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: JoltField + FieldOps<C>,
    {
        dispatch!(self, t => t.evaluate_mle(r))
    }

    pub fn prefixes(&self) -> &'static [Prefixes] {
        dispatch!(self, t => PrefixSuffixDecomposition::prefixes(t))
    }

    pub fn suffixes(&self) -> &'static [Suffixes] {
        dispatch!(self, t => PrefixSuffixDecomposition::suffixes(t))
    }

    pub fn combine<F: JoltField>(
        &self,
        prefixes: &[PrefixEval<F>],
        suffixes: &[SuffixEval<F>],
    ) -> F {
        dispatch!(self, t => PrefixSuffixDecomposition::combine(t, prefixes, suffixes))
    }
}

/// The 128-bit lookup index of a row with instruction inputs `(left, right)`.
pub fn lookup_index(op: AluOp, left: u64, right: u64) -> u128 {
    let (x, y) = (u128::from(left), u128::from(right));
    match op.operand_mode() {
        OperandMode::Interleaved => interleave_bits(left, right),
        OperandMode::Add => x + y,
        OperandMode::Sub => x + (1u128 << 64) - y,
        OperandMode::Mul => x * y,
    }
}

/// `materialize_entry` of `op`'s table at the row's lookup index.
pub fn table_output(op: AluOp, left: u64, right: u64) -> u64 {
    WasmTable::of(op).materialize_entry(lookup_index(op, left, right))
}

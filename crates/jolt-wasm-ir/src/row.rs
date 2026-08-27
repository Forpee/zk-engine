//! Proof-row classification of the IR.
//!
//! Every [`Ir`] instruction is described by a [`RowSpec`]: which registers it
//! reads and writes, its immediate, its [`RowFlags`], and the [`AluOp`] whose
//! table produces its lookup output. The flags mirror Jolt's circuit and
//! instruction flags; `jolt-wasm-backend`'s `check_record` and
//! `jolt_r1cs::constraints::wasm` are the constraint-form restatements.
//! Differences from RV64: branch and jump targets are absolute IR pcs, jumps
//! write no link register, no operand is pc-relative, and `Halt` holds the
//! pc.
//!
//! Advice rows (`RowFlag::Advice`) mirror RV64's: the row's lookup is the
//! `RangeCheck` table over the raw index `rd`, so the prover-supplied value is
//! range-checked to 64 bits and written through `WriteLookupToRd`; the R1CS
//! leaves the right lookup operand of an advice row unconstrained.
//!
//! [`AluOp::evaluate`] is the one owner of instruction semantics.

use crate::ir::{AdviceHint, AluOp, Ir, Operand, OperandMode, Reg};
use crate::ops::Width;

/// One row flag: the WebAssembly proof row's circuit/instruction flags.
/// `LeftIs*`/`RightIs*` select the instruction inputs; `*Operands` select how
/// the lookup index is formed; the rest guard constraints. Declaration order
/// is the flag's bit and its column order in the bytecode table, the R1CS,
/// and the polynomial ids — append-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
#[repr(u8)]
pub enum RowFlag {
    LeftIsRs1,
    RightIsRs2,
    RightIsImm,
    /// Lookup index is `left + right`.
    AddOperands,
    /// Lookup index is `left - right + 2^64`.
    SubOperands,
    /// Lookup index is `left * right`.
    MulOperands,
    WriteLookupToRd,
    Load,
    Store,
    Jump,
    Branch,
    Assert,
    Halt,
    /// The row never completes: executing it traps.
    Trap,
    /// `rd` is prover-supplied and unconstrained by this row.
    Advice,
}

impl RowFlag {
    /// Every flag, in bit order.
    pub const ALL: [RowFlag; RowFlag::COUNT] = [
        RowFlag::LeftIsRs1,
        RowFlag::RightIsRs2,
        RowFlag::RightIsImm,
        RowFlag::AddOperands,
        RowFlag::SubOperands,
        RowFlag::MulOperands,
        RowFlag::WriteLookupToRd,
        RowFlag::Load,
        RowFlag::Store,
        RowFlag::Jump,
        RowFlag::Branch,
        RowFlag::Assert,
        RowFlag::Halt,
        RowFlag::Trap,
        RowFlag::Advice,
    ];
    /// Number of flags.
    pub const COUNT: usize = 15;

    #[inline]
    pub const fn bit(self) -> u32 {
        self as u32
    }

    #[inline]
    pub const fn mask(self) -> RowFlags {
        RowFlags(1 << self.bit())
    }
}

/// A set of [`RowFlag`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct RowFlags(u32);

impl RowFlags {
    pub const EMPTY: RowFlags = RowFlags(0);

    pub const fn union(self, other: RowFlags) -> RowFlags {
        RowFlags(self.0 | other.0)
    }

    /// Whether any flag of `flags` is set.
    #[inline]
    pub const fn intersects(self, flags: RowFlags) -> bool {
        self.0 & flags.0 != 0
    }

    #[inline]
    pub const fn has(self, flag: RowFlag) -> bool {
        self.0 & flag.mask().0 != 0
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Reconstruct a set from its bits (unknown bits are dropped).
    pub const fn from_bits(bits: u32) -> RowFlags {
        RowFlags(bits & ((1 << RowFlag::COUNT) - 1))
    }
}

impl std::ops::BitOr for RowFlags {
    type Output = RowFlags;
    fn bitor(self, rhs: RowFlags) -> RowFlags {
        self.union(rhs)
    }
}

impl std::ops::BitOr<RowFlag> for RowFlags {
    type Output = RowFlags;
    fn bitor(self, rhs: RowFlag) -> RowFlags {
        self.union(rhs.mask())
    }
}

impl std::ops::BitOr for RowFlag {
    type Output = RowFlags;
    fn bitor(self, rhs: RowFlag) -> RowFlags {
        self.mask().union(rhs.mask())
    }
}

impl std::ops::BitOr<RowFlags> for RowFlag {
    type Output = RowFlags;
    fn bitor(self, rhs: RowFlags) -> RowFlags {
        self.mask().union(rhs)
    }
}

impl From<RowFlag> for RowFlags {
    fn from(flag: RowFlag) -> RowFlags {
        flag.mask()
    }
}

/// What a row looks up: a catalog table, or prover advice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum Lookup {
    Table(AluOp),
    Advice(AdviceHint),
}

impl Lookup {
    /// The catalog op whose table the row reads: advice rows read
    /// `RangeCheck` (the `Add(W64)` table) at the raw index `rd`.
    pub fn table_op(self) -> AluOp {
        match self {
            Lookup::Table(op) => op,
            Lookup::Advice(_) => AluOp::Add(Width::W64),
        }
    }

    /// Whether the lookup index is the raw right lookup operand (the
    /// read-address-fingerprint leg applies): combined-operand tables and
    /// advice rows.
    pub fn is_raw_index(self) -> bool {
        match self {
            Lookup::Table(op) => op.operand_mode() != OperandMode::Interleaved,
            Lookup::Advice(_) => true,
        }
    }
}

/// The static row description of one instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowSpec {
    pub flags: RowFlags,
    pub rs1: Option<Reg>,
    pub rs2: Option<Reg>,
    pub rd: Option<Reg>,
    pub imm: u64,
    pub lookup: Option<Lookup>,
}

impl RowSpec {
    const EMPTY: RowSpec = RowSpec {
        flags: RowFlags::EMPTY,
        rs1: None,
        rs2: None,
        rd: None,
        imm: 0,
        lookup: None,
    };

    /// Left instruction input from the record's `rs1` read (WebAssembly rows
    /// have no pc-relative operands).
    pub fn left_input(&self, rs1_value: u64) -> u64 {
        if self.flags.has(RowFlag::LeftIsRs1) {
            rs1_value
        } else {
            0
        }
    }

    pub fn right_input(&self, rs2_value: u64) -> u64 {
        if self.flags.has(RowFlag::RightIsRs2) {
            rs2_value
        } else if self.flags.has(RowFlag::RightIsImm) {
            self.imm
        } else {
            0
        }
    }

    /// Whether the row's lookup index is its raw right lookup operand.
    pub fn raf_flag(&self) -> bool {
        self.lookup.is_some_and(Lookup::is_raw_index)
    }

    /// The lookup output for the given instruction inputs (advice rows take
    /// the recorded `rd`).
    pub fn output(&self, left: u64, right: u64, rd_write: u64) -> u64 {
        match self.lookup {
            Some(Lookup::Table(op)) => op.evaluate(left, right),
            Some(Lookup::Advice(_)) => rd_write,
            None => 0,
        }
    }
}

fn operand_flags(op: AluOp) -> RowFlags {
    match op.operand_mode() {
        OperandMode::Interleaved => RowFlags::EMPTY,
        OperandMode::Add => RowFlag::AddOperands.mask(),
        OperandMode::Sub => RowFlag::SubOperands.mask(),
        OperandMode::Mul => RowFlag::MulOperands.mask(),
    }
}

/// Right-operand flags, register, and immediate for an [`Operand`].
fn right_operand(rs2: Operand) -> (RowFlags, Option<Reg>, u64) {
    match rs2 {
        Operand::Reg(r) => (RowFlag::RightIsRs2.mask(), Some(r), 0),
        Operand::Imm(v) => (RowFlag::RightIsImm.mask(), None, v),
    }
}

/// Row classification of an instruction.
pub trait RowModel {
    fn row_spec(self) -> RowSpec;
}

impl RowModel for Ir {
    fn row_spec(self) -> RowSpec {
        use RowFlag as F;
        let base = RowSpec::EMPTY;
        let identity = AluOp::Add(Width::W64);
        match self {
            Ir::Nop => base,
            Ir::Halt => RowSpec {
                flags: F::Halt.mask(),
                ..base
            },
            Ir::Trap => RowSpec {
                flags: F::Trap.mask(),
                ..base
            },
            Ir::Alu { op, rd, rs1, rs2 } => {
                let (right, rs2, imm) = right_operand(rs2);
                RowSpec {
                    flags: F::LeftIsRs1 | right | F::WriteLookupToRd | operand_flags(op),
                    rs1: Some(rs1),
                    rs2,
                    rd: Some(rd),
                    imm,
                    lookup: Some(Lookup::Table(op)),
                }
            }
            Ir::Advice { hint, rd, rs1, rs2 } => RowSpec {
                flags: F::Advice | F::WriteLookupToRd,
                rs1: Some(rs1),
                rs2: Some(rs2),
                rd: Some(rd),
                lookup: Some(Lookup::Advice(hint)),
                ..base
            },
            Ir::Assert { op, rs1, rs2, .. } => {
                let (right, rs2, imm) = right_operand(rs2);
                RowSpec {
                    flags: F::LeftIsRs1 | right | F::Assert | operand_flags(op),
                    rs1: Some(rs1),
                    rs2,
                    rd: None,
                    imm,
                    lookup: Some(Lookup::Table(op)),
                }
            }
            Ir::Load {
                rd,
                base: rs,
                offset,
            } => RowSpec {
                flags: F::Load.mask(),
                rs1: Some(rs),
                rd: Some(rd),
                imm: offset as u64,
                ..base
            },
            Ir::Store {
                base: rs,
                value,
                offset,
            } => RowSpec {
                flags: F::Store.mask(),
                rs1: Some(rs),
                rs2: Some(value),
                imm: offset as u64,
                ..base
            },
            Ir::Jump { target } => RowSpec {
                flags: F::RightIsImm | F::AddOperands | F::Jump,
                imm: u64::from(target),
                lookup: Some(Lookup::Table(identity)),
                ..base
            },
            Ir::JumpReg { rs } => RowSpec {
                flags: F::LeftIsRs1 | F::AddOperands | F::Jump,
                rs1: Some(rs),
                lookup: Some(Lookup::Table(identity)),
                ..base
            },
            Ir::Branch {
                op,
                rs1,
                rs2,
                target,
            } => RowSpec {
                flags: F::LeftIsRs1 | F::RightIsRs2 | F::Branch | operand_flags(op),
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm: u64::from(target),
                lookup: Some(Lookup::Table(op)),
                ..base
            },
        }
    }
}

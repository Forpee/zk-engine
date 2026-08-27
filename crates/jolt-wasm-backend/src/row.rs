//! Proof-row classification of the IR.
//!
//! Every [`Ir`] instruction is described by a [`RowSpec`]: which registers it
//! reads and writes, its immediate, its [`RowFlags`], and the [`AluOp`] whose
//! table produces its lookup output. The flags and the constraints in
//! [`check_record`] mirror Jolt's uniform R1CS (`jolt-r1cs`'s
//! `constraints/rv64.rs`) with WebAssembly-specific adjustments: branch and
//! jump targets are absolute IR pcs, jumps write no link register, `Halt`
//! holds the pc, and `MemoryGrow` is its own row class.
//!
//! [`AluOp::evaluate`] is the one owner of instruction semantics; the
//! interpreter executes rows through it, `check_record` re-derives every
//! record column from it in constraint form, and `jolt-wasm-tables` checks
//! each table against it.

use jolt_wasm_ir::layout::{MEMORY_SIZE_ADDR, PAGE_SIZE};
use jolt_wasm_ir::{AdviceHint, AluOp, Ir, Operand, OperandMode, Pc, Reg, Width};

use crate::machine::{RamAccess, Record};

/// Row flags. `Left*`/`Right*` select the instruction inputs; `*Operands`
/// select how the lookup operands are combined; the rest guard constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct RowFlags(u32);

impl RowFlags {
    pub const LEFT_IS_RS1: RowFlags = RowFlags(1 << 0);
    pub const LEFT_IS_PC: RowFlags = RowFlags(1 << 1);
    pub const RIGHT_IS_RS2: RowFlags = RowFlags(1 << 2);
    pub const RIGHT_IS_IMM: RowFlags = RowFlags(1 << 3);
    /// Lookup index is `left + right`.
    pub const ADD_OPERANDS: RowFlags = RowFlags(1 << 4);
    /// Lookup index is `left - right + 2^64`.
    pub const SUB_OPERANDS: RowFlags = RowFlags(1 << 5);
    /// Lookup index is `left * right`.
    pub const MUL_OPERANDS: RowFlags = RowFlags(1 << 6);
    pub const WRITE_LOOKUP_TO_RD: RowFlags = RowFlags(1 << 7);
    pub const LOAD: RowFlags = RowFlags(1 << 8);
    pub const STORE: RowFlags = RowFlags(1 << 9);
    pub const JUMP: RowFlags = RowFlags(1 << 10);
    pub const BRANCH: RowFlags = RowFlags(1 << 11);
    pub const ASSERT: RowFlags = RowFlags(1 << 12);
    pub const HALT: RowFlags = RowFlags(1 << 13);
    pub const MEMORY_GROW: RowFlags = RowFlags(1 << 14);
    /// The row never completes: executing it traps.
    pub const TRAP: RowFlags = RowFlags(1 << 15);
    /// `rd` is prover-supplied and unconstrained by this row.
    pub const ADVICE: RowFlags = RowFlags(1 << 16);

    pub const fn union(self, other: RowFlags) -> RowFlags {
        RowFlags(self.0 | other.0)
    }

    #[inline]
    pub const fn has(self, flag: RowFlags) -> bool {
        self.0 & flag.0 != 0
    }

    pub const fn bits(self) -> u32 {
        self.0
    }
}

impl std::ops::BitOr for RowFlags {
    type Output = RowFlags;
    fn bitor(self, rhs: RowFlags) -> RowFlags {
        self.union(rhs)
    }
}

/// What a row looks up: a catalog table, or prover advice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lookup {
    Table(AluOp),
    Advice(AdviceHint),
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
        flags: RowFlags(0),
        rs1: None,
        rs2: None,
        rd: None,
        imm: 0,
        lookup: None,
    };

    /// Left instruction input from the record's register reads and pc.
    pub fn left_input(&self, rs1_value: u64, pc: Pc) -> u64 {
        if self.flags.has(RowFlags::LEFT_IS_RS1) {
            rs1_value
        } else if self.flags.has(RowFlags::LEFT_IS_PC) {
            u64::from(pc)
        } else {
            0
        }
    }

    pub fn right_input(&self, rs2_value: u64) -> u64 {
        if self.flags.has(RowFlags::RIGHT_IS_RS2) {
            rs2_value
        } else if self.flags.has(RowFlags::RIGHT_IS_IMM) {
            self.imm
        } else {
            0
        }
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
        OperandMode::Interleaved => RowFlags(0),
        OperandMode::Add => RowFlags::ADD_OPERANDS,
        OperandMode::Sub => RowFlags::SUB_OPERANDS,
        OperandMode::Mul => RowFlags::MUL_OPERANDS,
    }
}

/// Right-operand flags, register, and immediate for an [`Operand`].
fn right_operand(rs2: Operand) -> (RowFlags, Option<Reg>, u64) {
    match rs2 {
        Operand::Reg(r) => (RowFlags::RIGHT_IS_RS2, Some(r), 0),
        Operand::Imm(v) => (RowFlags::RIGHT_IS_IMM, None, v),
    }
}

/// Row classification of an instruction.
pub trait RowModel {
    fn row_spec(self) -> RowSpec;
}

impl RowModel for Ir {
    fn row_spec(self) -> RowSpec {
        use RowFlags as F;
        let base = RowSpec::EMPTY;
        let identity = AluOp::Add(Width::W64);
        match self {
            Ir::Nop => base,
            Ir::Halt => RowSpec {
                flags: F::HALT,
                ..base
            },
            Ir::Trap => RowSpec {
                flags: F::TRAP,
                ..base
            },
            Ir::Alu { op, rd, rs1, rs2 } => {
                let (right, rs2, imm) = right_operand(rs2);
                RowSpec {
                    flags: F::LEFT_IS_RS1 | right | F::WRITE_LOOKUP_TO_RD | operand_flags(op),
                    rs1: Some(rs1),
                    rs2,
                    rd: Some(rd),
                    imm,
                    lookup: Some(Lookup::Table(op)),
                }
            }
            Ir::Advice { hint, rd, rs1, rs2 } => RowSpec {
                flags: F::LEFT_IS_RS1 | F::RIGHT_IS_RS2 | F::ADVICE,
                rs1: Some(rs1),
                rs2: Some(rs2),
                rd: Some(rd),
                lookup: Some(Lookup::Advice(hint)),
                ..base
            },
            Ir::Assert { op, rs1, rs2, .. } => {
                let (right, rs2, imm) = right_operand(rs2);
                RowSpec {
                    flags: F::LEFT_IS_RS1 | right | F::ASSERT | operand_flags(op),
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
                flags: F::LOAD,
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
                flags: F::STORE,
                rs1: Some(rs),
                rs2: Some(value),
                imm: offset as u64,
                ..base
            },
            Ir::Jump { target } => RowSpec {
                flags: F::RIGHT_IS_IMM | F::ADD_OPERANDS | F::JUMP,
                imm: u64::from(target),
                lookup: Some(Lookup::Table(identity)),
                ..base
            },
            Ir::JumpReg { rs } => RowSpec {
                flags: F::LEFT_IS_RS1 | F::ADD_OPERANDS | F::JUMP,
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
                flags: F::LEFT_IS_RS1 | F::RIGHT_IS_RS2 | F::BRANCH | operand_flags(op),
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm: u64::from(target),
                lookup: Some(Lookup::Table(op)),
                ..base
            },
            Ir::MemoryGrow { rd, rs } => RowSpec {
                flags: F::LEFT_IS_RS1 | F::MEMORY_GROW,
                rs1: Some(rs),
                rd: Some(rd),
                imm: MEMORY_SIZE_ADDR,
                ..base
            },
        }
    }
}

/// A record column that disagrees with the row constraints.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RowViolation {
    #[error("pc {pc}: register reads {actual:?} do not match the row spec {expected:?}")]
    RegisterReads {
        pc: Pc,
        expected: (Option<Reg>, Option<Reg>),
        actual: (Option<Reg>, Option<Reg>),
    },
    #[error("pc {pc}: rd {actual:?} does not match the row spec {expected:?}")]
    RegisterWrite {
        pc: Pc,
        expected: Option<Reg>,
        actual: Option<Reg>,
    },
    #[error("pc {pc}: assert/branch row uses non-boolean {op:?}")]
    NotBoolean { pc: Pc, op: AluOp },
    #[error("pc {pc}: assert row has lookup output {output}")]
    Assert { pc: Pc, output: u64 },
    #[error("pc {pc}: rd write {actual:#x} != lookup output {expected:#x}")]
    RdWrite { pc: Pc, expected: u64, actual: u64 },
    #[error("pc {pc}: RAM access {ram:?} violates the row's memory contract")]
    Ram { pc: Pc, ram: RamAccess },
    #[error("pc {pc}: next pc {actual} != {expected}")]
    NextPc { pc: Pc, expected: Pc, actual: Pc },
    #[error("pc {pc}: jump target {0} is not a pc", pc = .1)]
    InvalidJump(u64, Pc),
    #[error("pc {pc}: memory.grow row is inconsistent")]
    MemoryGrow { pc: Pc },
}

/// Check one record against its row's constraints (the executable spec of
/// the WASM uniform R1CS). Returns the first violated constraint.
pub fn check_record(record: &Record) -> Result<(), RowViolation> {
    let pc = record.pc;
    let spec = record.instruction.row_spec();
    let flags = spec.flags;

    let reads = (
        record.rs1.map(|r| r.register),
        record.rs2.map(|r| r.register),
    );
    if reads != (spec.rs1, spec.rs2) {
        return Err(RowViolation::RegisterReads {
            pc,
            expected: (spec.rs1, spec.rs2),
            actual: reads,
        });
    }
    let rd = record.rd.map(|w| w.register);
    if rd != spec.rd {
        return Err(RowViolation::RegisterWrite {
            pc,
            expected: spec.rd,
            actual: rd,
        });
    }
    let rs1_value = record.rs1.map_or(0, |r| r.value);
    let rs2_value = record.rs2.map_or(0, |r| r.value);
    let rd_write = record.rd.map_or(0, |w| w.post_value);

    if let (true, Some(Lookup::Table(op))) =
        (flags.has(RowFlags::ASSERT | RowFlags::BRANCH), spec.lookup)
    {
        if !op.is_boolean() {
            return Err(RowViolation::NotBoolean { pc, op });
        }
    }
    let left = spec.left_input(rs1_value, pc);
    let right = spec.right_input(rs2_value);
    let output = spec.output(left, right, rd_write);

    if flags.has(RowFlags::ASSERT) && output != 1 {
        return Err(RowViolation::Assert { pc, output });
    }
    if flags.has(RowFlags::WRITE_LOOKUP_TO_RD) && rd_write != output {
        return Err(RowViolation::RdWrite {
            pc,
            expected: output,
            actual: rd_write,
        });
    }

    let ram_error = || RowViolation::Ram {
        pc,
        ram: record.ram,
    };
    let address = rs1_value.wrapping_add(spec.imm);
    match record.ram {
        RamAccess::Read(read) => {
            if !flags.has(RowFlags::LOAD) || read.address != address || read.value != rd_write {
                return Err(ram_error());
            }
        }
        RamAccess::Write(write) => {
            if flags.has(RowFlags::STORE) {
                if write.address != address || write.post_value != rs2_value {
                    return Err(ram_error());
                }
            } else if flags.has(RowFlags::MEMORY_GROW) {
                let grown = write.pre_value + rs1_value.wrapping_mul(PAGE_SIZE);
                let ok = write.address == MEMORY_SIZE_ADDR
                    && if rd_write == u64::from(u32::MAX) {
                        write.post_value == write.pre_value
                    } else {
                        write.post_value == grown && rd_write == write.pre_value / PAGE_SIZE
                    };
                if !ok {
                    return Err(RowViolation::MemoryGrow { pc });
                }
            } else {
                return Err(ram_error());
            }
        }
        RamAccess::NoOp => {
            if flags.has(RowFlags::LOAD | RowFlags::STORE | RowFlags::MEMORY_GROW) {
                return Err(ram_error());
            }
        }
    }

    let expected_next = if flags.has(RowFlags::HALT) {
        pc
    } else if flags.has(RowFlags::JUMP) {
        Pc::try_from(output).map_err(|_| RowViolation::InvalidJump(output, pc))?
    } else if flags.has(RowFlags::BRANCH) && output == 1 {
        spec.imm as Pc
    } else {
        pc + 1
    };
    if record.next_pc != expected_next {
        return Err(RowViolation::NextPc {
            pc,
            expected: expected_next,
            actual: record.next_pc,
        });
    }
    Ok(())
}

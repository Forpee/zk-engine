//! The constraint-form check of a record against its row (the executable
//! spec of the WASM uniform R1CS, `jolt_r1cs::constraints::wasm`).

use jolt_wasm_ir::row::{Lookup, RowFlag, RowModel};
use jolt_wasm_ir::{AluOp, Pc, Reg};

use crate::machine::{RamAccess, Record};

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

    if let (true, Some(Lookup::Table(op))) = (
        flags.intersects(RowFlag::Assert | RowFlag::Branch),
        spec.lookup,
    ) {
        if !op.is_boolean() {
            return Err(RowViolation::NotBoolean { pc, op });
        }
    }
    let left = spec.left_input(rs1_value);
    let right = spec.right_input(rs2_value);
    let output = spec.output(left, right, rd_write);

    if flags.has(RowFlag::Assert) && output != 1 {
        return Err(RowViolation::Assert { pc, output });
    }
    if flags.has(RowFlag::WriteLookupToRd) && rd_write != output {
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
            if !flags.has(RowFlag::Load) || read.address != address || read.value != rd_write {
                return Err(ram_error());
            }
        }
        RamAccess::Write(write) => {
            if !flags.has(RowFlag::Store)
                || write.address != address
                || write.post_value != rs2_value
            {
                return Err(ram_error());
            }
        }
        RamAccess::NoOp => {
            if flags.intersects(RowFlag::Load | RowFlag::Store) {
                return Err(ram_error());
            }
        }
    }

    let expected_next = if flags.has(RowFlag::Halt) {
        pc
    } else if flags.has(RowFlag::Jump) {
        Pc::try_from(output).map_err(|_| RowViolation::InvalidJump(output, pc))?
    } else if flags.has(RowFlag::Branch) && output == 1 {
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

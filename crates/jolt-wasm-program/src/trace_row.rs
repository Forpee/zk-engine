//! The compact, proof-facing trace row: one [`Record`] materialized as 64
//! bytes with the logical-column accessor API the witness and sumcheck code
//! consume (`rs1_value`, `ram_address`, `rd_write_value`, …), the WASM
//! analogue of RV64's `JoltTraceRow`.
//!
//! Physical storage aliases logical columns that are equal or mutually
//! exclusive for the row's class (from its flags):
//!
//! | class        | slot0 | slot1        | slot2          | slot3           |
//! |--------------|-------|--------------|----------------|-----------------|
//! | non-memory   | rs1   | rs2          | rd pre         | rd write        |
//! | load         | rs1   | ram address  | rd pre         | rd write (= ram read = ram write) |
//! | store        | rs1   | rs2 (= ram write) | ram read  | ram address     |
//!
//! The static half of the row (`imm`, flags, register and table ids) is
//! duplicated from the bytecode so hot loops need no table lookup;
//! [`WasmTraceRow::bytecode_row`] recovers it.

use jolt_wasm_backend::{RamAccess, Record};
use jolt_wasm_ir::row::{Lookup, RowFlag, RowFlags, RowModel, RowSpec};
use jolt_wasm_ir::{AluOp, BytecodeRow, Ir, OperandMode, Pc, Reg, REGISTER_NONE};
use jolt_wasm_tables::{lookup_index_for, WasmTable};

/// Compact, copyable proof-facing trace row (64 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct WasmTraceRow {
    slots: [u64; 4],
    imm: u64,
    pc: Pc,
    next_pc: Pc,
    flags: RowFlags,
    rs1: u8,
    rs2: u8,
    rd: u8,
    lookup: Option<Lookup>,
    _reserved: [u8; 4],
}

const _: () = assert!(
    std::mem::size_of::<WasmTraceRow>() == 64,
    "WasmTraceRow must stay 64 bytes; any size change should be intentional and reviewed"
);

/// A record whose values do not satisfy its row's contract.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TraceRowError {
    #[error("pc {pc}: register operands {actual:?} do not match the row spec {expected:?}")]
    RegisterOperands {
        pc: Pc,
        expected: (Option<Reg>, Option<Reg>, Option<Reg>),
        actual: (Option<Reg>, Option<Reg>, Option<Reg>),
    },
    #[error("pc {pc}: RAM access {ram:?} violates the row's memory contract: {detail}")]
    MemoryContract {
        pc: Pc,
        ram: RamAccess,
        detail: &'static str,
    },
}

impl Default for WasmTraceRow {
    /// The canonical no-op/padding row: `Halt` at pc 0.
    fn default() -> Self {
        Self::no_op()
    }
}

impl WasmTraceRow {
    /// The canonical no-op row: `Halt` at pc 0, a pc self-loop with no
    /// reads, writes, or RAM access.
    pub fn no_op() -> Self {
        Self::from_parts(&Ir::Halt.row_spec(), 0, 0, [0; 4])
    }

    fn from_parts(spec: &RowSpec, pc: Pc, next_pc: Pc, slots: [u64; 4]) -> Self {
        Self {
            slots,
            imm: spec.imm,
            pc,
            next_pc,
            flags: spec.flags,
            rs1: spec.rs1.map_or(REGISTER_NONE, Reg::id),
            rs2: spec.rs2.map_or(REGISTER_NONE, Reg::id),
            rd: spec.rd.map_or(REGISTER_NONE, Reg::id),
            lookup: spec.lookup,
            _reserved: [0; 4],
        }
    }

    /// Materialize a record, checking it against its row's contract.
    pub fn from_record(record: &Record) -> Result<Self, TraceRowError> {
        let pc = record.pc;
        let spec = record.instruction.row_spec();
        let operands = (
            record.rs1.map(|r| r.register),
            record.rs2.map(|r| r.register),
            record.rd.map(|w| w.register),
        );
        if operands != (spec.rs1, spec.rs2, spec.rd) {
            return Err(TraceRowError::RegisterOperands {
                pc,
                expected: (spec.rs1, spec.rs2, spec.rd),
                actual: operands,
            });
        }
        let contract = |detail| TraceRowError::MemoryContract {
            pc,
            ram: record.ram,
            detail,
        };
        let rs1 = record.rs1.map_or(0, |r| r.value);
        let rs2 = record.rs2.map_or(0, |r| r.value);
        let rd_pre = record.rd.map_or(0, |w| w.pre_value);
        let rd_write = record.rd.map_or(0, |w| w.post_value);
        let flags = spec.flags;
        let slots = if flags.has(RowFlag::Load) {
            let RamAccess::Read(read) = record.ram else {
                return Err(contract("load row without a RAM read"));
            };
            if read.value != rd_write {
                return Err(contract("load value must equal the rd write"));
            }
            [rs1, read.address, rd_pre, rd_write]
        } else if flags.has(RowFlag::Store) {
            let RamAccess::Write(write) = record.ram else {
                return Err(contract("store row without a RAM write"));
            };
            if write.post_value != rs2 {
                return Err(contract("store value must equal rs2"));
            }
            [rs1, rs2, write.pre_value, write.address]
        } else {
            if record.ram != RamAccess::NoOp {
                return Err(contract("non-memory row carries a RAM access"));
            }
            [rs1, rs2, rd_pre, rd_write]
        };
        Ok(Self::from_parts(&spec, pc, record.next_pc, slots))
    }

    #[inline(always)]
    pub fn flags(&self) -> RowFlags {
        self.flags
    }

    #[inline(always)]
    pub fn is_load(&self) -> bool {
        self.flags.has(RowFlag::Load)
    }

    #[inline(always)]
    pub fn is_store(&self) -> bool {
        self.flags.has(RowFlag::Store)
    }

    /// The padding/no-op class: `Halt`.
    #[inline(always)]
    pub fn is_noop(&self) -> bool {
        self.flags.has(RowFlag::Halt)
    }

    #[inline(always)]
    pub fn rs1_value(&self) -> u64 {
        self.slots[0]
    }

    #[inline(always)]
    pub fn rs2_value(&self) -> u64 {
        if self.is_load() {
            0
        } else {
            self.slots[1]
        }
    }

    #[inline(always)]
    pub fn rd_pre_value(&self) -> u64 {
        if self.is_store() {
            0
        } else {
            self.slots[2]
        }
    }

    #[inline(always)]
    pub fn rd_write_value(&self) -> u64 {
        if self.is_store() {
            0
        } else {
            self.slots[3]
        }
    }

    #[inline(always)]
    pub fn ram_address(&self) -> u64 {
        if self.is_load() {
            self.slots[1]
        } else if self.is_store() {
            self.slots[3]
        } else {
            0
        }
    }

    #[inline(always)]
    pub fn ram_read_value(&self) -> u64 {
        if self.is_load() {
            self.slots[3]
        } else if self.is_store() {
            self.slots[2]
        } else {
            0
        }
    }

    #[inline(always)]
    pub fn ram_write_value(&self) -> u64 {
        if self.is_load() {
            self.slots[3]
        } else if self.is_store() {
            self.slots[1]
        } else {
            0
        }
    }

    /// Bytecode index of the row.
    #[inline(always)]
    pub fn pc(&self) -> Pc {
        self.pc
    }

    #[inline(always)]
    pub fn next_pc(&self) -> Pc {
        self.next_pc
    }

    /// The raw immediate bits.
    #[inline(always)]
    pub fn imm(&self) -> u64 {
        self.imm
    }

    /// The immediate as the constraints see it (see
    /// [`BytecodeRow::imm_signed`]).
    #[inline]
    pub fn imm_signed(&self) -> i128 {
        self.bytecode_row().imm_signed()
    }

    #[inline(always)]
    pub fn rs1_index(&self) -> Option<Reg> {
        register(self.rs1)
    }

    #[inline(always)]
    pub fn rs2_index(&self) -> Option<Reg> {
        register(self.rs2)
    }

    #[inline(always)]
    pub fn rd_index(&self) -> Option<Reg> {
        register(self.rd)
    }

    /// What the row looks up.
    #[inline(always)]
    pub fn lookup(&self) -> Option<Lookup> {
        self.lookup
    }

    /// The catalog op whose table the row reads (advice rows read
    /// `RangeCheck` at the raw index `rd`).
    #[inline]
    pub fn table_op(&self) -> Option<AluOp> {
        self.lookup.map(Lookup::table_op)
    }

    /// Whether the row's lookup index is its raw right lookup operand.
    #[inline]
    pub fn raf_flag(&self) -> bool {
        self.lookup.is_some_and(Lookup::is_raw_index)
    }

    /// The catalog table id, if the row looks one up.
    #[inline]
    pub fn table(&self) -> Option<usize> {
        self.table_op().map(|op| WasmTable::of(op).index())
    }

    /// The static half of the row, as committed in the bytecode table.
    pub fn bytecode_row(&self) -> BytecodeRow {
        BytecodeRow {
            imm: self.imm,
            flags: self.flags,
            rs1: self.rs1,
            rs2: self.rs2,
            rd: self.rd,
            lookup: self.lookup,
        }
    }

    /// Left instruction input (`rs1` or 0 per the flags).
    #[inline]
    pub fn left_input(&self) -> u64 {
        if self.flags.has(RowFlag::LeftIsRs1) {
            self.rs1_value()
        } else {
            0
        }
    }

    /// Right instruction input (`rs2`, the immediate, or 0 per the flags).
    #[inline]
    pub fn right_input(&self) -> u64 {
        if self.flags.has(RowFlag::RightIsRs2) {
            self.rs2_value()
        } else if self.flags.has(RowFlag::RightIsImm) {
            self.imm
        } else {
            0
        }
    }

    /// How the instruction inputs form the lookup index.
    #[inline]
    pub fn operand_mode(&self) -> OperandMode {
        if self.flags.has(RowFlag::AddOperands) {
            OperandMode::Add
        } else if self.flags.has(RowFlag::SubOperands) {
            OperandMode::Sub
        } else if self.flags.has(RowFlag::MulOperands) {
            OperandMode::Mul
        } else {
            OperandMode::Interleaved
        }
    }

    /// The 128-bit lookup index, when the row looks a table up: the operands
    /// under the op's mode, or the raw advice value (`rd`) for advice rows.
    pub fn lookup_index(&self) -> Option<u128> {
        match self.lookup {
            Some(Lookup::Table(op)) => Some(lookup_index_for(
                op.operand_mode(),
                self.left_input(),
                self.right_input(),
            )),
            Some(Lookup::Advice(_)) => Some(u128::from(self.rd_write_value())),
            None => None,
        }
    }

    /// The `(left, right)` lookup operands the R1CS sees: the raw index as the
    /// right operand for combined-operand and advice rows.
    pub fn lookup_operands(&self) -> (u64, u128) {
        match self.lookup {
            Some(Lookup::Table(op)) if op.operand_mode() == OperandMode::Interleaved => {
                (self.left_input(), u128::from(self.right_input()))
            }
            Some(_) => (0, self.lookup_index().unwrap_or(0)),
            None => (self.left_input(), u128::from(self.right_input())),
        }
    }

    /// The lookup output: the table entry at the row's index (for advice rows
    /// `RangeCheck` of the advice, i.e. the advice itself).
    pub fn lookup_output(&self) -> u64 {
        match (self.table_op(), self.lookup_index()) {
            (Some(op), Some(index)) => WasmTable::of(op).materialize_entry(index),
            _ => 0,
        }
    }
}

fn register(id: u8) -> Option<Reg> {
    (id != REGISTER_NONE).then(|| Reg::from_id(id)).flatten()
}

/// Materialize the full proof-facing trace once from a record stream.
pub fn build_trace_rows(records: &[Record]) -> Result<Vec<WasmTraceRow>, TraceRowError> {
    records.iter().map(WasmTraceRow::from_record).collect()
}

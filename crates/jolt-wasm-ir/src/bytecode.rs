//! The committed bytecode table.
//!
//! One [`BytecodeRow`] per IR pc — the static half of a proof row: flags,
//! register ids, immediate, and lookup identity — padded with the `Halt` row
//! (pc 0, a pc self-loop with no writes: the canonical no-op) to a power of
//! two. A trace [`Record`](jolt_wasm_backend::Record) links to it by `pc`
//! directly; IR pcs are dense, so there is no expanded/unexpanded pc map.
//!
//! [`BytecodeColumn`] enumerates the per-pc columns the bytecode read-RAF
//! argument folds; [`WasmBytecode::column`] and [`WasmBytecode::column_values`]
//! are what the prover materializes and the verifier evaluates.

use std::collections::BTreeMap;

use crate::ir::{AluOp, Ir, IrProgram, Pc, Reg};
use crate::row::{Lookup, RowFlag, RowFlags, RowModel, RowSpec};

/// Sentinel for "no register" in the packed ids.
pub const REGISTER_NONE: u8 = 0xFF;

/// The static half of a proof row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct BytecodeRow {
    pub imm: u64,
    pub flags: RowFlags,
    /// `rs1`/`rs2`/`rd` register ids, or [`REGISTER_NONE`].
    pub rs1: u8,
    pub rs2: u8,
    pub rd: u8,
    /// What the row looks up, if anything. The catalog table id is
    /// `jolt_wasm_tables::WasmTable::of(op).index()`.
    pub lookup: Option<Lookup>,
}

impl BytecodeRow {
    /// The static row of an instruction.
    pub fn of(instruction: Ir) -> Self {
        Self::from_spec(&instruction.row_spec())
    }

    pub fn from_spec(spec: &RowSpec) -> Self {
        Self {
            imm: spec.imm,
            flags: spec.flags,
            rs1: register_id(spec.rs1),
            rs2: register_id(spec.rs2),
            rd: register_id(spec.rd),
            lookup: spec.lookup,
        }
    }

    /// The catalog op whose table the row reads (advice rows read
    /// `RangeCheck`).
    pub fn table_op(&self) -> Option<AluOp> {
        self.lookup.map(Lookup::table_op)
    }

    /// Whether the row's lookup index is its raw right lookup operand.
    pub fn raf_flag(&self) -> bool {
        self.lookup.is_some_and(Lookup::is_raw_index)
    }

    pub fn rs1(&self) -> Option<Reg> {
        register(self.rs1)
    }

    pub fn rs2(&self) -> Option<Reg> {
        register(self.rs2)
    }

    pub fn rd(&self) -> Option<Reg> {
        register(self.rd)
    }

    /// The immediate as the field element the constraints see: memory rows
    /// carry a signed byte offset (shadow-stack reloads use `SP − 8`), every
    /// other row an unsigned 64-bit operand or target.
    pub fn imm_signed(&self) -> i128 {
        if self.flags.intersects(RowFlag::Load | RowFlag::Store) {
            i128::from(self.imm as i64)
        } else {
            i128::from(self.imm)
        }
    }

    /// Canonical 16-byte little-endian encoding of the static columns; the
    /// lookup identity is appended by the committing layer as the table id.
    pub fn encode(&self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&self.imm.to_le_bytes());
        out[8..12].copy_from_slice(&self.flags.bits().to_le_bytes());
        out[12] = self.rs1;
        out[13] = self.rs2;
        out[14] = self.rd;
        out[15] = u8::from(self.lookup.is_some());
        out
    }
}

fn register_id(reg: Option<Reg>) -> u8 {
    reg.map_or(REGISTER_NONE, Reg::id)
}

fn register(id: u8) -> Option<Reg> {
    (id != REGISTER_NONE).then(|| Reg::from_id(id)).flatten()
}

/// A per-pc column of the bytecode table, as the read-RAF stages fold them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BytecodeColumn {
    /// The pc itself (the identity column; `UnexpandedPC` in RV64 terms).
    Pc,
    Imm,
    /// One row flag bit.
    Flag(RowFlag),
    /// Register ids as values (`REGISTER_NONE` → 0 with the matching flag
    /// off); one-hot register columns derive from these.
    Rs1,
    Rs2,
    Rd,
    /// `1` iff the row looks up this catalog op's table.
    TableFlag(AluOp),
    /// `1` iff the row's lookup index is its raw right operand (the
    /// read-address-fingerprint flag).
    RafFlag,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PreprocessingError {
    #[error("program must begin with the Halt trampoline at pc 0, found {0:?}")]
    MissingHaltTrampoline(Option<Ir>),
    #[error("pc {pc}: jump/branch target {target} is outside the {len}-row program")]
    TargetOutOfRange { pc: Pc, target: u64, len: usize },
    #[error("pc {pc}: branch/assert uses non-boolean {instruction:?}")]
    NotBoolean { pc: Pc, instruction: Ir },
    #[error("program has {0} rows; the bytecode index must fit u32")]
    ProgramTooLarge(usize),
    #[error("export `{name}` (function {function}) has no entry stub inside the program")]
    EntryOutOfRange { name: String, function: u32 },
}

/// The padded bytecode table.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct WasmBytecode {
    rows: Vec<BytecodeRow>,
    /// Number of real (unpadded) rows.
    len: usize,
    /// Exported function name → entry-stub pc.
    entries: BTreeMap<String, Pc>,
}

impl WasmBytecode {
    /// Validate and pad the program's code into the committed table.
    pub fn preprocess(program: &IrProgram) -> Result<Self, PreprocessingError> {
        let code = &program.code;
        let len = code.len();
        if code.first() != Some(&Ir::Halt) {
            return Err(PreprocessingError::MissingHaltTrampoline(
                code.first().copied(),
            ));
        }
        if Pc::try_from(len).is_err() {
            return Err(PreprocessingError::ProgramTooLarge(len));
        }
        let mut rows = Vec::with_capacity(len.next_power_of_two().max(2));
        for (pc, instruction) in code.iter().enumerate() {
            let pc = pc as Pc;
            let spec = instruction.row_spec();
            let target = match *instruction {
                Ir::Jump { target } | Ir::Branch { target, .. } => Some(u64::from(target)),
                _ => None,
            };
            if let Some(target) = target {
                if target >= len as u64 {
                    return Err(PreprocessingError::TargetOutOfRange { pc, target, len });
                }
            }
            if let (true, Some(Lookup::Table(op))) = (
                spec.flags.intersects(RowFlag::Branch | RowFlag::Assert),
                spec.lookup,
            ) {
                if !op.is_boolean() {
                    return Err(PreprocessingError::NotBoolean {
                        pc,
                        instruction: *instruction,
                    });
                }
            }
            rows.push(BytecodeRow::from_spec(&spec));
        }
        let halt = rows[0];
        let code_size = len.next_power_of_two().max(2);
        rows.resize(code_size, halt);

        let mut entries = BTreeMap::new();
        for (name, function) in &program.exports {
            let pc =
                *program
                    .entries
                    .get(name)
                    .ok_or_else(|| PreprocessingError::EntryOutOfRange {
                        name: name.clone(),
                        function: *function,
                    })?;
            if pc as usize >= len {
                return Err(PreprocessingError::EntryOutOfRange {
                    name: name.clone(),
                    function: *function,
                });
            }
            let _ = entries.insert(name.clone(), pc);
        }
        Ok(Self { rows, len, entries })
    }

    /// Padded size: a power of two, at least 2.
    pub fn code_size(&self) -> usize {
        self.rows.len()
    }

    /// Number of real rows (before padding).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn log_code_size(&self) -> usize {
        self.rows.len().trailing_zeros() as usize
    }

    pub fn rows(&self) -> &[BytecodeRow] {
        &self.rows
    }

    /// The row at `pc` (the padding row for pcs past the program).
    pub fn row(&self, pc: Pc) -> BytecodeRow {
        self.rows.get(pc as usize).copied().unwrap_or(self.rows[0])
    }

    /// Entry-stub pc of an exported function: where a trace of it begins.
    pub fn entry(&self, export: &str) -> Option<Pc> {
        self.entries.get(export).copied()
    }

    /// Every export's entry-stub pc.
    pub fn entries(&self) -> &BTreeMap<String, Pc> {
        &self.entries
    }

    /// The value of `column` at `pc` (signed: the `Imm` column of a memory
    /// row is its byte offset, see [`BytecodeRow::imm_signed`]).
    pub fn column(&self, pc: Pc, column: BytecodeColumn) -> i128 {
        let row = self.row(pc);
        match column {
            BytecodeColumn::Pc => i128::from(pc),
            BytecodeColumn::Imm => row.imm_signed(),
            BytecodeColumn::Flag(flag) => i128::from(row.flags.has(flag)),
            BytecodeColumn::Rs1 => register_value(row.rs1),
            BytecodeColumn::Rs2 => register_value(row.rs2),
            BytecodeColumn::Rd => register_value(row.rd),
            BytecodeColumn::TableFlag(op) => i128::from(row.table_op() == Some(op)),
            BytecodeColumn::RafFlag => i128::from(row.raf_flag()),
        }
    }

    /// The whole column over the padded table (the MLE's evaluations on the
    /// hypercube).
    pub fn column_values(&self, column: BytecodeColumn) -> Vec<i128> {
        (0..self.rows.len() as Pc)
            .map(|pc| self.column(pc, column))
            .collect()
    }

    /// Canonical encoding of the padded table's static columns: the bytes a
    /// program commitment or fingerprint is taken over (the committing layer
    /// appends each row's table id).
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.rows.len() * 16);
        for row in &self.rows {
            out.extend_from_slice(&row.encode());
        }
        out
    }
}

fn register_value(id: u8) -> i128 {
    if id == REGISTER_NONE {
        0
    } else {
        i128::from(id)
    }
}

//! Lowering from validated source operators to the register-machine IR: an
//! instruction selector onto the [`AluOp`] table catalog.
//!
//! Register assignment: function frame slot `i` (locals first, then the
//! operand stack) is `Reg::frame_slot(i)`. Structured control flow becomes
//! forward/backward jumps; values carried by a branch are moved down to the
//! target label's base height.
//!
//! Operand forwarding: `local.get` and `*.const` emit nothing — the stack
//! slot is marked *pending* ([`FunctionCtx::pending`]) and the consuming
//! operator reads the local's register, or takes the constant as its
//! immediate, directly. Pending slots are materialized (one `mov`/`const`)
//! before anything that observes the register file positionally: control
//! flow, calls, `return`, and a `local.set`/`tee` of the local they alias.
//! Symmetrically, an operator followed by `local.set`/`tee` writes the local
//! directly.
//!
//! Calling convention (static at the call site; `call_indirect` resolves the
//! callee's entry pc from the function table in guest RAM — see
//! [`Emitter::emit_indirect_target`] — and jumps through a register):
//! 1. spill the caller's live frame slots `0..live` to the shadow stack at
//!    `SP + 8*i`, then the return address at `SP + 8*live`;
//! 2. load the callee's parameters (frame slots `0..P`) from the spilled
//!    argument slots; bump `SP` by `8*(live+1)`; jump to the callee entry;
//! 3. the callee zeroes its declared locals on entry and on `return` moves its
//!    results into `T0..`, reloads `RA` from `SP - 8`, and jumps to it;
//! 4. back at the return pc the caller restores `SP`, reloads its live slots
//!    below the arguments, and moves the results into place.
//!
//! Expansions (operators with no table of their own):
//! - `shl` = `Pow2` then `Mul`; `shr_u`/`shr_s`/`rotr` = `ShiftRightBitmask`
//!   then `Srl`/`Sra`/`Rotr`; `rotl` = `rotr` by the negated count; 32-bit
//!   variants mask the count to 5 bits and canonicalize with
//!   `LowerHalfWord` where the 64-bit table would sign-fill;
//! - `div`/`rem` = `Advice` quotient pinned by `MulUNoOverflow`, `LeU`, and
//!   `LtU` asserts; signed variants go through `NegateIf` magnitudes;
//! - 32-bit signed compares sign-extend both operands first; `gt`/`le` swap
//!   operands of `lt`/`ge`;
//! - `extend8_s`/`extend16_s` = multiply up then `Sra` by a constant bitmask;
//! - `clz`/`ctz`/`popcnt` are tables (32-bit `clz` subtracts 32, 32-bit
//!   `ctz` sets bit 32 first);
//! - linear-memory accesses of any width/alignment (`layout::linear_address`:
//!   one RAM cell per 4-byte wasm word): effective guest address `t`, a
//!   one-row bounds assert `t < LIMIT_w` against the per-width limit
//!   registers (`Reg::limit`, maintained by the entry stub and
//!   `memory.grow`), floor to the cell, then a branch on the lane offset
//!   `a & 3`: an access inside one wasm word (the common, aligned case) is
//!   a plain cell read/write — shifted by `s = 8·(a & 3)` only for
//!   sub-word widths — and an `i64` access is two cells; an access that
//!   crosses a wasm word takes an out-of-line cold block spanning up to three
//!   cells. Cold blocks are placed after the function body so the hot path
//!   never jumps over them.

use jolt_wasm_ir::layout::{
    input_address, linear_address, output_address, table_slot_address, DEFAULT_MAX_PAGES,
    GLOBALS_BASE, LINEAR_CELL_BYTES, MAX_PAGES, MAX_TABLE_SLOTS, MEMORY_SIZE_ADDR, PAGE_SIZE,
    SHADOW_STACK_BASE, TABLE_SLOT_BYTES, TERMINATION_ADDR, WASM_WORD_BYTES, WORD_BYTES,
};
use jolt_wasm_ir::{
    shift_right_bitmask, AdviceHint, AluOp, AssertFailure, DataSegment, Ir, IrFunction, IrProgram,
    MemoryLimits, Operand, Pc, Reg, TableSlot, Width, MAX_FRAME_SLOTS, MAX_RESULTS,
};

use crate::error::LowerError;
use crate::module::{FuncType, Function, WasmModule};
use crate::source::{BinaryOp, BlockSig, ConvertOp, MemWidth, UnaryOp, ValType, WasmOp};

impl WasmModule {
    /// Lower the validated module to the register-machine IR.
    pub fn lower(&self) -> Result<IrProgram, LowerError> {
        lower(self)
    }
}

pub fn lower(module: &WasmModule) -> Result<IrProgram, LowerError> {
    let memory = match module.memory {
        Some(decl) => {
            let max_pages = decl.max_pages.unwrap_or(DEFAULT_MAX_PAGES).min(MAX_PAGES);
            if decl.initial_pages > max_pages {
                return Err(LowerError::MemoryTooLarge {
                    pages: decl.initial_pages,
                    max: max_pages,
                });
            }
            MemoryLimits {
                initial_pages: decl.initial_pages,
                max_pages,
            }
        }
        None => MemoryLimits {
            initial_pages: 0,
            max_pages: 0,
        },
    };

    let table_slots = module.table.map_or(0, |t| t.initial);
    if table_slots > MAX_TABLE_SLOTS {
        return Err(LowerError::TableTooLarge {
            slots: table_slots,
            max: MAX_TABLE_SLOTS,
        });
    }
    let type_ids = canonical_type_ids(&module.types);

    let mut signatures = Vec::with_capacity(module.functions.len());
    for (index, function) in module.functions.iter().enumerate() {
        let index = index as u32;
        let ty = module
            .func_type(index)
            .map_err(|_| LowerError::FunctionIndex(index))?;
        let id = *type_ids
            .get(function.type_index as usize)
            .ok_or(LowerError::FunctionIndex(index))?;
        let locals = ty.params.len() + function.locals.len();
        let slots = locals + function.max_height as usize;
        if slots > MAX_FRAME_SLOTS {
            return Err(LowerError::FrameTooLarge {
                function: index,
                slots,
                max: MAX_FRAME_SLOTS,
            });
        }
        if ty.results.len() > MAX_RESULTS {
            return Err(LowerError::TooManyResults {
                function: index,
                results: ty.results.len(),
                max: MAX_RESULTS,
            });
        }
        signatures.push(Signature {
            id,
            params: ty.params.len() as u32,
            results: ty.results.len() as u32,
            locals: locals as u32,
            frame_slots: slots,
        });
    }
    // The signature of a `call_indirect` type index that no function has:
    // params/results from the type, no frame.
    let type_signatures: Vec<Signature> = module
        .types
        .iter()
        .zip(&type_ids)
        .map(|(ty, id)| Signature {
            id: *id,
            params: ty.params.len() as u32,
            results: ty.results.len() as u32,
            locals: 0,
            frame_slots: 0,
        })
        .collect();

    let mut emitter = Emitter {
        code: vec![Ir::Halt],
        call_fixups: Vec::new(),
        cold: Vec::new(),
        signatures: &signatures,
        type_signatures: &type_signatures,
        table_slots,
        max_memory_bytes: memory.max_pages * PAGE_SIZE,
    };
    let mut functions = Vec::with_capacity(module.functions.len());
    for (index, function) in module.functions.iter().enumerate() {
        let index = index as u32;
        let entry = emitter.pc()?;
        emitter.lower_function(index, function)?;
        let sig = &signatures[index as usize];
        functions.push(IrFunction {
            entry,
            params: sig.params,
            results: sig.results,
            frame_slots: sig.frame_slots,
        });
    }
    let mut entries = std::collections::BTreeMap::new();
    for (name, function) in &module.exports {
        let pc = emitter.pc()?;
        emitter.emit_entry_stub(*function, module.start)?;
        let _ = entries.insert(name.clone(), pc);
    }
    for (pc, callee) in emitter.call_fixups {
        let entry = functions
            .get(callee as usize)
            .ok_or(LowerError::FunctionIndex(callee))?
            .entry;
        set_target(&mut emitter.code[pc as usize], entry);
    }
    let mut table = vec![None; table_slots as usize];
    for segment in &module.elements {
        for (i, function) in segment.functions.iter().enumerate() {
            let slot =
                table
                    .get_mut(segment.offset as usize + i)
                    .ok_or(LowerError::TableTooLarge {
                        slots: segment.offset + segment.functions.len() as u64,
                        max: table_slots,
                    })?;
            *slot = match *function {
                Some(f) => Some(TableSlot {
                    entry: functions
                        .get(f as usize)
                        .ok_or(LowerError::FunctionIndex(f))?
                        .entry,
                    signature: signatures[f as usize].id,
                }),
                None => None,
            };
        }
    }

    Ok(IrProgram {
        code: emitter.code,
        functions,
        exports: module.exports.clone(),
        entries,
        memory,
        globals: module.globals.iter().map(|g| g.init).collect(),
        data: module
            .data
            .iter()
            .map(|d| DataSegment {
                offset: d.offset,
                bytes: d.bytes.clone(),
            })
            .collect(),
        table,
    })
}

/// One id per structurally distinct function type (`call_indirect` checks
/// the callee's signature structurally, not by type index).
fn canonical_type_ids(types: &[FuncType]) -> Vec<u32> {
    let mut canonical: Vec<&FuncType> = Vec::new();
    types
        .iter()
        .map(|ty| {
            canonical.iter().position(|c| *c == ty).unwrap_or_else(|| {
                canonical.push(ty);
                canonical.len() - 1
            }) as u32
        })
        .collect()
}

/// A call target: a function index, or a table slot held in a register with
/// the expected signature.
#[derive(Debug, Clone, Copy)]
enum Callee {
    Direct(u32),
    Indirect { type_index: u32, index: Reg },
}

#[derive(Debug, Clone, Copy)]
struct Signature {
    /// Canonical function-type id (see [`canonical_type_ids`]).
    id: u32,
    params: u32,
    results: u32,
    /// Parameters plus declared locals.
    locals: u32,
    frame_slots: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LabelKind {
    Block,
    Loop {
        entry: Pc,
    },
    /// `if` whose `else`/`end` still needs to patch the entry branch.
    If {
        skip: Pc,
    },
    Else,
}

#[derive(Debug)]
struct Label {
    kind: LabelKind,
    /// Operand-stack height below the label's parameters/results.
    base: u32,
    sig: BlockSig,
    /// Forward jumps to patch to the label's end pc.
    fixups: Vec<Pc>,
}

impl Label {
    /// Values a branch to this label carries.
    fn arity(&self) -> u32 {
        match self.kind {
            LabelKind::Loop { .. } => self.sig.params,
            LabelKind::Block | LabelKind::If { .. } | LabelKind::Else => self.sig.results,
        }
    }
}

/// An out-of-line block for a rare case: the rows to emit after the function
/// body, the branch at `entry_fixup` that jumps to it, and the pc the block
/// jumps back to.
struct ColdBlock {
    entry_fixup: Pc,
    rows: Vec<Ir>,
    rejoin: Pc,
}

struct Emitter<'a> {
    code: Vec<Ir>,
    call_fixups: Vec<(Pc, u32)>,
    cold: Vec<ColdBlock>,
    signatures: &'a [Signature],
    /// Per type index: the signature a `call_indirect` expects.
    type_signatures: &'a [Signature],
    /// Slots of the function table (bounds of a `call_indirect` index).
    table_slots: u64,
    /// The page cap in bytes: `memory.grow` past it fails.
    max_memory_bytes: u64,
}

/// Per-function lowering state.
struct FunctionCtx {
    index: u32,
    sig: Signature,
    labels: Vec<Label>,
    /// `pending[depth]`: operand-stack slot `depth` holds an unmaterialized
    /// value (its register is stale).
    pending: Vec<Option<Pending>>,
}

/// A stack value not yet written to its slot's register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pending {
    /// A copy of the local.
    Local(u32),
    Const(u64),
}

impl FunctionCtx {
    /// Register for operand-stack depth `depth` (0-based from the bottom).
    fn stack(&self, depth: u32) -> Result<Reg, LowerError> {
        self.slot(self.sig.locals + depth)
    }

    fn take_pending(&mut self, depth: u32) -> Option<Pending> {
        self.pending.get_mut(depth as usize).and_then(Option::take)
    }

    fn set_pending(&mut self, depth: u32, value: Pending) {
        let depth = depth as usize;
        if self.pending.len() <= depth {
            self.pending.resize(depth + 1, None);
        }
        self.pending[depth] = Some(value);
    }

    fn slot(&self, slot: u32) -> Result<Reg, LowerError> {
        Reg::frame_slot(slot as usize).ok_or(LowerError::FrameTooLarge {
            function: self.index,
            slots: slot as usize + 1,
            max: MAX_FRAME_SLOTS,
        })
    }

    fn label(&self, depth: u32) -> Result<&Label, LowerError> {
        let n = self.labels.len();
        (depth as usize)
            .checked_add(1)
            .and_then(|d| n.checked_sub(d))
            .and_then(|i| self.labels.get(i))
            .ok_or(LowerError::LabelDepth(depth))
    }
}

const fn reg(r: Reg) -> Operand {
    Operand::Reg(r)
}

const fn imm(v: u64) -> Operand {
    Operand::Imm(v)
}

/// Word load at `base + offset` into `rd`.
const fn load(rd: Reg, base: Reg, offset: i64) -> Ir {
    Ir::Load { rd, base, offset }
}

/// Word store of `value` at `base + offset`.
const fn store(base: Reg, value: Reg, offset: i64) -> Ir {
    Ir::Store {
        base,
        value,
        offset,
    }
}

const fn add64(rd: Reg, rs1: Reg, rs2: Operand) -> Ir {
    Ir::alu(AluOp::Add(Width::W64), rd, rs1, rs2)
}

const fn mul64(rd: Reg, rs1: Reg, rs2: Operand) -> Ir {
    Ir::alu(AluOp::Mul(Width::W64), rd, rs1, rs2)
}

const fn assert(op: AluOp, failure: AssertFailure, rs1: Reg, rs2: Operand) -> Ir {
    Ir::Assert {
        op,
        failure,
        rs1,
        rs2,
    }
}

/// Guest bytes per wasm byte in linear memory (`layout::linear_address`).
const CELLS_PER_BYTE: u64 = LINEAR_CELL_BYTES / WASM_WORD_BYTES;
/// Mask selecting the lane offset `2·(a & 3)` of a guest linear address.
const LANE_MASK: u64 = LINEAR_CELL_BYTES - CELLS_PER_BYTE;
/// Bit shift per unit of lane offset: `s = 8·(a & 3) = 4 · (2·(a & 3))`.
const SHIFT_PER_LANE: u64 = 8 / CELLS_PER_BYTE;
const WASM_WORD_MASK: u64 = (1 << (8 * WASM_WORD_BYTES)) - 1;

/// The bounds-limit register of a `width`-byte access.
fn limit(width: MemWidth) -> Reg {
    match width {
        MemWidth::B1 => Reg::LIMIT_B,
        MemWidth::B2 => Reg::LIMIT_H,
        MemWidth::B4 => Reg::LIMIT_W,
        MemWidth::B8 => Reg::LIMIT_D,
    }
}

/// `rd` = the low `width` bytes of `rs`, extended to `result` bits.
fn narrow_rows(rd: Reg, rs: Reg, width: MemWidth, signed: bool, result: Width) -> Vec<Ir> {
    match (width, signed) {
        (MemWidth::B8, _) => vec![Ir::mov(rd, rs)],
        (_, false) => vec![Ir::alu_imm(AluOp::And, rd, rs, width.mask())],
        (MemWidth::B1, true) => sign_extend_rows(result, rd, rs, 8),
        (MemWidth::B2, true) => sign_extend_rows(result, rd, rs, 16),
        (MemWidth::B4, true) => match result {
            Width::W64 => vec![Ir::alu_imm(AluOp::SignExtendWord, rd, rs, 0)],
            Width::W32 => vec![Ir::alu_imm(AluOp::And, rd, rs, width.mask())],
        },
    }
}

/// Sign-extend the low `bits` (8 or 16) of `rs` to `width`, canonical.
fn sign_extend_rows(width: Width, rd: Reg, rs: Reg, bits: u64) -> Vec<Ir> {
    let up = 64 - bits;
    let (target, canonicalize) = match width {
        Width::W64 => (rd, false),
        Width::W32 => (Reg::T0, true),
    };
    let mut rows = vec![
        mul64(target, rs, imm(1 << up)),
        Ir::alu_imm(AluOp::Sra, target, target, shift_right_bitmask(up)),
    ];
    if canonicalize {
        rows.push(Ir::alu_imm(AluOp::LowerHalfWord, rd, target, 0));
    }
    rows
}

/// `(i32::MIN, -1)` at `width`, the signed-division overflow pair.
const fn signed_overflow_pair(width: Width) -> (u64, u64) {
    match width {
        Width::W32 => (0x8000_0000, 0xFFFF_FFFF),
        Width::W64 => (1 << 63, u64::MAX),
    }
}

impl Emitter<'_> {
    fn pc(&self) -> Result<Pc, LowerError> {
        Pc::try_from(self.code.len()).map_err(|_| LowerError::ProgramTooLarge(Pc::BITS))
    }

    fn emit(&mut self, ir: Ir) -> Result<Pc, LowerError> {
        let pc = self.pc()?;
        self.code.push(ir);
        Ok(pc)
    }

    fn emit_all(&mut self, irs: impl IntoIterator<Item = Ir>) -> Result<(), LowerError> {
        for ir in irs {
            let _ = self.emit(ir)?;
        }
        Ok(())
    }

    fn patch(&mut self, pc: Pc, target: Pc) {
        set_target(&mut self.code[pc as usize], target);
    }

    /// Write a pending value into its slot's register.
    fn materialize(
        &mut self,
        ctx: &FunctionCtx,
        depth: u32,
        value: Pending,
    ) -> Result<(), LowerError> {
        let rd = ctx.stack(depth)?;
        let _ = self.emit(match value {
            Pending::Local(local) => Ir::mov(rd, ctx.slot(local)?),
            Pending::Const(imm) => Ir::const_(rd, imm),
        })?;
        Ok(())
    }

    /// Materialize every pending slot.
    fn materialize_all(&mut self, ctx: &mut FunctionCtx) -> Result<(), LowerError> {
        for depth in 0..ctx.pending.len() {
            if let Some(value) = ctx.pending[depth].take() {
                self.materialize(ctx, depth as u32, value)?;
            }
        }
        Ok(())
    }

    /// Materialize the pending slots aliasing `local` (about to be written).
    fn materialize_local(&mut self, ctx: &mut FunctionCtx, local: u32) -> Result<(), LowerError> {
        for depth in 0..ctx.pending.len() {
            if ctx.pending[depth] == Some(Pending::Local(local)) {
                ctx.pending[depth] = None;
                self.materialize(ctx, depth as u32, Pending::Local(local))?;
            }
        }
        Ok(())
    }

    /// The register holding the value at stack depth `depth`: the aliased
    /// local for a pending copy, the slot itself otherwise (a pending
    /// constant is written to it first). Consuming a value clears its
    /// pending mark (the slot is popped or overwritten).
    fn src(&mut self, ctx: &mut FunctionCtx, depth: u32) -> Result<Reg, LowerError> {
        match ctx.take_pending(depth) {
            Some(Pending::Local(local)) => ctx.slot(local),
            Some(value @ Pending::Const(_)) => {
                self.materialize(ctx, depth, value)?;
                ctx.stack(depth)
            }
            None => ctx.stack(depth),
        }
    }

    /// Like [`Self::src`], but a pending constant becomes an immediate.
    fn src_operand(ctx: &mut FunctionCtx, depth: u32) -> Result<Operand, LowerError> {
        match ctx.take_pending(depth) {
            Some(Pending::Const(imm)) => Ok(Operand::Imm(imm)),
            Some(Pending::Local(local)) => ctx.slot(local).map(Operand::Reg),
            None => ctx.stack(depth).map(Operand::Reg),
        }
    }

    /// The destination register of an operator producing the value at stack
    /// depth `depth`: the local itself when the next operator is a
    /// `local.set`/`tee` of it (which is then skipped), else the slot.
    fn dest(
        &mut self,
        ctx: &mut FunctionCtx,
        depth: u32,
        next: Option<&WasmOp>,
    ) -> Result<(Reg, bool), LowerError> {
        match next {
            Some(&(WasmOp::LocalSet(local) | WasmOp::LocalTee(local))) => {
                self.materialize_local(ctx, local)?;
                if matches!(next, Some(WasmOp::LocalTee(_))) {
                    ctx.set_pending(depth, Pending::Local(local));
                }
                Ok((ctx.slot(local)?, true))
            }
            _ => Ok((ctx.stack(depth)?, false)),
        }
    }

    fn lower_function(&mut self, index: u32, function: &Function) -> Result<(), LowerError> {
        let sig = self.signatures[index as usize];
        let mut ctx = FunctionCtx {
            index,
            sig,
            labels: vec![Label {
                kind: LabelKind::Block,
                base: 0,
                sig: BlockSig {
                    params: 0,
                    results: sig.results,
                },
                fixups: Vec::new(),
            }],
            pending: Vec::new(),
        };
        for slot in sig.params..sig.locals {
            let rd = ctx.slot(slot)?;
            let _ = self.emit(Ir::mov(rd, Reg::ZERO))?;
        }
        let mut skip = false;
        for (i, source) in function.body.iter().enumerate() {
            if std::mem::take(&mut skip) {
                continue;
            }
            let next = function.body.get(i + 1).map(|s| &s.op);
            skip = self.lower_op(&mut ctx, &source.op, source.height, next)?;
        }
        if !ctx.labels.is_empty() {
            return Err(LowerError::MalformedControl(index));
        }
        self.flush_cold()
    }

    /// Place the pending cold blocks here (after a function body), patching
    /// their entry branches and rejoin jumps.
    fn flush_cold(&mut self) -> Result<(), LowerError> {
        for block in std::mem::take(&mut self.cold) {
            let start = self.pc()?;
            self.patch(block.entry_fixup, start);
            self.emit_all(block.rows)?;
            let _ = self.emit(Ir::Jump {
                target: block.rejoin,
            })?;
        }
        Ok(())
    }

    /// Register a cold block entered from `entry_fixup` that rejoins at the
    /// current pc.
    fn cold(&mut self, entry_fixup: Pc, rows: Vec<Ir>) -> Result<(), LowerError> {
        let rejoin = self.pc()?;
        self.cold.push(ColdBlock {
            entry_fixup,
            rows,
            rejoin,
        });
        Ok(())
    }

    /// Move the top `n` stack values down so they start at depth `base`.
    fn move_down(
        &mut self,
        ctx: &FunctionCtx,
        height: u32,
        base: u32,
        n: u32,
    ) -> Result<(), LowerError> {
        let src = height - n;
        if src == base {
            return Ok(());
        }
        for k in 0..n {
            let rd = ctx.stack(base + k)?;
            let rs = ctx.stack(src + k)?;
            let _ = self.emit(Ir::mov(rd, rs))?;
        }
        Ok(())
    }

    fn needs_moves(ctx: &FunctionCtx, height: u32, depth: u32) -> Result<bool, LowerError> {
        let label = ctx.label(depth)?;
        let n = label.arity();
        Ok(n > 0 && height - n != label.base)
    }

    /// Unconditional branch to label `depth` with the stack at `height`.
    fn branch(&mut self, ctx: &mut FunctionCtx, height: u32, depth: u32) -> Result<(), LowerError> {
        let label = ctx.label(depth)?;
        let (kind, base, n) = (label.kind, label.base, label.arity());
        let is_function_label = depth as usize + 1 == ctx.labels.len();
        if is_function_label {
            return self.emit_return(ctx, height);
        }
        self.move_down(ctx, height, base, n)?;
        match kind {
            LabelKind::Loop { entry } => {
                let _ = self.emit(Ir::Jump { target: entry })?;
            }
            LabelKind::Block | LabelKind::If { .. } | LabelKind::Else => {
                let pc = self.emit(Ir::Jump { target: 0 })?;
                let i = ctx.labels.len() - 1 - depth as usize;
                ctx.labels[i].fixups.push(pc);
            }
        }
        Ok(())
    }

    fn emit_return(&mut self, ctx: &FunctionCtx, height: u32) -> Result<(), LowerError> {
        let results = ctx.sig.results;
        for k in 0..results {
            let rs = ctx.stack(height - results + k)?;
            let rd = Reg::temp(k as usize).ok_or(LowerError::TooManyResults {
                function: ctx.index,
                results: results as usize,
                max: MAX_RESULTS,
            })?;
            let _ = self.emit(Ir::mov(rd, rs))?;
        }
        self.emit_all([load(Reg::RA, Reg::SP, -8), Ir::JumpReg { rs: Reg::RA }])
    }

    fn emit_call(
        &mut self,
        ctx: &FunctionCtx,
        height: u32,
        callee: Callee,
    ) -> Result<(), LowerError> {
        let callee_sig = match callee {
            Callee::Direct(function) => *self
                .signatures
                .get(function as usize)
                .ok_or(LowerError::FunctionIndex(function))?,
            Callee::Indirect { type_index, .. } => *self
                .type_signatures
                .get(type_index as usize)
                .ok_or(LowerError::FunctionIndex(type_index))?,
        };
        let (params, results) = (callee_sig.params, callee_sig.results);
        let live = ctx.sig.locals + height;
        let frame_bytes = u64::from(live + 1) * 8;

        for slot in 0..live {
            let value = ctx.slot(slot)?;
            let _ = self.emit(store(Reg::SP, value, i64::from(slot) * 8))?;
        }
        let ra_pc = self.emit(Ir::const_(Reg::RA, 0))?;
        let _ = self.emit(store(Reg::SP, Reg::RA, i64::from(live) * 8))?;
        let args_base = live - params;
        for j in 0..params {
            let rd = ctx.slot(j)?;
            let _ = self.emit(load(rd, Reg::SP, i64::from(args_base + j) * 8))?;
        }
        let _ = self.emit(add64(Reg::SP, Reg::SP, imm(frame_bytes)))?;
        match callee {
            Callee::Direct(function) => {
                let jump_pc = self.emit(Ir::Jump { target: 0 })?;
                self.call_fixups.push((jump_pc, function));
            }
            Callee::Indirect { index, .. } => {
                // `RA` is free again: its value is spilled, and the callee
                // reloads it from the shadow stack on return. The index
                // register is an operand-stack slot above the arguments, so
                // the parameter loads did not touch it.
                self.emit_indirect_target(index, callee_sig.id)?;
                let _ = self.emit(Ir::JumpReg { rs: Reg::RA })?;
            }
        }

        let return_pc = self.pc()?;
        self.patch(ra_pc, return_pc);
        let _ = self.emit(add64(Reg::SP, Reg::SP, imm(frame_bytes.wrapping_neg())))?;
        for slot in 0..args_base {
            let rd = ctx.slot(slot)?;
            let _ = self.emit(load(rd, Reg::SP, i64::from(slot) * 8))?;
        }
        for k in 0..results {
            let rd = ctx.slot(args_base + k)?;
            let rs = Reg::temp(k as usize).ok_or(LowerError::TooManyResults {
                function: ctx.index,
                results: results as usize,
                max: MAX_RESULTS,
            })?;
            let _ = self.emit(Ir::mov(rd, rs))?;
        }
        Ok(())
    }

    /// `RA` = the entry pc of function-table slot `index`, trapping when the
    /// index is past the table or the slot's signature word is not
    /// `signature + 1` (null slots hold `0`). Uses `T0` as scratch.
    fn emit_indirect_target(&mut self, index: Reg, signature: u32) -> Result<(), LowerError> {
        self.emit_all([
            assert(
                AluOp::LtU,
                AssertFailure::TableOutOfBounds,
                index,
                imm(self.table_slots),
            ),
            mul64(Reg::RA, index, imm(TABLE_SLOT_BYTES)),
            load(
                Reg::T0,
                Reg::RA,
                (table_slot_address(0) + WORD_BYTES) as i64,
            ),
            assert(
                AluOp::Eq,
                AssertFailure::IndirectCallTypeMismatch,
                Reg::T0,
                imm(u64::from(signature) + 1),
            ),
            load(Reg::RA, Reg::RA, table_slot_address(0) as i64),
        ])
    }

    /// The entry stub of exported `function`: from an all-zero register file,
    /// set `SP`, run `start`, load the parameters from the public input words,
    /// call, store the results to the public output words, set the
    /// termination word, and halt.
    fn emit_entry_stub(&mut self, function: u32, start: Option<u32>) -> Result<(), LowerError> {
        let sig = *self
            .signatures
            .get(function as usize)
            .ok_or(LowerError::FunctionIndex(function))?;
        // The stub is a frame with no locals; its operand stack holds the
        // arguments, then the results.
        let ctx = FunctionCtx {
            index: function,
            sig: Signature {
                id: sig.id,
                params: 0,
                results: 0,
                locals: 0,
                frame_slots: sig.params.max(sig.results) as usize,
            },
            labels: Vec::new(),
            pending: Vec::new(),
        };
        let _ = self.emit(Ir::const_(Reg::SP, SHADOW_STACK_BASE + 8))?;
        let _ = self.emit(load(Reg::T0, Reg::ZERO, MEMORY_SIZE_ADDR as i64))?;
        self.emit_limits(Reg::T0)?;
        if let Some(start) = start {
            self.emit_call(&ctx, 0, Callee::Direct(start))?;
        }
        for i in 0..sig.params {
            let rd = ctx.slot(i)?;
            let _ = self.emit(load(rd, Reg::ZERO, input_address(u64::from(i)) as i64))?;
        }
        self.emit_call(&ctx, sig.params, Callee::Direct(function))?;
        for k in 0..sig.results {
            let value = ctx.slot(k)?;
            let _ = self.emit(store(Reg::ZERO, value, output_address(u64::from(k)) as i64))?;
        }
        self.emit_all([
            Ir::const_(Reg::T0, 1),
            store(Reg::ZERO, Reg::T0, TERMINATION_ADDR as i64),
            Ir::Jump {
                target: IrProgram::HALT_PC,
            },
        ])
    }

    /// `memory.grow` by `rs` pages into `rd`: the old size in pages, or
    /// `u32::MAX` when the new size would exceed the page cap (in which case
    /// the size word is unchanged).
    fn emit_memory_grow(&mut self, rd: Reg, rs: Reg) -> Result<(), LowerError> {
        self.emit_all([
            load(Reg::T0, Reg::ZERO, MEMORY_SIZE_ADDR as i64),
            mul64(Reg::T1, rs, imm(PAGE_SIZE)),
            add64(Reg::T1, Reg::T0, reg(Reg::T1)),
            Ir::alu_imm(AluOp::LeU, Reg::T2, Reg::T1, self.max_memory_bytes),
        ])?;
        let fail = self.emit(Ir::branch_if_zero(Reg::T2, 0))?;
        let _ = self.emit(store(Reg::ZERO, Reg::T1, MEMORY_SIZE_ADDR as i64))?;
        self.emit_limits(Reg::T1)?;
        let _ = self.emit(Ir::alu_imm(
            AluOp::Srl,
            rd,
            Reg::T0,
            shift_right_bitmask(16),
        ))?;
        let done = self.emit(Ir::Jump { target: 0 })?;
        let fail_pc = self.pc()?;
        self.patch(fail, fail_pc);
        let _ = self.emit(Ir::const_(rd, u64::from(u32::MAX)))?;
        let end = self.pc()?;
        self.patch(done, end);
        Ok(())
    }

    /// Set the bounds-limit registers from the memory size in bytes held in
    /// `size`: `LIMIT_w = linear_address(size) − 2·w + 1`. Uses `T2`.
    fn emit_limits(&mut self, size: Reg) -> Result<(), LowerError> {
        let _ = self.emit(mul64(Reg::T2, size, imm(CELLS_PER_BYTE)))?;
        for width in [MemWidth::B1, MemWidth::B2, MemWidth::B4, MemWidth::B8] {
            let bytes = u64::from(width.bytes());
            let _ = self.emit(add64(
                limit(width),
                Reg::T2,
                imm(linear_address(0)
                    .wrapping_sub(CELLS_PER_BYTE * bytes)
                    .wrapping_add(1)),
            ))?;
        }
        Ok(())
    }

    /// Effective guest address `T0 = linear_address(addr + offset)`, the
    /// bounds assert, then `T1` = the containing cell's address and `T2` =
    /// `2·(a & 3)`, the lane offset (in guest bytes) within it.
    fn emit_address(&mut self, addr: Reg, offset: u64, width: MemWidth) -> Result<(), LowerError> {
        self.emit_all([
            add64(Reg::T0, addr, reg(addr)),
            add64(Reg::T0, Reg::T0, imm(linear_address(offset))),
            assert(
                AluOp::LtU,
                AssertFailure::OutOfBounds(width.bytes()),
                Reg::T0,
                reg(limit(width)),
            ),
            Ir::alu_imm(AluOp::And, Reg::T2, Reg::T0, LANE_MASK),
            Ir::alu_imm(AluOp::And, Reg::T1, Reg::T0, !(LINEAR_CELL_BYTES - 1)),
        ])
    }

    /// Branch to a cold block iff the access at lane offset `T2` of `width`
    /// bytes crosses its wasm word (never for a byte). Returns the branch pc,
    /// which [`Self::cold`] patches. `T4` is scratch.
    fn emit_crossing_branch(&mut self, width: MemWidth) -> Result<Option<Pc>, LowerError> {
        match width {
            MemWidth::B1 => Ok(None),
            MemWidth::B4 | MemWidth::B8 => self.emit(Ir::branch_if_nonzero(Reg::T2, 0)).map(Some),
            MemWidth::B2 => {
                // Crosses iff `a & 3 == 3`, i.e. `T2 == 6`.
                let _ = self.emit(Ir::const_(Reg::T4, 5))?;
                self.emit(Ir::Branch {
                    op: AluOp::LtU,
                    rs1: Reg::T4,
                    rs2: Reg::T2,
                    target: 0,
                })
                .map(Some)
            }
        }
    }

    /// Load `width` bytes at `addr + offset` into `rd`, sign- or
    /// zero-extended to `result`.
    fn emit_load(
        &mut self,
        rd: Reg,
        addr: Reg,
        offset: u64,
        width: MemWidth,
        signed: bool,
        result: Width,
    ) -> Result<(), LowerError> {
        self.emit_address(addr, offset, width)?;
        let crossing = self.emit_crossing_branch(width)?;
        // Hot: the lane is inside one wasm word (one cell), or an aligned
        // `i64` (two cells).
        match width {
            MemWidth::B8 => self.emit_all([
                load(Reg::T3, Reg::T1, 0),
                load(Reg::T4, Reg::T1, LINEAR_CELL_BYTES as i64),
                mul64(Reg::T4, Reg::T4, imm(1 << 32)),
                Ir::alu(AluOp::Or, rd, Reg::T3, reg(Reg::T4)),
            ])?,
            MemWidth::B4 if !signed => {
                let _ = self.emit(load(rd, Reg::T1, 0))?;
            }
            MemWidth::B4 => {
                let _ = self.emit(load(Reg::T3, Reg::T1, 0))?;
                self.emit_narrow(rd, Reg::T3, width, signed, result)?;
            }
            MemWidth::B1 | MemWidth::B2 => {
                self.emit_all([
                    load(Reg::T3, Reg::T1, 0),
                    mul64(Reg::T2, Reg::T2, imm(SHIFT_PER_LANE)),
                    Ir::alu_imm(AluOp::ShiftRightBitmask, Reg::T4, Reg::T2, 0),
                    Ir::alu(AluOp::Srl, Reg::T3, Reg::T3, reg(Reg::T4)),
                ])?;
                self.emit_narrow(rd, Reg::T3, width, signed, result)?;
            }
        }
        // Cold: the lane crosses into the next wasm word(s): little-endian
        // value `w0 >> s | w1 << (32 − s) [| w2 << (64 − s)]`.
        if let Some(crossing) = crossing {
            let mut cold = vec![
                mul64(Reg::T2, Reg::T2, imm(SHIFT_PER_LANE)),
                load(Reg::T3, Reg::T1, 0),
                Ir::alu_imm(AluOp::ShiftRightBitmask, Reg::T4, Reg::T2, 0),
                Ir::alu(AluOp::Srl, Reg::T3, Reg::T3, reg(Reg::T4)),
                load(Reg::T4, Reg::T1, LINEAR_CELL_BYTES as i64),
                Ir::const_(Reg::T0, 32),
                Ir::alu(AluOp::Sub(Width::W64), Reg::T0, Reg::T0, reg(Reg::T2)),
                Ir::alu_imm(AluOp::Pow2, Reg::T0, Reg::T0, 0),
                mul64(Reg::T4, Reg::T4, reg(Reg::T0)),
                Ir::alu(AluOp::Or, Reg::T3, Reg::T3, reg(Reg::T4)),
            ];
            if width == MemWidth::B8 {
                cold.extend([
                    load(Reg::T4, Reg::T1, 2 * LINEAR_CELL_BYTES as i64),
                    mul64(Reg::T4, Reg::T4, reg(Reg::T0)),
                    mul64(Reg::T4, Reg::T4, imm(1 << 32)),
                    Ir::alu(AluOp::Or, Reg::T3, Reg::T3, reg(Reg::T4)),
                ]);
            }
            cold.extend(narrow_rows(rd, Reg::T3, width, signed, result));
            self.cold(crossing, cold)?;
        }
        Ok(())
    }

    /// `rd` = the low `width` bytes of `rs`, extended to `result` bits.
    fn emit_narrow(
        &mut self,
        rd: Reg,
        rs: Reg,
        width: MemWidth,
        signed: bool,
        result: Width,
    ) -> Result<(), LowerError> {
        self.emit_all(narrow_rows(rd, rs, width, signed, result))
    }

    /// Sign-extend the low `bits` (8 or 16) of `rs` to `width`, canonical.
    fn emit_sign_extend(
        &mut self,
        width: Width,
        rd: Reg,
        rs: Reg,
        bits: u64,
    ) -> Result<(), LowerError> {
        self.emit_all(sign_extend_rows(width, rd, rs, bits))
    }

    /// Store the low `width` bytes of `value` (an `i32`, or an `i64` when
    /// `wide`) at `base + offset`.
    fn emit_store(
        &mut self,
        base: Reg,
        value: Reg,
        wide: bool,
        offset: u64,
        width: MemWidth,
    ) -> Result<(), LowerError> {
        self.emit_address(base, offset, width)?;
        let crossing = self.emit_crossing_branch(width)?;
        let mask = width.mask();
        // Hot: the lane is inside one wasm word — a whole word is a plain
        // cell store (an `i64` narrowed first), an `i64` is two cells, a
        // sub-word clears its lane and inserts `value << s`.
        match width {
            MemWidth::B4 if !wide => {
                let _ = self.emit(store(Reg::T1, value, 0))?;
            }
            MemWidth::B4 => self.emit_all([
                Ir::alu_imm(AluOp::And, Reg::T3, value, mask),
                store(Reg::T1, Reg::T3, 0),
            ])?,
            MemWidth::B8 => self.emit_all([
                Ir::alu_imm(AluOp::And, Reg::T3, value, WASM_WORD_MASK),
                store(Reg::T1, Reg::T3, 0),
                Ir::alu_imm(AluOp::Srl, Reg::T3, value, shift_right_bitmask(32)),
                store(Reg::T1, Reg::T3, LINEAR_CELL_BYTES as i64),
            ])?,
            MemWidth::B1 | MemWidth::B2 => self.emit_all([
                mul64(Reg::T2, Reg::T2, imm(SHIFT_PER_LANE)),
                load(Reg::T3, Reg::T1, 0),
                Ir::alu_imm(AluOp::Pow2, Reg::T4, Reg::T2, 0),
                Ir::const_(Reg::T0, mask),
                mul64(Reg::T0, Reg::T0, reg(Reg::T4)),
                Ir::alu(AluOp::Andn, Reg::T3, Reg::T3, reg(Reg::T0)),
                Ir::alu_imm(AluOp::And, Reg::T0, value, mask),
                mul64(Reg::T0, Reg::T0, reg(Reg::T4)),
                Ir::alu(AluOp::Or, Reg::T3, Reg::T3, reg(Reg::T0)),
                store(Reg::T1, Reg::T3, 0),
            ])?,
        }
        // Cold: the lane crosses into the next wasm word(s). With `s =
        // 8·(a & 3)`: cell 0 keeps its bytes below `s` and takes `value << s`
        // (truncated to the cell); cell 1 takes `value >> (32 − s)`; for an
        // `i64`, cell 2 takes `value >> (64 − s) = (value >> (32 − s)) >> 32`.
        // `T0` = `s`, then the `32 − s` shift bitmask.
        if let Some(crossing) = crossing {
            let bm32 = shift_right_bitmask(32);
            let mut cold = vec![
                mul64(Reg::T2, Reg::T2, imm(SHIFT_PER_LANE)),
                load(Reg::T3, Reg::T1, 0),
                Ir::alu_imm(AluOp::Pow2, Reg::T4, Reg::T2, 0),
                Ir::const_(Reg::T0, mask),
                mul64(Reg::T0, Reg::T0, reg(Reg::T4)),
                Ir::alu_imm(AluOp::And, Reg::T0, Reg::T0, WASM_WORD_MASK),
                Ir::alu(AluOp::Andn, Reg::T3, Reg::T3, reg(Reg::T0)),
                Ir::alu_imm(AluOp::And, Reg::T0, value, mask),
                mul64(Reg::T0, Reg::T0, reg(Reg::T4)),
                Ir::alu_imm(AluOp::And, Reg::T0, Reg::T0, WASM_WORD_MASK),
                Ir::alu(AluOp::Or, Reg::T3, Reg::T3, reg(Reg::T0)),
                store(Reg::T1, Reg::T3, 0),
                Ir::const_(Reg::T0, 32),
                Ir::alu(AluOp::Sub(Width::W64), Reg::T0, Reg::T0, reg(Reg::T2)),
                Ir::alu_imm(AluOp::ShiftRightBitmask, Reg::T0, Reg::T0, 0),
                load(Reg::T3, Reg::T1, LINEAR_CELL_BYTES as i64),
                Ir::const_(Reg::T4, mask),
                Ir::alu(AluOp::Srl, Reg::T4, Reg::T4, reg(Reg::T0)),
                Ir::alu_imm(AluOp::And, Reg::T4, Reg::T4, WASM_WORD_MASK),
                Ir::alu(AluOp::Andn, Reg::T3, Reg::T3, reg(Reg::T4)),
                Ir::alu_imm(AluOp::And, Reg::T4, value, mask),
                Ir::alu(AluOp::Srl, Reg::T4, Reg::T4, reg(Reg::T0)),
                Ir::alu_imm(AluOp::And, Reg::T4, Reg::T4, WASM_WORD_MASK),
                Ir::alu(AluOp::Or, Reg::T3, Reg::T3, reg(Reg::T4)),
                store(Reg::T1, Reg::T3, LINEAR_CELL_BYTES as i64),
            ];
            if width == MemWidth::B8 {
                cold.extend([
                    load(Reg::T3, Reg::T1, 2 * LINEAR_CELL_BYTES as i64),
                    Ir::const_(Reg::T4, u64::MAX),
                    Ir::alu(AluOp::Srl, Reg::T4, Reg::T4, reg(Reg::T0)),
                    Ir::alu_imm(AluOp::Srl, Reg::T4, Reg::T4, bm32),
                    Ir::alu(AluOp::Andn, Reg::T3, Reg::T3, reg(Reg::T4)),
                    Ir::alu(AluOp::Srl, Reg::T4, value, reg(Reg::T0)),
                    Ir::alu_imm(AluOp::Srl, Reg::T4, Reg::T4, bm32),
                    Ir::alu(AluOp::Or, Reg::T3, Reg::T3, reg(Reg::T4)),
                    store(Reg::T1, Reg::T3, 2 * LINEAR_CELL_BYTES as i64),
                ]);
            }
            self.cold(crossing, cold)?;
        }
        Ok(())
    }

    /// `rd = op(a, b)` for a source binary operator: one catalog row or its
    /// expansion.
    fn emit_binary(
        &mut self,
        width: Width,
        op: BinaryOp,
        rd: Reg,
        a: Reg,
        b: Operand,
    ) -> Result<(), LowerError> {
        use AluOp as A;
        let alu = |op, rd, rs1, rs2| Ir::alu(op, rd, rs1, reg(rs2));
        let direct = |op| Ir::alu(op, rd, a, b);
        let b = match (op, b) {
            (BinaryOp::Add, _) => return self.emit_all([direct(A::Add(width))]),
            (BinaryOp::Sub, _) => return self.emit_all([direct(A::Sub(width))]),
            (BinaryOp::Mul, _) => return self.emit_all([direct(A::Mul(width))]),
            (BinaryOp::And, _) => return self.emit_all([direct(A::And)]),
            (BinaryOp::Or, _) => return self.emit_all([direct(A::Or)]),
            (BinaryOp::Xor, _) => return self.emit_all([direct(A::Xor)]),
            (BinaryOp::Eq, _) => return self.emit_all([direct(A::Eq)]),
            (BinaryOp::Ne, _) => return self.emit_all([direct(A::Ne)]),
            (BinaryOp::LtU, _) => return self.emit_all([direct(A::LtU)]),
            (BinaryOp::LeU, _) => return self.emit_all([direct(A::LeU)]),
            (BinaryOp::GeU, _) => return self.emit_all([direct(A::GeU)]),
            (_, Operand::Reg(b)) => b,
            (_, Operand::Imm(imm)) => {
                let _ = self.emit(Ir::const_(Reg::T0, imm))?;
                Reg::T0
            }
        };
        match op {
            BinaryOp::Add
            | BinaryOp::Sub
            | BinaryOp::Mul
            | BinaryOp::And
            | BinaryOp::Or
            | BinaryOp::Xor
            | BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::LtU
            | BinaryOp::LeU
            | BinaryOp::GeU => unreachable!("handled above"),
            BinaryOp::GtU => self.emit_all([alu(A::LtU, rd, b, a)]),
            BinaryOp::LtS | BinaryOp::GtS | BinaryOp::LeS | BinaryOp::GeS => {
                let (table, swap) = match op {
                    BinaryOp::LtS => (A::LtS, false),
                    BinaryOp::GtS => (A::LtS, true),
                    BinaryOp::GeS => (A::GeS, false),
                    _ => (A::GeS, true),
                };
                let (x, y) = match width {
                    Width::W64 => (a, b),
                    Width::W32 => {
                        self.emit_all([
                            Ir::alu_imm(A::SignExtendWord, Reg::T0, a, 0),
                            Ir::alu_imm(A::SignExtendWord, Reg::T1, b, 0),
                        ])?;
                        (Reg::T0, Reg::T1)
                    }
                };
                let (x, y) = if swap { (y, x) } else { (x, y) };
                self.emit_all([alu(table, rd, x, y)])
            }
            BinaryOp::Shl => match width {
                Width::W64 => self.emit_all([
                    Ir::alu_imm(A::Pow2, Reg::T0, b, 0),
                    alu(A::Mul(Width::W64), rd, a, Reg::T0),
                ]),
                Width::W32 => self.emit_all([
                    Ir::alu_imm(A::And, Reg::T0, b, 31),
                    Ir::alu_imm(A::Pow2, Reg::T0, Reg::T0, 0),
                    alu(A::Mul(Width::W32), rd, a, Reg::T0),
                ]),
            },
            BinaryOp::ShrU => {
                self.emit_shift_bitmask(width, b)?;
                self.emit_all([alu(A::Srl, rd, a, Reg::T0)])
            }
            BinaryOp::ShrS => match width {
                Width::W64 => {
                    self.emit_shift_bitmask(width, b)?;
                    self.emit_all([alu(A::Sra, rd, a, Reg::T0)])
                }
                Width::W32 => {
                    self.emit_shift_bitmask(width, b)?;
                    self.emit_all([
                        Ir::alu_imm(A::SignExtendWord, Reg::T1, a, 0),
                        alu(A::Sra, Reg::T1, Reg::T1, Reg::T0),
                        Ir::alu_imm(A::LowerHalfWord, rd, Reg::T1, 0),
                    ])
                }
            },
            BinaryOp::Rotr => self.emit_rotr(width, rd, a, b),
            BinaryOp::Rotl => {
                let _ = self.emit(Ir::alu(A::Sub(Width::W64), Reg::T1, Reg::ZERO, reg(b)))?;
                self.emit_rotr(width, rd, a, Reg::T1)
            }
            BinaryOp::DivU | BinaryOp::RemU => {
                self.emit_assert_nonzero(b)?;
                self.emit_unsigned_divmod(width, a, b)?;
                let rs = if op == BinaryOp::DivU {
                    Reg::T3
                } else {
                    Reg::T4
                };
                self.emit_all([Ir::mov(rd, rs)])
            }
            BinaryOp::DivS | BinaryOp::RemS => self.emit_signed_divmod(width, op, rd, a, b),
        }
    }

    /// `T0` = the right-shift bitmask for count `b` masked to the width.
    fn emit_shift_bitmask(&mut self, width: Width, b: Reg) -> Result<(), LowerError> {
        let count = match width {
            Width::W64 => b,
            Width::W32 => {
                let _ = self.emit(Ir::alu_imm(AluOp::And, Reg::T0, b, 31))?;
                Reg::T0
            }
        };
        let _ = self.emit(Ir::alu_imm(AluOp::ShiftRightBitmask, Reg::T0, count, 0))?;
        Ok(())
    }

    /// `rd = rotr(a, b)`; 32-bit rotates duplicate the word into the high
    /// half and shift the 64-bit value.
    fn emit_rotr(&mut self, width: Width, rd: Reg, a: Reg, b: Reg) -> Result<(), LowerError> {
        match width {
            Width::W64 => {
                self.emit_shift_bitmask(width, b)?;
                self.emit_all([Ir::alu(AluOp::Rotr, rd, a, reg(Reg::T0))])
            }
            Width::W32 => {
                self.emit_all([
                    mul64(Reg::T2, a, imm(1 << 32)),
                    Ir::alu(AluOp::Or, Reg::T2, Reg::T2, reg(a)),
                ])?;
                self.emit_shift_bitmask(width, b)?;
                self.emit_all([
                    Ir::alu(AluOp::Srl, Reg::T2, Reg::T2, reg(Reg::T0)),
                    Ir::alu_imm(AluOp::LowerHalfWord, rd, Reg::T2, 0),
                ])
            }
        }
    }

    fn emit_assert_nonzero(&mut self, b: Reg) -> Result<(), LowerError> {
        self.emit_all([assert(AluOp::Ne, AssertFailure::DivideByZero, b, imm(0))])
    }

    /// Unsigned `a / b` into `T3` and `a % b` into `T4` (`b != 0` already
    /// asserted), pinned by `q·b` not overflowing, `q·b <= a`, and
    /// `a − q·b < b`. Uses `T2` as scratch. At 32 bits the product cannot
    /// wrap, so the overflow assert is skipped.
    fn emit_unsigned_divmod(&mut self, width: Width, a: Reg, b: Reg) -> Result<(), LowerError> {
        let _ = self.emit(Ir::Advice {
            hint: AdviceHint::QuotientU,
            rd: Reg::T3,
            rs1: a,
            rs2: b,
        })?;
        if width == Width::W64 {
            let _ = self.emit(assert(
                AluOp::MulUNoOverflow,
                AssertFailure::IntegerOverflow,
                Reg::T3,
                reg(b),
            ))?;
        }
        self.emit_all([
            mul64(Reg::T2, Reg::T3, reg(b)),
            assert(AluOp::LeU, AssertFailure::IntegerOverflow, Reg::T2, reg(a)),
            Ir::alu(AluOp::Sub(Width::W64), Reg::T4, a, reg(Reg::T2)),
            assert(AluOp::LtU, AssertFailure::IntegerOverflow, Reg::T4, reg(b)),
        ])
    }

    /// Signed division through magnitudes: operands are sign-extended to 64
    /// bits (32-bit case), negated by their own sign, divided unsigned, and
    /// the result re-signed (quotient by `a ^ b`, remainder by `a`).
    fn emit_signed_divmod(
        &mut self,
        width: Width,
        op: BinaryOp,
        rd: Reg,
        a: Reg,
        b: Reg,
    ) -> Result<(), LowerError> {
        self.emit_assert_nonzero(b)?;
        if op == BinaryOp::DivS {
            let (min, neg_one) = signed_overflow_pair(width);
            self.emit_all([
                Ir::alu_imm(AluOp::Eq, Reg::T2, a, min),
                Ir::alu_imm(AluOp::Eq, Reg::T4, b, neg_one),
                Ir::alu(AluOp::And, Reg::T2, Reg::T2, reg(Reg::T4)),
                assert(AluOp::Eq, AssertFailure::IntegerOverflow, Reg::T2, imm(0)),
            ])?;
        }
        let widen = |rd, rs| match width {
            Width::W64 => Ir::mov(rd, rs),
            Width::W32 => Ir::alu_imm(AluOp::SignExtendWord, rd, rs, 0),
        };
        self.emit_all([
            widen(Reg::T0, a),
            Ir::alu(AluOp::NegateIf, Reg::T0, Reg::T0, reg(Reg::T0)),
            widen(Reg::T1, b),
            Ir::alu(AluOp::NegateIf, Reg::T1, Reg::T1, reg(Reg::T1)),
        ])?;
        self.emit_unsigned_divmod(Width::W64, Reg::T0, Reg::T1)?;
        let (sign_source, magnitude) = if op == BinaryOp::DivS {
            let _ = self.emit(Ir::alu(AluOp::Xor, Reg::T2, a, reg(b)))?;
            (Reg::T2, Reg::T3)
        } else {
            (a, Reg::T4)
        };
        let _ = self.emit(widen(Reg::T2, sign_source))?;
        let (target, canonicalize) = match width {
            Width::W64 => (rd, false),
            Width::W32 => (Reg::T2, true),
        };
        let _ = self.emit(Ir::alu(AluOp::NegateIf, target, Reg::T2, reg(magnitude)))?;
        if canonicalize {
            let _ = self.emit(Ir::alu_imm(AluOp::LowerHalfWord, rd, Reg::T2, 0))?;
        }
        Ok(())
    }

    fn emit_unary(&mut self, width: Width, op: UnaryOp, rd: Reg, a: Reg) -> Result<(), LowerError> {
        use AluOp as A;
        match (op, width) {
            (UnaryOp::Eqz, _) => self.emit_all([Ir::alu_imm(A::Eq, rd, a, 0)]),
            (UnaryOp::Popcnt, _) => self.emit_all([Ir::alu_imm(A::Popcnt, rd, a, 0)]),
            (UnaryOp::Clz, Width::W64) => self.emit_all([Ir::alu_imm(A::Clz, rd, a, 0)]),
            (UnaryOp::Clz, Width::W32) => self.emit_all([
                Ir::alu_imm(A::Clz, Reg::T0, a, 0),
                Ir::alu_imm(A::Sub(Width::W64), rd, Reg::T0, 32),
            ]),
            (UnaryOp::Ctz, Width::W64) => self.emit_all([Ir::alu_imm(A::Ctz, rd, a, 0)]),
            (UnaryOp::Ctz, Width::W32) => self.emit_all([
                Ir::alu_imm(A::Or, Reg::T0, a, 1 << 32),
                Ir::alu_imm(A::Ctz, rd, Reg::T0, 0),
            ]),
            (UnaryOp::Extend8S, _) => self.emit_sign_extend(width, rd, a, 8),
            (UnaryOp::Extend16S, _) => self.emit_sign_extend(width, rd, a, 16),
            (UnaryOp::Extend32S, _) => self.emit_all([Ir::alu_imm(A::SignExtendWord, rd, a, 0)]),
        }
    }

    /// Lower one operator at stack height `h`. Returns whether `next` (the
    /// following operator) was absorbed as the destination of this one.
    fn lower_op(
        &mut self,
        ctx: &mut FunctionCtx,
        op: &WasmOp,
        h: u32,
        next: Option<&WasmOp>,
    ) -> Result<bool, LowerError> {
        let function = ctx.index;
        let malformed = move || LowerError::MalformedControl(function);
        let mut absorbed = false;
        match *op {
            WasmOp::Nop => {}
            WasmOp::Drop => {
                let _ = ctx.take_pending(h - 1);
            }
            WasmOp::Unreachable => {
                let _ = self.emit(Ir::Trap)?;
            }
            WasmOp::Const(_, value) => ctx.set_pending(h, Pending::Const(value)),
            WasmOp::LocalGet(i) => ctx.set_pending(h, Pending::Local(i)),
            WasmOp::LocalSet(i) | WasmOp::LocalTee(i) => {
                let value = ctx.take_pending(h - 1);
                self.materialize_local(ctx, i)?;
                let rd = ctx.slot(i)?;
                match value {
                    Some(Pending::Local(local)) if local == i => {}
                    Some(Pending::Local(local)) => {
                        let _ = self.emit(Ir::mov(rd, ctx.slot(local)?))?;
                    }
                    Some(Pending::Const(imm)) => {
                        let _ = self.emit(Ir::const_(rd, imm))?;
                    }
                    None => {
                        let _ = self.emit(Ir::mov(rd, ctx.stack(h - 1)?))?;
                    }
                }
                if matches!(*op, WasmOp::LocalTee(_)) {
                    ctx.set_pending(h - 1, Pending::Local(i));
                }
            }
            WasmOp::GlobalGet(g) => {
                let (rd, skip) = self.dest(ctx, h, next)?;
                absorbed = skip;
                let _ = self.emit(load(rd, Reg::ZERO, global_address(g)))?;
            }
            WasmOp::GlobalSet(g) => {
                let value = self.src(ctx, h - 1)?;
                let _ = self.emit(store(Reg::ZERO, value, global_address(g)))?;
            }
            WasmOp::Load {
                ty,
                width,
                signed,
                offset,
            } => {
                let addr = self.src(ctx, h - 1)?;
                let (rd, skip) = self.dest(ctx, h - 1, next)?;
                absorbed = skip;
                self.emit_load(rd, addr, offset, width, signed, ty.width())?;
            }
            WasmOp::Store { ty, width, offset } => {
                let (base, value) = (self.src(ctx, h - 2)?, self.src(ctx, h - 1)?);
                self.emit_store(base, value, ty == ValType::I64, offset, width)?;
            }
            WasmOp::MemorySize => {
                let (rd, skip) = self.dest(ctx, h, next)?;
                absorbed = skip;
                self.emit_all([
                    load(Reg::T0, Reg::ZERO, MEMORY_SIZE_ADDR as i64),
                    Ir::alu_imm(AluOp::Srl, rd, Reg::T0, shift_right_bitmask(16)),
                ])?;
            }
            WasmOp::MemoryGrow => {
                let rs = self.src(ctx, h - 1)?;
                let (rd, skip) = self.dest(ctx, h - 1, next)?;
                absorbed = skip;
                self.emit_memory_grow(rd, rs)?;
            }
            WasmOp::Unary(width, op) => {
                let a = self.src(ctx, h - 1)?;
                let (rd, skip) = self.dest(ctx, h - 1, next)?;
                absorbed = skip;
                self.emit_unary(width, op, rd, a)?;
            }
            WasmOp::Binary(width, op) => {
                let rs1 = self.src(ctx, h - 2)?;
                let rs2 = if binary_takes_immediate(op) {
                    Self::src_operand(ctx, h - 1)?
                } else {
                    Operand::Reg(self.src(ctx, h - 1)?)
                };
                let (rd, skip) = self.dest(ctx, h - 2, next)?;
                absorbed = skip;
                self.emit_binary(width, op, rd, rs1, rs2)?;
            }
            WasmOp::Convert(op) => {
                let a = self.src(ctx, h - 1)?;
                let (rd, skip) = self.dest(ctx, h - 1, next)?;
                absorbed = skip;
                let _ = self.emit(match op {
                    ConvertOp::WrapI64 => Ir::alu_imm(AluOp::And, rd, a, 0xFFFF_FFFF),
                    ConvertOp::ExtendI32S => Ir::alu_imm(AluOp::SignExtendWord, rd, a, 0),
                    ConvertOp::ExtendI32U => Ir::mov(rd, a),
                })?;
            }
            WasmOp::Select => {
                let (v1, v2, cond) = (
                    self.src(ctx, h - 3)?,
                    self.src(ctx, h - 2)?,
                    self.src(ctx, h - 1)?,
                );
                let rd = ctx.stack(h - 3)?;
                if v1 != rd {
                    let _ = self.emit(Ir::mov(rd, v1))?;
                }
                let skip = self.emit(Ir::branch_if_nonzero(cond, 0))?;
                let _ = self.emit(Ir::mov(rd, v2))?;
                let end = self.pc()?;
                self.patch(skip, end);
            }
            WasmOp::Block(sig) => {
                self.materialize_all(ctx)?;
                ctx.labels.push(Label {
                    kind: LabelKind::Block,
                    base: h - sig.params,
                    sig,
                    fixups: Vec::new(),
                });
            }
            WasmOp::Loop(sig) => {
                self.materialize_all(ctx)?;
                let entry = self.pc()?;
                ctx.labels.push(Label {
                    kind: LabelKind::Loop { entry },
                    base: h - sig.params,
                    sig,
                    fixups: Vec::new(),
                });
            }
            WasmOp::If(sig) => {
                let cond = self.src(ctx, h - 1)?;
                self.materialize_all(ctx)?;
                let skip = self.emit(Ir::branch_if_zero(cond, 0))?;
                ctx.labels.push(Label {
                    kind: LabelKind::If { skip },
                    base: h - 1 - sig.params,
                    sig,
                    fixups: Vec::new(),
                });
            }
            WasmOp::Else => {
                self.materialize_all(ctx)?;
                let label = ctx.labels.last_mut().ok_or_else(malformed)?;
                let LabelKind::If { skip } = label.kind else {
                    return Err(malformed());
                };
                let to_end = self.emit(Ir::Jump { target: 0 })?;
                label.fixups.push(to_end);
                label.kind = LabelKind::Else;
                let else_pc = self.pc()?;
                self.patch(skip, else_pc);
            }
            WasmOp::End => {
                self.materialize_all(ctx)?;
                let label = ctx.labels.pop().ok_or_else(malformed)?;
                if ctx.labels.is_empty() {
                    self.emit_return(ctx, h)?;
                }
                let end = self.pc()?;
                if let LabelKind::If { skip } = label.kind {
                    self.patch(skip, end);
                }
                for pc in label.fixups {
                    self.patch(pc, end);
                }
            }
            WasmOp::Br(depth) => {
                self.materialize_all(ctx)?;
                self.branch(ctx, h, depth)?;
            }
            WasmOp::BrIf(depth) => {
                let cond = self.src(ctx, h - 1)?;
                self.materialize_all(ctx)?;
                let h = h - 1;
                let is_function_label = depth as usize + 1 == ctx.labels.len();
                if !is_function_label && !Self::needs_moves(ctx, h, depth)? {
                    let label_index = ctx.labels.len() - 1 - depth as usize;
                    match ctx.labels[label_index].kind {
                        LabelKind::Loop { entry } => {
                            let _ = self.emit(Ir::branch_if_nonzero(cond, entry))?;
                        }
                        LabelKind::Block | LabelKind::If { .. } | LabelKind::Else => {
                            let pc = self.emit(Ir::branch_if_nonzero(cond, 0))?;
                            ctx.labels[label_index].fixups.push(pc);
                        }
                    }
                } else {
                    let skip = self.emit(Ir::branch_if_zero(cond, 0))?;
                    self.branch(ctx, h, depth)?;
                    let end = self.pc()?;
                    self.patch(skip, end);
                }
            }
            WasmOp::BrTable {
                ref targets,
                default,
            } => {
                let index = self.src(ctx, h - 1)?;
                self.materialize_all(ctx)?;
                let h = h - 1;
                for (i, depth) in targets.iter().enumerate() {
                    let _ = self.emit(Ir::const_(Reg::T0, i as u64))?;
                    let skip = self.emit(Ir::Branch {
                        op: AluOp::Ne,
                        rs1: index,
                        rs2: Reg::T0,
                        target: 0,
                    })?;
                    self.branch(ctx, h, *depth)?;
                    let next = self.pc()?;
                    self.patch(skip, next);
                }
                self.branch(ctx, h, default)?;
            }
            WasmOp::Return => {
                self.materialize_all(ctx)?;
                self.emit_return(ctx, h)?;
            }
            WasmOp::Call(callee) => {
                self.materialize_all(ctx)?;
                self.emit_call(ctx, h, Callee::Direct(callee))?;
            }
            WasmOp::CallIndirect(type_index) => {
                self.materialize_all(ctx)?;
                let index = ctx.stack(h - 1)?;
                self.emit_call(ctx, h - 1, Callee::Indirect { type_index, index })?;
            }
        }
        Ok(absorbed)
    }
}

/// Whether [`Emitter::emit_binary`] can take the right operand as an
/// immediate: the single-row catalog ops (`Operand::Imm` is a row's
/// `RightIsImm`); the expansions read `b` as a register.
fn binary_takes_immediate(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Add
            | BinaryOp::Sub
            | BinaryOp::Mul
            | BinaryOp::And
            | BinaryOp::Or
            | BinaryOp::Xor
            | BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::LtU
            | BinaryOp::LeU
            | BinaryOp::GeU
    )
}

fn global_address(index: u32) -> i64 {
    (GLOBALS_BASE + 8 * u64::from(index)) as i64
}

/// Rewrite the jump target (or the return-address immediate of a call site's
/// `const RA` row) of a fixup site.
fn set_target(ir: &mut Ir, new_target: Pc) {
    match ir {
        Ir::Jump { target } | Ir::Branch { target, .. } => *target = new_target,
        Ir::Alu {
            rs2: Operand::Imm(imm),
            ..
        } => *imm = u64::from(new_target),
        _ => {}
    }
}

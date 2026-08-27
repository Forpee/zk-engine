//! Lowering from validated source operators to the register-machine IR: an
//! instruction selector onto the [`AluOp`] table catalog.
//!
//! Register assignment: function frame slot `i` (locals first, then the
//! operand stack) is `Reg::frame_slot(i)`. Structured control flow becomes
//! forward/backward jumps; values carried by a branch are moved down to the
//! target label's base height.
//!
//! Calling convention (all static at the call site):
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
//! - linear-memory accesses of any width/alignment: effective address,
//!   bounds assert against the memory-size word, floor to the containing
//!   doubleword, and word load(s)/store(s) of that doubleword *and the next*
//!   combined with bitmask shifts (`s = 8·(addr & 7)`; the second word is
//!   untouched by a non-crossing access, and the slack word past the end of
//!   memory keeps it in bounds).

use jolt_wasm_ir::layout::{
    input_address, output_address, DEFAULT_MAX_PAGES, GLOBALS_BASE, LINEAR_MEMORY_BASE, MAX_PAGES,
    MEMORY_SIZE_ADDR, PAGE_SIZE, SHADOW_STACK_BASE, TERMINATION_ADDR,
};
use jolt_wasm_ir::{
    shift_right_bitmask, AdviceHint, AluOp, AssertFailure, DataSegment, Ir, IrFunction, IrProgram,
    MemoryLimits, Operand, Pc, Reg, Width, MAX_FRAME_SLOTS, MAX_RESULTS,
};

use crate::error::LowerError;
use crate::module::{Function, WasmModule};
use crate::source::{BinaryOp, BlockSig, ConvertOp, MemWidth, UnaryOp, WasmOp};

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

    let mut signatures = Vec::with_capacity(module.functions.len());
    for (index, function) in module.functions.iter().enumerate() {
        let index = index as u32;
        let ty = module
            .func_type(index)
            .map_err(|_| LowerError::FunctionIndex(index))?;
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
            params: ty.params.len() as u32,
            results: ty.results.len() as u32,
            locals: locals as u32,
            frame_slots: slots,
        });
    }

    let mut emitter = Emitter {
        code: vec![Ir::Halt],
        call_fixups: Vec::new(),
        signatures: &signatures,
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
    })
}

#[derive(Debug, Clone, Copy)]
struct Signature {
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

struct Emitter<'a> {
    code: Vec<Ir>,
    call_fixups: Vec<(Pc, u32)>,
    signatures: &'a [Signature],
    /// The page cap in bytes: `memory.grow` past it fails.
    max_memory_bytes: u64,
}

/// Per-function lowering state.
struct FunctionCtx {
    index: u32,
    sig: Signature,
    labels: Vec<Label>,
}

impl FunctionCtx {
    /// Register for operand-stack depth `depth` (0-based from the bottom).
    fn stack(&self, depth: u32) -> Result<Reg, LowerError> {
        self.slot(self.sig.locals + depth)
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
        };
        for slot in sig.params..sig.locals {
            let rd = ctx.slot(slot)?;
            let _ = self.emit(Ir::mov(rd, Reg::ZERO))?;
        }
        for source in &function.body {
            self.lower_op(&mut ctx, &source.op, source.height)?;
        }
        if !ctx.labels.is_empty() {
            return Err(LowerError::MalformedControl(index));
        }
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

    fn emit_call(&mut self, ctx: &FunctionCtx, height: u32, callee: u32) -> Result<(), LowerError> {
        let callee_sig = *self
            .signatures
            .get(callee as usize)
            .ok_or(LowerError::FunctionIndex(callee))?;
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
        let jump_pc = self.emit(Ir::Jump { target: 0 })?;
        self.call_fixups.push((jump_pc, callee));

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
                function: callee,
                results: results as usize,
                max: MAX_RESULTS,
            })?;
            let _ = self.emit(Ir::mov(rd, rs))?;
        }
        Ok(())
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
                params: 0,
                results: 0,
                locals: 0,
                frame_slots: sig.params.max(sig.results) as usize,
            },
            labels: Vec::new(),
        };
        let _ = self.emit(Ir::const_(Reg::SP, SHADOW_STACK_BASE + 8))?;
        if let Some(start) = start {
            self.emit_call(&ctx, 0, start)?;
        }
        for i in 0..sig.params {
            let rd = ctx.slot(i)?;
            let _ = self.emit(load(rd, Reg::ZERO, input_address(u64::from(i)) as i64))?;
        }
        self.emit_call(&ctx, sig.params, function)?;
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
        self.emit_all([
            store(Reg::ZERO, Reg::T1, MEMORY_SIZE_ADDR as i64),
            Ir::alu_imm(AluOp::Srl, rd, Reg::T0, shift_right_bitmask(16)),
        ])?;
        let done = self.emit(Ir::Jump { target: 0 })?;
        let fail_pc = self.pc()?;
        self.patch(fail, fail_pc);
        let _ = self.emit(Ir::const_(rd, u64::from(u32::MAX)))?;
        let end = self.pc()?;
        self.patch(done, end);
        Ok(())
    }

    /// Effective guest address, bounds assert, then `T1` = the containing
    /// doubleword's address and `T0` = `8·(addr & 7)`, the bit shift of the
    /// access within it.
    fn emit_address(&mut self, addr: Reg, offset: u64, width: MemWidth) -> Result<(), LowerError> {
        let bytes = u64::from(width.bytes());
        self.emit_all([
            add64(Reg::T0, addr, imm(LINEAR_MEMORY_BASE.wrapping_add(offset))),
            add64(
                Reg::T2,
                Reg::T0,
                imm(bytes.wrapping_sub(LINEAR_MEMORY_BASE)),
            ),
            load(Reg::T1, Reg::ZERO, MEMORY_SIZE_ADDR as i64),
            assert(
                AluOp::LeU,
                AssertFailure::OutOfBounds(width.bytes()),
                Reg::T2,
                reg(Reg::T1),
            ),
            Ir::alu_imm(AluOp::And, Reg::T1, Reg::T0, !7),
            Ir::alu_imm(AluOp::And, Reg::T0, Reg::T0, 7),
            mul64(Reg::T0, Reg::T0, imm(8)),
        ])
    }

    /// Load `width` bytes at `addr + offset` into `rd`, sign- or
    /// zero-extended to `result`.
    fn emit_load(
        &mut self,
        rd: Reg,
        offset: u64,
        width: MemWidth,
        signed: bool,
        result: Width,
    ) -> Result<(), LowerError> {
        self.emit_address(rd, offset, width)?;
        // Low word >> s, high word << (64 - s) (zero when s = 0, via
        // `(w · 2^(63-s)) · 2`), combined into T2.
        self.emit_all([
            load(Reg::T2, Reg::T1, 0),
            Ir::alu_imm(AluOp::ShiftRightBitmask, Reg::T3, Reg::T0, 0),
            Ir::alu(AluOp::Srl, Reg::T2, Reg::T2, reg(Reg::T3)),
            load(Reg::T3, Reg::T1, 8),
            Ir::alu_imm(AluOp::Xor, Reg::T4, Reg::T0, 63),
            Ir::alu_imm(AluOp::Pow2, Reg::T4, Reg::T4, 0),
            mul64(Reg::T3, Reg::T3, reg(Reg::T4)),
            mul64(Reg::T3, Reg::T3, imm(2)),
            Ir::alu(AluOp::Or, Reg::T2, Reg::T2, reg(Reg::T3)),
        ])?;
        self.emit_narrow(rd, Reg::T2, width, signed, result)
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
        match (width, signed) {
            (MemWidth::B8, _) => self.emit_all([Ir::mov(rd, rs)]),
            (_, false) => self.emit_all([Ir::alu_imm(AluOp::And, rd, rs, width.mask())]),
            (MemWidth::B1, true) => self.emit_sign_extend(result, rd, rs, 8),
            (MemWidth::B2, true) => self.emit_sign_extend(result, rd, rs, 16),
            (MemWidth::B4, true) => match result {
                Width::W64 => self.emit_all([Ir::alu_imm(AluOp::SignExtendWord, rd, rs, 0)]),
                Width::W32 => self.emit_all([Ir::alu_imm(AluOp::And, rd, rs, width.mask())]),
            },
        }
    }

    /// Sign-extend the low `bits` (8 or 16) of `rs` to `width`, canonical.
    fn emit_sign_extend(
        &mut self,
        width: Width,
        rd: Reg,
        rs: Reg,
        bits: u64,
    ) -> Result<(), LowerError> {
        let up = 64 - bits;
        let (target, canonicalize) = match width {
            Width::W64 => (rd, false),
            Width::W32 => (Reg::T0, true),
        };
        self.emit_all([
            mul64(target, rs, imm(1 << up)),
            Ir::alu_imm(AluOp::Sra, target, target, shift_right_bitmask(up)),
        ])?;
        if canonicalize {
            let _ = self.emit(Ir::alu_imm(AluOp::LowerHalfWord, rd, target, 0))?;
        }
        Ok(())
    }

    /// Store the low `width` bytes of `value` at `base + offset`.
    fn emit_store(
        &mut self,
        base: Reg,
        value: Reg,
        offset: u64,
        width: MemWidth,
    ) -> Result<(), LowerError> {
        self.emit_address(base, offset, width)?;
        let mask = width.mask();
        let bm1 = shift_right_bitmask(1);
        // Low word: clear the target bytes, insert `value << s`.
        self.emit_all([
            load(Reg::T2, Reg::T1, 0),
            Ir::alu_imm(AluOp::Pow2, Reg::T4, Reg::T0, 0),
            Ir::const_(Reg::T3, mask),
            mul64(Reg::T3, Reg::T3, reg(Reg::T4)),
            Ir::alu(AluOp::Andn, Reg::T2, Reg::T2, reg(Reg::T3)),
            Ir::alu_imm(AluOp::And, Reg::T3, value, mask),
            mul64(Reg::T3, Reg::T3, reg(Reg::T4)),
            Ir::alu(AluOp::Or, Reg::T2, Reg::T2, reg(Reg::T3)),
            store(Reg::T1, Reg::T2, 0),
        ])?;
        // High word: the bytes that cross, i.e. `value >> (64 - s)`, computed
        // as `(value >> (63 - s)) >> 1` so that s = 0 contributes nothing.
        self.emit_all([
            load(Reg::T2, Reg::T1, 8),
            Ir::alu_imm(AluOp::Xor, Reg::T4, Reg::T0, 63),
            Ir::alu_imm(AluOp::ShiftRightBitmask, Reg::T4, Reg::T4, 0),
            Ir::const_(Reg::T3, mask),
            Ir::alu(AluOp::Srl, Reg::T3, Reg::T3, reg(Reg::T4)),
            Ir::alu_imm(AluOp::Srl, Reg::T3, Reg::T3, bm1),
            Ir::alu(AluOp::Andn, Reg::T2, Reg::T2, reg(Reg::T3)),
            Ir::alu_imm(AluOp::And, Reg::T3, value, mask),
            Ir::alu(AluOp::Srl, Reg::T3, Reg::T3, reg(Reg::T4)),
            Ir::alu_imm(AluOp::Srl, Reg::T3, Reg::T3, bm1),
            Ir::alu(AluOp::Or, Reg::T2, Reg::T2, reg(Reg::T3)),
            store(Reg::T1, Reg::T2, 8),
        ])
    }

    /// `rd = op(a, b)` for a source binary operator: one catalog row or its
    /// expansion.
    fn emit_binary(
        &mut self,
        width: Width,
        op: BinaryOp,
        rd: Reg,
        a: Reg,
        b: Reg,
    ) -> Result<(), LowerError> {
        use AluOp as A;
        let alu = |op, rd, rs1, rs2| Ir::alu(op, rd, rs1, reg(rs2));
        match op {
            BinaryOp::Add => self.emit_all([alu(A::Add(width), rd, a, b)]),
            BinaryOp::Sub => self.emit_all([alu(A::Sub(width), rd, a, b)]),
            BinaryOp::Mul => self.emit_all([alu(A::Mul(width), rd, a, b)]),
            BinaryOp::And => self.emit_all([alu(A::And, rd, a, b)]),
            BinaryOp::Or => self.emit_all([alu(A::Or, rd, a, b)]),
            BinaryOp::Xor => self.emit_all([alu(A::Xor, rd, a, b)]),
            BinaryOp::Eq => self.emit_all([alu(A::Eq, rd, a, b)]),
            BinaryOp::Ne => self.emit_all([alu(A::Ne, rd, a, b)]),
            BinaryOp::LtU => self.emit_all([alu(A::LtU, rd, a, b)]),
            BinaryOp::GtU => self.emit_all([alu(A::LtU, rd, b, a)]),
            BinaryOp::LeU => self.emit_all([alu(A::LeU, rd, a, b)]),
            BinaryOp::GeU => self.emit_all([alu(A::GeU, rd, a, b)]),
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

    fn lower_op(&mut self, ctx: &mut FunctionCtx, op: &WasmOp, h: u32) -> Result<(), LowerError> {
        let malformed = || LowerError::MalformedControl(ctx.index);
        match *op {
            WasmOp::Nop | WasmOp::Drop => {}
            WasmOp::Unreachable => {
                let _ = self.emit(Ir::Trap)?;
            }
            WasmOp::Const(_, value) => {
                let rd = ctx.stack(h)?;
                let _ = self.emit(Ir::const_(rd, value))?;
            }
            WasmOp::LocalGet(i) => {
                let (rd, rs) = (ctx.stack(h)?, ctx.slot(i)?);
                let _ = self.emit(Ir::mov(rd, rs))?;
            }
            WasmOp::LocalSet(i) | WasmOp::LocalTee(i) => {
                let (rd, rs) = (ctx.slot(i)?, ctx.stack(h - 1)?);
                let _ = self.emit(Ir::mov(rd, rs))?;
            }
            WasmOp::GlobalGet(g) => {
                let rd = ctx.stack(h)?;
                let _ = self.emit(load(rd, Reg::ZERO, global_address(g)))?;
            }
            WasmOp::GlobalSet(g) => {
                let value = ctx.stack(h - 1)?;
                let _ = self.emit(store(Reg::ZERO, value, global_address(g)))?;
            }
            WasmOp::Load {
                ty,
                width,
                signed,
                offset,
            } => {
                let rd = ctx.stack(h - 1)?;
                self.emit_load(rd, offset, width, signed, ty.width())?;
            }
            WasmOp::Store { width, offset } => {
                let (base, value) = (ctx.stack(h - 2)?, ctx.stack(h - 1)?);
                self.emit_store(base, value, offset, width)?;
            }
            WasmOp::MemorySize => {
                let rd = ctx.stack(h)?;
                self.emit_all([
                    load(Reg::T0, Reg::ZERO, MEMORY_SIZE_ADDR as i64),
                    Ir::alu_imm(AluOp::Srl, rd, Reg::T0, shift_right_bitmask(16)),
                ])?;
            }
            WasmOp::MemoryGrow => {
                let rd = ctx.stack(h - 1)?;
                self.emit_memory_grow(rd, rd)?;
            }
            WasmOp::Unary(width, op) => {
                let rd = ctx.stack(h - 1)?;
                self.emit_unary(width, op, rd, rd)?;
            }
            WasmOp::Binary(width, op) => {
                let (rs1, rs2) = (ctx.stack(h - 2)?, ctx.stack(h - 1)?);
                self.emit_binary(width, op, rs1, rs1, rs2)?;
            }
            WasmOp::Convert(op) => {
                let rd = ctx.stack(h - 1)?;
                let _ = self.emit(match op {
                    ConvertOp::WrapI64 => Ir::alu_imm(AluOp::And, rd, rd, 0xFFFF_FFFF),
                    ConvertOp::ExtendI32S => Ir::alu_imm(AluOp::SignExtendWord, rd, rd, 0),
                    ConvertOp::ExtendI32U => Ir::mov(rd, rd),
                })?;
            }
            WasmOp::Select => {
                let (rd, v2, cond) = (ctx.stack(h - 3)?, ctx.stack(h - 2)?, ctx.stack(h - 1)?);
                let skip = self.emit(Ir::branch_if_nonzero(cond, 0))?;
                let _ = self.emit(Ir::mov(rd, v2))?;
                let end = self.pc()?;
                self.patch(skip, end);
            }
            WasmOp::Block(sig) => ctx.labels.push(Label {
                kind: LabelKind::Block,
                base: h - sig.params,
                sig,
                fixups: Vec::new(),
            }),
            WasmOp::Loop(sig) => {
                let entry = self.pc()?;
                ctx.labels.push(Label {
                    kind: LabelKind::Loop { entry },
                    base: h - sig.params,
                    sig,
                    fixups: Vec::new(),
                });
            }
            WasmOp::If(sig) => {
                let cond = ctx.stack(h - 1)?;
                let skip = self.emit(Ir::branch_if_zero(cond, 0))?;
                ctx.labels.push(Label {
                    kind: LabelKind::If { skip },
                    base: h - 1 - sig.params,
                    sig,
                    fixups: Vec::new(),
                });
            }
            WasmOp::Else => {
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
            WasmOp::Br(depth) => self.branch(ctx, h, depth)?,
            WasmOp::BrIf(depth) => {
                let cond = ctx.stack(h - 1)?;
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
                let index = ctx.stack(h - 1)?;
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
            WasmOp::Return => self.emit_return(ctx, h)?,
            WasmOp::Call(callee) => self.emit_call(ctx, h, callee)?,
        }
        Ok(())
    }
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

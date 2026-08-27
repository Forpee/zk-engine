//! The lowered register-machine IR.
//!
//! Every WebAssembly operand-stack slot and local is a fixed virtual register
//! (validation makes the stack height static at each instruction), so each IR
//! instruction is a three-address op with at most two register reads, one
//! register write, and at most one memory access — the shape of a Jolt proof
//! row. Calls spill the caller's frame to a shadow stack in guest RAM (see
//! `jolt-wasm-frontend`'s lowering).
//!
//! Arithmetic rows are [`Ir::Alu`] over the [`AluOp`] **table catalog**: every
//! variant is one prefix–suffix-decomposable lookup table (realized in
//! `jolt-wasm-tables`). WebAssembly operators without a table — `div`/`rem`,
//! `shl`/`rotl`, sub-word memory access, sign extension of bytes, 32-bit
//! signed compares — are expanded by the lowering into sequences of catalog
//! rows, [`Ir::Advice`], and [`Ir::Assert`].
//!
//! Memory contract: [`Ir::Load`]/[`Ir::Store`] move one naturally aligned
//! 64-bit word, matching the doubleword-addressable RAM argument.

use std::collections::BTreeMap;

use crate::ops::Width;

/// Program counter: index into [`IrProgram::code`].
pub type Pc = u32;

/// Size of the virtual register file. Must be a power of two.
pub const REGISTER_COUNT: usize = 128;

/// Frame slots available per function: locals + operand stack.
pub const MAX_FRAME_SLOTS: usize = REGISTER_COUNT - Reg::FRAME_BASE as usize;

/// Maximum results a function may return: one temporary each.
pub const MAX_RESULTS: usize = (Reg::T4.0 - Reg::T0.0 + 1) as usize;

/// A virtual register id, always `< REGISTER_COUNT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Reg(u8);

impl Reg {
    /// Hardwired zero (never written; `mov rd, ZERO` clears a register).
    pub const ZERO: Reg = Reg(0);
    /// Shadow-stack pointer (guest address of the first free shadow slot).
    pub const SP: Reg = Reg(1);
    /// Return address scratch: written by the call site, reloaded by `return`.
    pub const RA: Reg = Reg(2);
    /// Temporaries `T0..=T4`; also carry function results across `return`.
    pub const T0: Reg = Reg(3);
    pub const T1: Reg = Reg(4);
    pub const T2: Reg = Reg(5);
    pub const T3: Reg = Reg(6);
    pub const T4: Reg = Reg(7);
    /// Linear-memory bounds limits, one per access width `w` (1, 2, 4, 8
    /// bytes): the guest address `LINEAR_MEMORY_BASE + 2·size − 2·w + 1`,
    /// so an access at effective guest address `t` is in bounds iff
    /// `t < limit(w)`. Set by the entry stub and by `memory.grow`.
    pub const LIMIT_B: Reg = Reg(8);
    pub const LIMIT_H: Reg = Reg(9);
    pub const LIMIT_W: Reg = Reg(10);
    pub const LIMIT_D: Reg = Reg(11);
    /// First register of the per-function frame.
    pub const FRAME_BASE: u8 = 12;

    /// The bounds-limit register for a `bytes`-wide access.
    pub const fn limit(bytes: u8) -> Option<Reg> {
        match bytes {
            1 => Some(Reg::LIMIT_B),
            2 => Some(Reg::LIMIT_H),
            4 => Some(Reg::LIMIT_W),
            8 => Some(Reg::LIMIT_D),
            _ => None,
        }
    }

    /// Register holding frame slot `slot` (local index, or locals + stack depth).
    pub fn frame_slot(slot: usize) -> Option<Reg> {
        (slot < MAX_FRAME_SLOTS).then(|| Reg(Self::FRAME_BASE + slot as u8))
    }

    /// The register with id `id`, for `id < REGISTER_COUNT`.
    pub fn from_id(id: u8) -> Option<Reg> {
        (usize::from(id) < REGISTER_COUNT).then_some(Reg(id))
    }

    /// Temporary `T0 + i`, for `i < MAX_RESULTS`.
    pub fn temp(i: usize) -> Option<Reg> {
        (i < MAX_RESULTS).then(|| Reg(Self::T0.0 + i as u8))
    }

    #[inline]
    pub fn id(self) -> u8 {
        self.0
    }

    #[inline]
    pub fn index(self) -> usize {
        usize::from(self.0)
    }
}

/// The right operand of an ALU or assert row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operand {
    Reg(Reg),
    Imm(u64),
}

/// How a row's two instruction inputs form its lookup index (mirrors Jolt's
/// `AddOperands`/`SubtractOperands`/`MultiplyOperands` circuit flags).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OperandMode {
    /// `interleave(left, right)`: a two-operand table.
    Interleaved,
    /// Raw index `left + right`.
    Add,
    /// Raw index `left − right + 2^64`.
    Sub,
    /// Raw index `left · right` (128-bit).
    Mul,
}

/// The lookup-table catalog. Every variant is exactly one table; the doc
/// comment gives the function of the instruction inputs `(x, y)`.
///
/// Values are canonical: `i32` results are zero-extended to 64 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum AluOp {
    /// `(x + y) mod 2^bits` — `RangeCheck` at 64, `LowerHalfWord` at 32.
    Add(Width),
    /// `(x − y) mod 2^bits`.
    Sub(Width),
    /// `(x · y) mod 2^bits`.
    Mul(Width),
    And,
    /// `x & !y`.
    Andn,
    Or,
    Xor,
    Eq,
    Ne,
    /// Unsigned `x < y`.
    LtU,
    /// Signed 64-bit `x < y`.
    LtS,
    /// Unsigned `x >= y`.
    GeU,
    /// Signed 64-bit `x >= y`.
    GeS,
    /// Unsigned `x <= y`.
    LeU,
    /// `x >> tz(y)` for a right-shift bitmask `y` (from [`AluOp::ShiftRightBitmask`]).
    Srl,
    /// Arithmetic `x >> tz(y)` for a bitmask `y`.
    Sra,
    /// `rotr(x, tz(y))` for a bitmask `y`.
    Rotr,
    /// `−y` if `x` is negative (64-bit sign), else `y`.
    NegateIf,
    /// `1` iff `x · y < 2^64`.
    MulUNoOverflow,
    /// `2^((x + y) mod 64)`.
    Pow2,
    /// The bitmask `ones(64 − s) << s` with `s = (x + y) mod 64`, consumed by
    /// [`AluOp::Srl`]/[`AluOp::Sra`]/[`AluOp::Rotr`].
    ShiftRightBitmask,
    /// Sign-extend the low 32 bits of `x + y` to 64.
    SignExtendWord,
    /// The low 32 bits of `x + y` (canonicalizes an `i32`).
    LowerHalfWord,
    /// Leading zeros of `x` (64 for zero).
    Clz,
    /// Trailing zeros of `x` (64 for zero).
    Ctz,
    /// Population count of `x`.
    Popcnt,
}

impl AluOp {
    pub fn operand_mode(self) -> OperandMode {
        match self {
            AluOp::Add(_)
            | AluOp::Pow2
            | AluOp::ShiftRightBitmask
            | AluOp::SignExtendWord
            | AluOp::LowerHalfWord => OperandMode::Add,
            AluOp::Sub(_) => OperandMode::Sub,
            AluOp::Mul(_) | AluOp::MulUNoOverflow => OperandMode::Mul,
            AluOp::And
            | AluOp::Andn
            | AluOp::Or
            | AluOp::Xor
            | AluOp::Eq
            | AluOp::Ne
            | AluOp::LtU
            | AluOp::LtS
            | AluOp::GeU
            | AluOp::GeS
            | AluOp::LeU
            | AluOp::Srl
            | AluOp::Sra
            | AluOp::Rotr
            | AluOp::NegateIf
            | AluOp::Clz
            | AluOp::Ctz
            | AluOp::Popcnt => OperandMode::Interleaved,
        }
    }

    /// Whether the output is always `0` or `1` (usable by assert/branch rows).
    pub fn is_boolean(self) -> bool {
        matches!(
            self,
            AluOp::Eq
                | AluOp::Ne
                | AluOp::LtU
                | AluOp::LtS
                | AluOp::GeU
                | AluOp::GeS
                | AluOp::LeU
                | AluOp::MulUNoOverflow
        )
    }

    /// The table's function of the instruction inputs. This is the reference
    /// semantics; `jolt-wasm-tables` checks each table's `materialize_entry`
    /// against it.
    pub fn evaluate(self, x: u64, y: u64) -> u64 {
        let canonical = |v: u64, width: Width| match width {
            Width::W32 => u64::from(v as u32),
            Width::W64 => v,
        };
        let raw = x.wrapping_add(y);
        match self {
            AluOp::Add(w) => canonical(x.wrapping_add(y), w),
            AluOp::Sub(w) => canonical(x.wrapping_sub(y), w),
            AluOp::Mul(w) => canonical(x.wrapping_mul(y), w),
            AluOp::And => x & y,
            AluOp::Andn => x & !y,
            AluOp::Or => x | y,
            AluOp::Xor => x ^ y,
            AluOp::Eq => u64::from(x == y),
            AluOp::Ne => u64::from(x != y),
            AluOp::LtU => u64::from(x < y),
            AluOp::LtS => u64::from((x as i64) < (y as i64)),
            AluOp::GeU => u64::from(x >= y),
            AluOp::GeS => u64::from((x as i64) >= (y as i64)),
            AluOp::LeU => u64::from(x <= y),
            AluOp::Srl => x.unbounded_shr(y.trailing_zeros()),
            AluOp::Sra => (x as i64).unbounded_shr(y.trailing_zeros()) as u64,
            AluOp::Rotr => x.rotate_right(y.trailing_zeros() % 64),
            AluOp::NegateIf => {
                if x >> 63 == 1 {
                    y.wrapping_neg()
                } else {
                    y
                }
            }
            AluOp::MulUNoOverflow => u64::from(x.checked_mul(y).is_some()),
            AluOp::Pow2 => 1u64 << (raw & 63),
            AluOp::ShiftRightBitmask => shift_right_bitmask(raw & 63),
            AluOp::SignExtendWord => raw as u32 as i32 as i64 as u64,
            AluOp::LowerHalfWord => u64::from(raw as u32),
            AluOp::Clz => u64::from(x.leading_zeros()),
            AluOp::Ctz => u64::from(x.trailing_zeros()),
            AluOp::Popcnt => u64::from(x.count_ones()),
        }
    }
}

/// The right-shift bitmask for `shift < 64`: `ones(64 − shift) << shift`.
pub const fn shift_right_bitmask(shift: u64) -> u64 {
    (((1u128 << (64 - shift)) - 1) as u64) << shift
}

/// The guest-visible trap raised when an [`Ir::Assert`] fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssertFailure {
    OutOfBounds(u8),
    DivideByZero,
    IntegerOverflow,
    /// `call_indirect` index past the table.
    TableOutOfBounds,
    /// `call_indirect` on a null slot or a callee of another signature.
    IndirectCallTypeMismatch,
}

/// What an [`Ir::Advice`] row computes (the honest prover's witness).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum AdviceHint {
    /// Unsigned quotient `rs1 / rs2` (`0` when `rs2 == 0`).
    QuotientU,
}

impl AdviceHint {
    pub fn compute(self, x: u64, y: u64) -> u64 {
        match self {
            AdviceHint::QuotientU => x.checked_div(y).unwrap_or(0),
        }
    }
}

/// One lowered instruction. Field names follow the proof-row vocabulary:
/// `rd` is the written register, `rs*` are read registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ir {
    Nop,
    /// Stops execution; the entry trampoline returns here.
    Halt,
    /// `unreachable`.
    Trap,
    /// `rd = op(rs1, rs2)`.
    Alu {
        op: AluOp,
        rd: Reg,
        rs1: Reg,
        rs2: Operand,
    },
    /// Prover-supplied value: `rd = hint(rs1, rs2)`. Unconstrained as a row;
    /// the surrounding `Assert` rows pin it.
    Advice {
        hint: AdviceHint,
        rd: Reg,
        rs1: Reg,
        rs2: Reg,
    },
    /// Trap with `failure` unless `op(rs1, rs2) == 1` (`op` is boolean).
    Assert {
        op: AluOp,
        failure: AssertFailure,
        rs1: Reg,
        rs2: Operand,
    },
    /// `rd = mem[base + offset]`, one aligned 64-bit word.
    Load {
        rd: Reg,
        base: Reg,
        offset: i64,
    },
    /// `mem[base + offset] = value`, one aligned 64-bit word.
    Store {
        base: Reg,
        value: Reg,
        offset: i64,
    },
    Jump {
        target: Pc,
    },
    /// Indirect jump to the address held in `rs` (function return).
    JumpReg {
        rs: Reg,
    },
    /// Jump to `target` iff `op(rs1, rs2) == 1` (`op` is boolean).
    Branch {
        op: AluOp,
        rs1: Reg,
        rs2: Reg,
        target: Pc,
    },
}

impl Ir {
    /// `rd = imm`.
    pub const fn const_(rd: Reg, imm: u64) -> Ir {
        Ir::Alu {
            op: AluOp::Add(Width::W64),
            rd,
            rs1: Reg::ZERO,
            rs2: Operand::Imm(imm),
        }
    }

    /// `rd = rs`.
    pub const fn mov(rd: Reg, rs: Reg) -> Ir {
        Ir::Alu {
            op: AluOp::Add(Width::W64),
            rd,
            rs1: rs,
            rs2: Operand::Imm(0),
        }
    }

    pub const fn alu(op: AluOp, rd: Reg, rs1: Reg, rs2: Operand) -> Ir {
        Ir::Alu { op, rd, rs1, rs2 }
    }

    /// `rd = op(rs1, imm)`.
    pub const fn alu_imm(op: AluOp, rd: Reg, rs1: Reg, imm: u64) -> Ir {
        Ir::Alu {
            op,
            rd,
            rs1,
            rs2: Operand::Imm(imm),
        }
    }

    /// Jump to `target` iff `rs == 0`.
    pub const fn branch_if_zero(rs: Reg, target: Pc) -> Ir {
        Ir::Branch {
            op: AluOp::Eq,
            rs1: rs,
            rs2: Reg::ZERO,
            target,
        }
    }

    /// Jump to `target` iff `rs != 0`.
    pub const fn branch_if_nonzero(rs: Reg, target: Pc) -> Ir {
        Ir::Branch {
            op: AluOp::Ne,
            rs1: rs,
            rs2: Reg::ZERO,
            target,
        }
    }
}

/// Static metadata for one lowered function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrFunction {
    /// First instruction of the function body (the local-zeroing prologue).
    pub entry: Pc,
    pub params: u32,
    pub results: u32,
    /// Registers `FRAME_BASE..FRAME_BASE + frame_slots` are this function's frame.
    pub frame_slots: usize,
}

/// One occupied function-table slot: the callee and its canonical signature
/// (structurally equal function types share one id).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TableSlot {
    pub entry: Pc,
    pub signature: u32,
}

impl TableSlot {
    /// The two guest words of a slot: the entry pc and `signature + 1`, both
    /// zero for a null slot. The one owner of the table encoding
    /// (`layout::TABLE_BASE`).
    pub fn words(slot: Option<TableSlot>) -> [u64; 2] {
        match slot {
            Some(slot) => [u64::from(slot.entry), u64::from(slot.signature) + 1],
            None => [0, 0],
        }
    }
}

/// Linear-memory limits in 64 KiB pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct MemoryLimits {
    pub initial_pages: u64,
    pub max_pages: u64,
}

/// An active data segment, resolved to a linear-memory offset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataSegment {
    pub offset: u64,
    pub bytes: Vec<u8>,
}

/// A complete lowered program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrProgram {
    pub code: Vec<Ir>,
    pub functions: Vec<IrFunction>,
    /// Exported function name → function index.
    pub exports: BTreeMap<String, u32>,
    /// Exported function name → its entry stub's pc. A stub starts from an
    /// all-zero register file: it sets `SP`, runs the `start` function,
    /// loads the parameters from the public input words, calls the function,
    /// stores its results to the public output words, sets the termination
    /// word, and jumps to [`IrProgram::HALT_PC`] — one contiguous trace.
    pub entries: BTreeMap<String, Pc>,
    /// Exported function name → number of public output words its entry
    /// stub stores: the function's results, or — for an export that returns
    /// a pointer and declares `jolt.outputs.<name>` — the words it copies
    /// from linear memory at that pointer.
    pub output_words: BTreeMap<String, u32>,
    pub memory: MemoryLimits,
    /// Initial global values (zero-extended to 64 bits).
    pub globals: Vec<u64>,
    pub data: Vec<DataSegment>,
    /// The `funcref` table after element initialization, `None` for a null
    /// slot; laid out in guest RAM at `layout::TABLE_BASE`.
    pub table: Vec<Option<TableSlot>>,
}

impl IrProgram {
    /// The pc the entry trampoline returns to; `code[HALT_PC]` is [`Ir::Halt`].
    pub const HALT_PC: Pc = 0;
}

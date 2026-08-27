//! Source WebAssembly operators: the supported integer core, decoded from
//! `wasmparser` operators. This is the *source* universe; it knows nothing
//! about registers, rows, or lookup tables (see `jolt_wasm_ir` for the
//! lowered universe).

use jolt_wasm_ir::Width;
use wasmparser::{BlockType, MemArg, Operator};

use crate::error::DecodeError;

/// Integer value types of the supported core (`i32`, `i64`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValType {
    I32,
    I64,
}

impl ValType {
    pub fn width(self) -> Width {
        match self {
            ValType::I32 => Width::W32,
            ValType::I64 => Width::W64,
        }
    }
}

/// Unary integer operators. Width-generic; the width lives on the operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Clz,
    Ctz,
    Popcnt,
    Eqz,
    Extend8S,
    Extend16S,
    /// `i64.extend32_s` (only meaningful at [`Width::W64`]).
    Extend32S,
}

/// Binary integer operators. Width-generic; the width lives on the operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    DivS,
    DivU,
    RemS,
    RemU,
    And,
    Or,
    Xor,
    Shl,
    ShrS,
    ShrU,
    Rotl,
    Rotr,
    Eq,
    Ne,
    LtS,
    LtU,
    GtS,
    GtU,
    LeS,
    LeU,
    GeS,
    GeU,
}

/// Width-changing conversions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvertOp {
    /// `i32.wrap_i64`
    WrapI64,
    /// `i64.extend_i32_s`
    ExtendI32S,
    /// `i64.extend_i32_u`
    ExtendI32U,
}

/// Number of bytes a load/store touches in linear memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemWidth {
    B1,
    B2,
    B4,
    B8,
}

impl MemWidth {
    #[inline]
    pub fn bytes(self) -> u8 {
        match self {
            MemWidth::B1 => 1,
            MemWidth::B2 => 2,
            MemWidth::B4 => 4,
            MemWidth::B8 => 8,
        }
    }

    /// Mask of the low `bytes` bytes.
    #[inline]
    pub fn mask(self) -> u64 {
        match self {
            MemWidth::B1 => 0xFF,
            MemWidth::B2 => 0xFFFF,
            MemWidth::B4 => 0xFFFF_FFFF,
            MemWidth::B8 => u64::MAX,
        }
    }
}

/// Convert a `wasmparser` value type into the supported integer core.
pub fn val_type(ty: wasmparser::ValType) -> Result<ValType, DecodeError> {
    match ty {
        wasmparser::ValType::I32 => Ok(ValType::I32),
        wasmparser::ValType::I64 => Ok(ValType::I64),
        other => Err(DecodeError::UnsupportedValType(other)),
    }
}

/// Block signature: `params -> results`, resolved against the type section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockSig {
    pub params: u32,
    pub results: u32,
}

/// One supported source operator. Immediates are resolved into the form the
/// lowering needs (block signatures counted, `br_table` targets materialized).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WasmOp {
    Unreachable,
    Nop,
    Block(BlockSig),
    Loop(BlockSig),
    If(BlockSig),
    Else,
    End,
    Br(u32),
    BrIf(u32),
    BrTable {
        targets: Vec<u32>,
        default: u32,
    },
    Return,
    Call(u32),
    Drop,
    Select,
    LocalGet(u32),
    LocalSet(u32),
    LocalTee(u32),
    GlobalGet(u32),
    GlobalSet(u32),
    Load {
        /// Result value type.
        ty: ValType,
        width: MemWidth,
        signed: bool,
        offset: u64,
    },
    Store {
        width: MemWidth,
        offset: u64,
    },
    MemorySize,
    MemoryGrow,
    Const(ValType, u64),
    Unary(Width, UnaryOp),
    Binary(Width, BinaryOp),
    Convert(ConvertOp),
}

/// Resolves a `BlockType` into a counted signature.
pub trait BlockTypes {
    fn block_sig(&self, ty: BlockType) -> Result<BlockSig, DecodeError>;
}

impl WasmOp {
    /// Decode one operator; anything outside the supported integer core is a
    /// typed [`DecodeError::UnsupportedOperator`].
    pub fn decode(op: &Operator<'_>, types: &impl BlockTypes) -> Result<Self, DecodeError> {
        use Width::{W32, W64};
        let load = |ty, width, signed, memarg: &MemArg| {
            check_memarg(memarg).map(|()| WasmOp::Load {
                ty,
                width,
                signed,
                offset: memarg.offset,
            })
        };
        let store = |width, memarg: &MemArg| {
            check_memarg(memarg).map(|()| WasmOp::Store {
                width,
                offset: memarg.offset,
            })
        };
        Ok(match op {
            Operator::Unreachable => WasmOp::Unreachable,
            Operator::Nop => WasmOp::Nop,
            Operator::Block { blockty } => WasmOp::Block(types.block_sig(*blockty)?),
            Operator::Loop { blockty } => WasmOp::Loop(types.block_sig(*blockty)?),
            Operator::If { blockty } => WasmOp::If(types.block_sig(*blockty)?),
            Operator::Else => WasmOp::Else,
            Operator::End => WasmOp::End,
            Operator::Br { relative_depth } => WasmOp::Br(*relative_depth),
            Operator::BrIf { relative_depth } => WasmOp::BrIf(*relative_depth),
            Operator::BrTable { targets } => WasmOp::BrTable {
                targets: targets.targets().collect::<Result<Vec<u32>, _>>()?,
                default: targets.default(),
            },
            Operator::Return => WasmOp::Return,
            Operator::Call { function_index } => WasmOp::Call(*function_index),
            Operator::Drop => WasmOp::Drop,
            Operator::Select => WasmOp::Select,
            Operator::LocalGet { local_index } => WasmOp::LocalGet(*local_index),
            Operator::LocalSet { local_index } => WasmOp::LocalSet(*local_index),
            Operator::LocalTee { local_index } => WasmOp::LocalTee(*local_index),
            Operator::GlobalGet { global_index } => WasmOp::GlobalGet(*global_index),
            Operator::GlobalSet { global_index } => WasmOp::GlobalSet(*global_index),

            Operator::I32Load { memarg } => load(ValType::I32, MemWidth::B4, false, memarg)?,
            Operator::I64Load { memarg } => load(ValType::I64, MemWidth::B8, false, memarg)?,
            Operator::I32Load8S { memarg } => load(ValType::I32, MemWidth::B1, true, memarg)?,
            Operator::I32Load8U { memarg } => load(ValType::I32, MemWidth::B1, false, memarg)?,
            Operator::I32Load16S { memarg } => load(ValType::I32, MemWidth::B2, true, memarg)?,
            Operator::I32Load16U { memarg } => load(ValType::I32, MemWidth::B2, false, memarg)?,
            Operator::I64Load8S { memarg } => load(ValType::I64, MemWidth::B1, true, memarg)?,
            Operator::I64Load8U { memarg } => load(ValType::I64, MemWidth::B1, false, memarg)?,
            Operator::I64Load16S { memarg } => load(ValType::I64, MemWidth::B2, true, memarg)?,
            Operator::I64Load16U { memarg } => load(ValType::I64, MemWidth::B2, false, memarg)?,
            Operator::I64Load32S { memarg } => load(ValType::I64, MemWidth::B4, true, memarg)?,
            Operator::I64Load32U { memarg } => load(ValType::I64, MemWidth::B4, false, memarg)?,
            Operator::I32Store { memarg } => store(MemWidth::B4, memarg)?,
            Operator::I64Store { memarg } => store(MemWidth::B8, memarg)?,
            Operator::I32Store8 { memarg } | Operator::I64Store8 { memarg } => {
                store(MemWidth::B1, memarg)?
            }
            Operator::I32Store16 { memarg } | Operator::I64Store16 { memarg } => {
                store(MemWidth::B2, memarg)?
            }
            Operator::I64Store32 { memarg } => store(MemWidth::B4, memarg)?,
            Operator::MemorySize { mem: 0 } => WasmOp::MemorySize,
            Operator::MemoryGrow { mem: 0 } => WasmOp::MemoryGrow,

            Operator::I32Const { value } => WasmOp::Const(ValType::I32, u64::from(*value as u32)),
            Operator::I64Const { value } => WasmOp::Const(ValType::I64, *value as u64),

            Operator::I32Eqz => WasmOp::Unary(W32, UnaryOp::Eqz),
            Operator::I32Clz => WasmOp::Unary(W32, UnaryOp::Clz),
            Operator::I32Ctz => WasmOp::Unary(W32, UnaryOp::Ctz),
            Operator::I32Popcnt => WasmOp::Unary(W32, UnaryOp::Popcnt),
            Operator::I32Extend8S => WasmOp::Unary(W32, UnaryOp::Extend8S),
            Operator::I32Extend16S => WasmOp::Unary(W32, UnaryOp::Extend16S),
            Operator::I64Eqz => WasmOp::Unary(W64, UnaryOp::Eqz),
            Operator::I64Clz => WasmOp::Unary(W64, UnaryOp::Clz),
            Operator::I64Ctz => WasmOp::Unary(W64, UnaryOp::Ctz),
            Operator::I64Popcnt => WasmOp::Unary(W64, UnaryOp::Popcnt),
            Operator::I64Extend8S => WasmOp::Unary(W64, UnaryOp::Extend8S),
            Operator::I64Extend16S => WasmOp::Unary(W64, UnaryOp::Extend16S),
            Operator::I64Extend32S => WasmOp::Unary(W64, UnaryOp::Extend32S),

            Operator::I32Eq => WasmOp::Binary(W32, BinaryOp::Eq),
            Operator::I32Ne => WasmOp::Binary(W32, BinaryOp::Ne),
            Operator::I32LtS => WasmOp::Binary(W32, BinaryOp::LtS),
            Operator::I32LtU => WasmOp::Binary(W32, BinaryOp::LtU),
            Operator::I32GtS => WasmOp::Binary(W32, BinaryOp::GtS),
            Operator::I32GtU => WasmOp::Binary(W32, BinaryOp::GtU),
            Operator::I32LeS => WasmOp::Binary(W32, BinaryOp::LeS),
            Operator::I32LeU => WasmOp::Binary(W32, BinaryOp::LeU),
            Operator::I32GeS => WasmOp::Binary(W32, BinaryOp::GeS),
            Operator::I32GeU => WasmOp::Binary(W32, BinaryOp::GeU),
            Operator::I32Add => WasmOp::Binary(W32, BinaryOp::Add),
            Operator::I32Sub => WasmOp::Binary(W32, BinaryOp::Sub),
            Operator::I32Mul => WasmOp::Binary(W32, BinaryOp::Mul),
            Operator::I32DivS => WasmOp::Binary(W32, BinaryOp::DivS),
            Operator::I32DivU => WasmOp::Binary(W32, BinaryOp::DivU),
            Operator::I32RemS => WasmOp::Binary(W32, BinaryOp::RemS),
            Operator::I32RemU => WasmOp::Binary(W32, BinaryOp::RemU),
            Operator::I32And => WasmOp::Binary(W32, BinaryOp::And),
            Operator::I32Or => WasmOp::Binary(W32, BinaryOp::Or),
            Operator::I32Xor => WasmOp::Binary(W32, BinaryOp::Xor),
            Operator::I32Shl => WasmOp::Binary(W32, BinaryOp::Shl),
            Operator::I32ShrS => WasmOp::Binary(W32, BinaryOp::ShrS),
            Operator::I32ShrU => WasmOp::Binary(W32, BinaryOp::ShrU),
            Operator::I32Rotl => WasmOp::Binary(W32, BinaryOp::Rotl),
            Operator::I32Rotr => WasmOp::Binary(W32, BinaryOp::Rotr),

            Operator::I64Eq => WasmOp::Binary(W64, BinaryOp::Eq),
            Operator::I64Ne => WasmOp::Binary(W64, BinaryOp::Ne),
            Operator::I64LtS => WasmOp::Binary(W64, BinaryOp::LtS),
            Operator::I64LtU => WasmOp::Binary(W64, BinaryOp::LtU),
            Operator::I64GtS => WasmOp::Binary(W64, BinaryOp::GtS),
            Operator::I64GtU => WasmOp::Binary(W64, BinaryOp::GtU),
            Operator::I64LeS => WasmOp::Binary(W64, BinaryOp::LeS),
            Operator::I64LeU => WasmOp::Binary(W64, BinaryOp::LeU),
            Operator::I64GeS => WasmOp::Binary(W64, BinaryOp::GeS),
            Operator::I64GeU => WasmOp::Binary(W64, BinaryOp::GeU),
            Operator::I64Add => WasmOp::Binary(W64, BinaryOp::Add),
            Operator::I64Sub => WasmOp::Binary(W64, BinaryOp::Sub),
            Operator::I64Mul => WasmOp::Binary(W64, BinaryOp::Mul),
            Operator::I64DivS => WasmOp::Binary(W64, BinaryOp::DivS),
            Operator::I64DivU => WasmOp::Binary(W64, BinaryOp::DivU),
            Operator::I64RemS => WasmOp::Binary(W64, BinaryOp::RemS),
            Operator::I64RemU => WasmOp::Binary(W64, BinaryOp::RemU),
            Operator::I64And => WasmOp::Binary(W64, BinaryOp::And),
            Operator::I64Or => WasmOp::Binary(W64, BinaryOp::Or),
            Operator::I64Xor => WasmOp::Binary(W64, BinaryOp::Xor),
            Operator::I64Shl => WasmOp::Binary(W64, BinaryOp::Shl),
            Operator::I64ShrS => WasmOp::Binary(W64, BinaryOp::ShrS),
            Operator::I64ShrU => WasmOp::Binary(W64, BinaryOp::ShrU),
            Operator::I64Rotl => WasmOp::Binary(W64, BinaryOp::Rotl),
            Operator::I64Rotr => WasmOp::Binary(W64, BinaryOp::Rotr),

            Operator::I32WrapI64 => WasmOp::Convert(ConvertOp::WrapI64),
            Operator::I64ExtendI32S => WasmOp::Convert(ConvertOp::ExtendI32S),
            Operator::I64ExtendI32U => WasmOp::Convert(ConvertOp::ExtendI32U),

            other => return Err(DecodeError::UnsupportedOperator(format!("{other:?}"))),
        })
    }
}

fn check_memarg(memarg: &MemArg) -> Result<(), DecodeError> {
    if memarg.memory != 0 {
        return Err(DecodeError::Unsupported("multi-memory"));
    }
    Ok(())
}

//! The Jolt WebAssembly IR: the contract between the frontend
//! (`jolt-wasm-frontend`, which produces an [`IrProgram`]) and the backend
//! (`jolt-wasm-backend`, which executes it; `jolt-wasm-tables`, which realizes
//! the [`AluOp`] catalog as lookup tables).

#![forbid(unsafe_code)]

pub mod bytecode;
pub mod ir;
pub mod layout;
pub mod ops;
pub mod row;

pub use bytecode::{BytecodeColumn, BytecodeRow, PreprocessingError, WasmBytecode, REGISTER_NONE};
pub use ir::{
    shift_right_bitmask, AdviceHint, AluOp, AssertFailure, DataSegment, Ir, IrFunction, IrProgram,
    MemoryLimits, Operand, OperandMode, Pc, Reg, TableSlot, MAX_FRAME_SLOTS, MAX_RESULTS,
    REGISTER_COUNT,
};
pub use ops::Width;
pub use row::{Lookup, RowFlag, RowFlags, RowModel, RowSpec};

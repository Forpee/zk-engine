//! WebAssembly frontend for Jolt: binary module → [`WasmModule`] (validated
//! source operators with static stack heights) → `jolt_wasm_ir::IrProgram`.
//!
//! ```ignore
//! let module = WasmModule::decode(&bytes)?;
//! let program = module.lower()?;
//! ```

#![forbid(unsafe_code)]

pub mod error;
pub mod lower;
pub mod module;
pub mod source;

pub use error::{DecodeError, LowerError};
pub use lower::lower;
pub use module::WasmModule;
pub use source::{BinaryOp, ConvertOp, MemWidth, UnaryOp, ValType, WasmOp};

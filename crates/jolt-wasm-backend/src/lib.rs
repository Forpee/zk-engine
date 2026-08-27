//! Execution backend for the Jolt WebAssembly IR: guest memory, the reference
//! interpreter emitting one [`Record`] per instruction, and the proof-row
//! model ([`RowSpec`], [`check_record`]).
//!
//! ```ignore
//! let run = Machine::new(&program)?.invoke("fib", &[20])?;
//! for record in &run.records {
//!     check_record(record)?;
//! }
//! ```

#![forbid(unsafe_code)]

pub mod error;
pub mod machine;
pub mod memory;
pub mod row;

pub use error::{ExecutionError, Trap};
pub use machine::{Execution, Machine, RamAccess, Record, RegisterRead, RegisterWrite};
pub use memory::Memory;
pub use row::{check_record, Lookup, RowFlags, RowModel, RowSpec, RowViolation};

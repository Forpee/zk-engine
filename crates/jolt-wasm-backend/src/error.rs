use jolt_wasm_ir::Pc;

/// Runtime traps and host-side execution failures.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum ExecutionError {
    #[error("no exported function named `{0}`")]
    UnknownExport(String),
    #[error("entry `{name}` takes {expected} arguments; {actual} were supplied")]
    ArgumentCount {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("trap at pc {pc}: {trap}")]
    Trap { pc: Pc, trap: Trap },
    #[error("execution exceeded {0} steps")]
    StepLimit(u64),
}

/// WebAssembly traps (and shadow-stack exhaustion, which the spec models as a
/// call-stack overflow).
#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum Trap {
    #[error("unreachable")]
    Unreachable,
    #[error("integer divide by zero")]
    DivideByZero,
    #[error("integer overflow")]
    IntegerOverflow,
    #[error("out-of-bounds memory access at {address:#x} ({width} bytes)")]
    OutOfBoundsMemory { address: u64, width: u8 },
    #[error("call stack exhausted")]
    CallStackExhausted,
    #[error("undefined element: call_indirect index past the table")]
    TableOutOfBounds,
    #[error("indirect call type mismatch (or null table slot)")]
    IndirectCallTypeMismatch,
    /// A word access at a non-8-byte-aligned address: a lowering invariant
    /// violation, not a guest fault.
    #[error("unaligned word access at {0:#x}")]
    UnalignedWord(u64),
    /// A linear-memory cell written with a value that is not a 32-bit wasm
    /// word: a lowering invariant violation, not a guest fault.
    #[error("linear-memory cell at {address:#x} written with {value:#x} (not a 32-bit word)")]
    CellOverflow { address: u64, value: u64 },
    #[error("jump to pc {0} outside the program")]
    InvalidJump(u64),
}

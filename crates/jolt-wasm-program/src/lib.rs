//! Proof-side preprocessing of an [`IrProgram`]: the bytecode table the
//! bytecode read-RAF argument commits to and reads by trace pc, the initial
//! memory image the RAM argument starts from, and the compact
//! [`WasmTraceRow`] the witness and sumcheck code consume.
//!
//! ```ignore
//! let program = WasmModule::decode(&bytes)?.lower()?;
//! let preprocessing = WasmProgramPreprocessing::new(&program, max_trace_length)?;
//! let row = preprocessing.bytecode.row(record.pc);
//! ```

#![forbid(unsafe_code)]

pub mod image;
pub mod public;
pub mod r1cs;
pub mod trace_row;

use jolt_wasm_ir::{IrProgram, MemoryLimits};

pub use image::{final_public_words, initial_memory_words, MemoryWord};
pub use jolt_wasm_ir::{BytecodeColumn, BytecodeRow, PreprocessingError, WasmBytecode};
pub use public::{
    max_ram_k, min_ram_k, ProgramImage, PublicInitialRam, PublicIoMemory, PublicMemorySegment,
    RamDomainError, PROGRAM_IMAGE_START_INDEX,
};
pub use trace_row::{build_trace_rows, TraceRowError, WasmTraceRow};

/// Everything the prover and verifier agree on about a program before any
/// execution: the bytecode table, the program's initial memory (without the
/// per-run inputs), and the trace budget.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct WasmProgramPreprocessing {
    pub bytecode: WasmBytecode,
    /// Non-zero program memory words (data segments, globals, the size word).
    pub program_memory: Vec<MemoryWord>,
    pub memory: MemoryLimits,
    /// Maximum padded trace length the proof will accept.
    pub max_trace_length: usize,
}

impl WasmProgramPreprocessing {
    pub fn new(program: &IrProgram, max_trace_length: usize) -> Result<Self, PreprocessingError> {
        Ok(Self {
            bytecode: WasmBytecode::preprocess(program)?,
            program_memory: initial_memory_words(program, &[]),
            memory: program.memory,
            max_trace_length,
        })
    }

    /// The initial memory image of a run: the program words plus the run's
    /// inputs in the public input words, in increasing address order.
    pub fn initial_memory(&self, inputs: &[u64]) -> Vec<MemoryWord> {
        let mut words = self.program_memory.clone();
        words.extend(
            inputs
                .iter()
                .enumerate()
                .filter(|(_, value)| **value != 0)
                .map(|(i, value)| MemoryWord {
                    address: jolt_wasm_ir::layout::input_address(i as u64),
                    value: *value,
                }),
        );
        words.sort_by_key(|word| word.address);
        words
    }

    /// The dense program image: every word from [`GLOBALS_BASE`] up to the
    /// last non-zero program word (globals and data). The memory-size word
    /// below it is public run-independent state, carried separately by
    /// [`PublicInitialRam`].
    pub fn program_image(&self) -> ProgramImage {
        ProgramImage::of_words(&self.program_memory)
    }
}

/// The public input/output of one execution: which export ran, its
/// arguments (the initial input words), and its results (the final output
/// words, with the termination word set).
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct PublicIo {
    pub entry: String,
    pub inputs: Vec<u64>,
    pub outputs: Vec<u64>,
}

impl PublicIo {
    /// The initial memory image of this run: the program memory plus the
    /// inputs.
    pub fn initial_memory(&self, program: &IrProgram) -> Vec<MemoryWord> {
        initial_memory_words(program, &self.inputs)
    }

    /// The words the final memory must hold.
    pub fn final_memory(&self) -> Vec<MemoryWord> {
        final_public_words(&self.outputs)
    }
}

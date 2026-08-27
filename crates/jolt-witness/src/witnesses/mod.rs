//! Atomic witness values: one newtype per witness, each with its
//! single-sourced trace derivation.
//!
//! Every file holds a family's newtypes together with their [`Extract`]
//! impls — the value type, its field encoding, and its derivation from a
//! trace row live side by side, and every consumer path (oracle tables,
//! bundles, streams) dispatches to the same impl. The newtypes themselves
//! are plain values: a backend with a different row representation can
//! construct them directly.
//!
//! Extractors recompute from row accessors — no memoization. The two
//! irreducible non-row inputs are the lookahead window (the `Next*` family
//! is a function of rows `t` and `t + 1`, with padding semantics at
//! `T - 1`) and the environment ([`WitnessEnv`]).

use jolt_field::JoltField;
use jolt_wasm_program::WasmProgramPreprocessing;

use crate::{TraceRow, WitnessError};

mod flags;
mod increments;
mod lookups;
mod one_hot;
mod operands;
mod pc;
mod ram;
mod registers;

pub use flags::{Flag, InstructionRafFlag, LookupTableFlag, ShouldBranch};
pub use increments::{RamInc, RdInc};
pub use lookups::{LookupIndex, LookupOutput, TableIndex};
pub use one_hot::{BytecodeRaChunk, InstructionRaChunk, RaChunkSelector, RamRaChunk};
pub use operands::{
    Imm, LeftInstructionInput, LeftLookupOperand, Product, RightInstructionInput,
    RightLookupOperand,
};
pub use pc::{BytecodePc, NextPc, Pc};
pub use ram::{RamAddress, RamHammingWeight, RamReadValue, RamWriteValue, RemappedRamAddress};
pub use registers::{RdWriteValue, Rs1Value, Rs2Value};

pub(crate) use ram::ram_access_address;

/// Non-row inputs of witness extraction: the program preprocessing.
/// Constructed by backends; opaque to consumers.
pub struct WitnessEnv<'a> {
    pub(crate) preprocessing: &'a WasmProgramPreprocessing,
}

impl<'a> WitnessEnv<'a> {
    pub fn new(preprocessing: &'a WasmProgramPreprocessing) -> Self {
        Self { preprocessing }
    }

    pub fn preprocessing(&self) -> &'a WasmProgramPreprocessing {
        self.preprocessing
    }
}

/// The field encoding of an atomic witness value.
pub trait ToField {
    fn to_field<F: JoltField>(self) -> F;
}

/// The single-sourced derivation of one atomic witness from a trace row.
pub trait Extract<R = TraceRow>: Sized {
    fn extract(row: &R, next: Option<&R>, env: &WitnessEnv<'_>) -> Result<Self, WitnessError>;
}

/// [`Extract`] for indexed witness families ([`Flag`], [`LookupTableFlag`]):
/// which member is extracted is bound at the use site.
pub trait ExtractIndexed<I, R = TraceRow>: Sized {
    fn extract_indexed(
        index: I,
        row: &R,
        next: Option<&R>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError>;
}

//! The trace→witness transformation layer for Jolt proving.
//!
//! Three maps, three homes:
//!
//! ```text
//! trace rows ──(one-to-many: Extract impls)──▶ atomic witnesses (witnesses/)
//! atomic witnesses ──(many-to-many: bundles)──▶ consumer bundles
//! bundles / ids ──(backends)──▶ kernels & commitment
//! ```
//!
//! A witness is an atomic value newtype with a single-sourced derivation from
//! a [`TraceRow`] (the WebAssembly proof row, `jolt_wasm_program::WasmTraceRow`).
//! Backends serve them two ways: the object-safe id-indexed
//! [`JoltWitnessOracle`] (the naive interpreter's path — one exhaustive match
//! over jolt-claims ids, no wildcard) and typed bundles over the streaming
//! pass. This crate defines **no id vocabulary of its own** — all ids are
//! jolt-claims'. Every source supports sequential cycle ranges. Slice-backed
//! sources may additionally expose random-access views for parallel
//! collection; checkpointed, re-emulating sources need not.

// Lets derive-generated `::jolt_witness::...` paths resolve inside this
// crate's own tests.
extern crate self as jolt_witness;

pub mod backend;
#[cfg(any(test, feature = "test-utils"))]
pub mod testing;
pub mod witnesses;

mod bundle;
mod consumer;
mod error;
mod shape;

#[cfg(any(test, feature = "test-utils"))]
pub use backend::fixed::FixedBackend;
pub use backend::trace::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
pub use backend::{
    validate_servable, BundleSource, JoltWitnessOracle, JoltWitnessPlane, ProgramSource,
};
pub use bundle::WitnessBundle;
pub use consumer::{
    collect_bundles, stream_witnesses, ChunkVisitor, CollectBundles, ConsumerSet, RandomAccessRows,
    RowSource, StreamConsumer,
};
pub use error::WitnessError;
pub use shape::{PolynomialEncoding, Shape};

/// The proof-facing trace row every witness derives from.
pub type TraceRow = jolt_wasm_program::WasmTraceRow;

#[doc(hidden)]
pub mod __private {
    pub use crate::TraceRow;
    pub use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltPolynomialId, JoltVirtualPolynomial,
    };
}

/// Word size of the WebAssembly Jolt VM this crate derives witnesses for.
pub const XLEN: usize = 64;

/// The full instruction lookup key width: two `XLEN`-bit operands.
pub const LOOKUP_ADDRESS_BITS: usize = 2 * XLEN;

/// Error label for the Jolt VM witness backend.
pub(crate) const JOLT_VM_LABEL: &str = "jolt_vm";

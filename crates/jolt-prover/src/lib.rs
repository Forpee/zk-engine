//! Modular prover for the Jolt zkVM: a pure consumer of the
//! `SymbolicSumcheck` / `ConcreteSumcheck` / `SumcheckBatch` abstraction stack.
//!
//! `jolt-claims` defines the algebra, `jolt-verifier`'s relations and generated
//! stage drivers define the protocol structure, `jolt-sumcheck` runs the round
//! loop, and `jolt-kernels` owns every field-element crunch (including the
//! naive reference tier). This crate is orchestration only: config and
//! preprocessing, transcript sequencing, kernel invocation, typed claim
//! assembly, and proof assembly. See `specs/clean-slate-prover.md`.
//!
//! The prover path is the homomorphic pipeline over an elliptic-curve PCS
//! (`dory`): streaming per-polynomial witness commitments, the stage 0–8
//! recipes, and the RLC-batched joint opening (`dory::prove`), in the
//! compiled proof mode (transparent, or BlindFold ZK under the `zk`
//! feature). Like `jolt-verifier`, one compiled prover proves exactly one
//! protocol: the `zk` feature swaps the shared recorders to the committed
//! flavor.
//!
//! [`config`]: ProverConfig
//! [`preprocessing`]: JoltProverPreprocessing
//! [`driver`]: StageProver
//! [`error`]: ProverError
//! [`recorder`]: ProofMode

#[cfg(feature = "zk")]
mod blindfold;
mod config;
pub mod dory;
pub mod driver;
mod error;
mod preprocessing;
#[cfg(feature = "profiling")]
pub mod profile;
mod recorder;
pub mod run;
pub mod stages;

pub use config::{CommittedProgramCandidates, ProverConfig};
pub use dory::prove;
pub use driver::{KernelSource, Proved, StageProver};
pub use error::ProverError;
pub use jolt_kernels::{JoltBackend, ProofSession};
pub use preprocessing::{
    preprocess_program, preprocessing_digest, CommittedProgramProverData, JoltProverPreprocessing,
};
pub use recorder::{ModeRecorder, ProofMode, ProvedUniskipMode};
pub use run::PreparedRun;

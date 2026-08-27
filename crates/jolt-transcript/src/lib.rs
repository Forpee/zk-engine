//! Fiat-Shamir transcripts for Jolt, backed by spongefish.
//!
//! Two surfaces:
//!
//! - **Split spongefish-native traits** ([`ProverTranscript`],
//!   [`VerifierTranscript`], [`OptimizedChallenge`]) — implemented directly
//!   on `spongefish::ProverState` / `spongefish::VerifierState`. Use these
//!   for new code.
//! - **Label/append facade** ([`Transcript`], [`AppendToTranscript`],
//!   [`Blake2bTranscript`]) — the surface the proof stack (`jolt-sumcheck`,
//!   `jolt-openings`, `jolt-crypto`, `jolt-verifier`, `jolt-prover`) uses.
//!
//! One sponge: spongefish `Blake2b512`.

#![deny(missing_docs)]
// In the jolt-verifier runtime closure: stricter panic and unsafe discipline
// than the workspace lints (specs/verifier-closure-lints.md).
#![forbid(unsafe_code)]
#![deny(
    clippy::indexing_slicing,
    clippy::get_unwrap,
    clippy::string_slice,
    clippy::fallible_impl_from,
    clippy::mem_forget,
    clippy::exit,
    clippy::panic_in_result_fn,
    clippy::let_underscore_must_use,
    clippy::host_endian_bytes,
    clippy::wildcard_enum_match_arm
)]

mod codec;
mod legacy;
mod prover;
mod setup;
mod verifier;

pub use codec::BytesMsg;
pub use legacy::{
    append_length_prefixed, AppendToTranscript, Label, LabelWithCount, SpongeTranscript,
    Transcript, U64Word, MAX_LABEL_LEN,
};
pub use setup::{prover_transcript, transcript_builder, verifier_transcript, PROTOCOL_ID};

/// Source-compatible re-exports of legacy label / count / word helpers
/// under their `jolt_transcript::domain::*` path (matches the path used
/// by jolt-dory and earlier modular consumers).
pub mod domain {
    pub use crate::legacy::{Label, LabelWithCount, U64Word};
}

pub use prover::{OptimizedChallenge, ProverTranscript};
pub use verifier::VerifierTranscript;

/// Fiat-Shamir transcript backed by Blake2b-512 (spongefish duplex sponge).
pub type Blake2bTranscript<F = jolt_field::Fr> =
    SpongeTranscript<spongefish::instantiations::Blake2b512, F>;

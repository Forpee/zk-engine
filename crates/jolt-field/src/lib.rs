//! Field and ring abstractions for the Jolt zkVM.
//!
//! A slim algebraic ladder — [`AdditiveGroup`] → [`Ring`] → [`Field`] — with
//! orthogonal capabilities: [`CanonicalBytes`]/[`CanonicalEncoding`] (the
//! Fiat-Shamir transcript surface and the field decode surface on top of it)
//! and [`WithAccumulator`] (deferred-reduction fused multiply-add).
//! [`JoltField`] is the blanket-implemented bundle of everything Jolt's
//! protocol stack requires of a scalar field: `Field + CanonicalEncoding +
//! WithAccumulator`. Because the impl is a blanket, no field type can forget
//! to opt in.
//!
//! The one backend is BN254 (`Fr`/`Fq` wrapping arkworks) plus
//! `WideAccumulator`, a 9-limb accumulator with deferred Montgomery
//! reduction (first-party Barrett/Montgomery kernels).
//!
//! # Byte compatibility (hard invariants)
//!
//! - Proof/wire serialization is serde + bincode over canonical
//!   little-endian bytes (see [`impl_serde_bytes!`]); deserialization
//!   rejects non-canonical encodings uniformly via
//!   [`CanonicalEncoding::from_bytes_le_checked`].
//! - Fiat-Shamir transcript bytes use the explicit little-endian encoding
//!   ([`CanonicalBytes::to_bytes_le`]) and never go through a serialization
//!   library.
//!
//! Both are pinned by `tests/golden_bytes.rs` and `tests/bn254_differential.rs`.

// In the jolt-verifier runtime closure: stricter panic discipline than the
// workspace lints (specs/verifier-closure-lints.md).
#![deny(
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
#![forbid(unsafe_code)]

mod algebra;
mod bn254;
mod limbs;
mod ops;
pub mod signed;

pub use algebra::{
    Accumulator, AdditiveGroup, CanonicalBytes, CanonicalEncoding, Field, JoltField,
    NaiveAccumulator, Ring, WithAccumulator,
};
pub use bn254::{Fq, Fr, FrSignedProductAccumulator, FrSmallScalarAccumulator, WideAccumulator};
pub use limbs::Limbs;
pub use num_traits::{One, Zero};

/// Backend-independent input and shape failures.
#[derive(Debug, thiserror::Error)]
pub enum FieldError {
    /// Invalid input parameter or value.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// Length mismatch between an expected and provided shape.
    #[error("invalid size: expected {expected}, actual {actual}")]
    InvalidSize { expected: usize, actual: usize },
}

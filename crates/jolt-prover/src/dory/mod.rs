//! The Dory (elliptic-curve) prove path: the homomorphic pipeline —
//! streaming per-polynomial witness commitments at stage 0, the shared
//! stage 1–7 recipes ([`crate::stages`]), and the RLC-batched stage-8 joint
//! opening.

mod prover;
pub mod stages;

pub use prover::prove;

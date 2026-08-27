use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_field::JoltField;
use jolt_openings::VerifierOpeningClaim;
use jolt_poly::{Point, HIGH_TO_LOW};

#[derive(Clone, Debug)]
pub struct Stage8ClearOutput<F: JoltField, C> {
    pub opening_claims: Vec<VerifierOpeningClaim<F, C>>,
    pub opening_ids: Vec<JoltOpeningId>,
    pub constraint_coefficients: Vec<F>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
    pub joint_claim: F,
    pub joint_commitment: C,
}

#[derive(Clone, Debug)]
pub struct Stage8ZkOutput<F: JoltField, C, H> {
    pub opening_ids: Vec<JoltOpeningId>,
    pub constraint_coefficients: Vec<F>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
    pub joint_commitment: C,
    pub hiding_evaluation_commitment: H,
}

#[derive(Clone, Debug)]
pub enum Stage8Output<F: JoltField, C, H> {
    Clear(Stage8ClearOutput<F, C>),
    /// The akita build's clear stage 8 verifies to completion inside
    /// [`super::verify`] (one packed OneHotTrace opening plus auxiliary packed
    /// openings), so no per-opening payload survives it.
    Zk(Stage8ZkOutput<F, C, H>),
}

impl<F: JoltField, C, H> Stage8Output<F, C, H> {
    pub fn zk(&self) -> Result<&Stage8ZkOutput<F, C, H>, crate::VerifierError> {
        match self {
            Self::Zk(output) => Ok(output),
            Self::Clear(_) => Err(crate::VerifierError::ExpectedCommittedProof { field: "stage8" }),
        }
    }
}

//! Spartan outer remainder symbolic sumcheck relation.

use jolt_field::Ring;
use jolt_wasm_ir::RowFlag;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::spartan::{
    outer_opening, outer_uniskip_opening, SpartanOuterDimensions, OUTER_REMAINDER_DEGREE,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId, SpartanOuterPublic,
};
use crate::{derived, opening, InputClaims, OutputClaims, SymbolicSumcheck};

/// Consumed Spartan outer remainder input: the uni-skip's reduced opening. The
/// relation reads only this value (its output point comes from its own sumcheck
/// point), so the input point is left empty. Generic over the cell.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct OuterRemainderInputClaims<C> {
    #[opening(UnivariateSkip, from = SpartanOuter)]
    pub outer_uniskip: C,
}

/// Produced Spartan outer remainder openings: one per R1CS-input variable, all
/// sharing the single remainder opening point. Generic over the opening cell (`F`
/// for the serialized wire value, `Vec<F>` for the derived opening point). Field
/// order is the canonical Fiat-Shamir / append order and MUST equal
/// [`SpartanOuterDimensions::variables`] /
/// [`SPARTAN_OUTER_R1CS_INPUTS`](crate::protocols::jolt::geometry::spartan::SPARTAN_OUTER_R1CS_INPUTS).
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(SpartanOuter)]
pub struct OuterRemainderOutputClaims<C> {
    #[opening(LeftInstructionInput)]
    pub left_instruction_input: C,
    #[opening(RightInstructionInput)]
    pub right_instruction_input: C,
    #[opening(Product)]
    pub product: C,
    #[opening(PC)]
    pub pc: C,
    #[opening(Imm)]
    pub imm: C,
    #[opening(RamAddress)]
    pub ram_address: C,
    #[opening(Rs1Value)]
    pub rs1_value: C,
    #[opening(Rs2Value)]
    pub rs2_value: C,
    #[opening(RdWriteValue)]
    pub rd_write_value: C,
    #[opening(RamReadValue)]
    pub ram_read_value: C,
    #[opening(RamWriteValue)]
    pub ram_write_value: C,
    #[opening(LeftLookupOperand)]
    pub left_lookup_operand: C,
    #[opening(RightLookupOperand)]
    pub right_lookup_operand: C,
    #[opening(NextPC)]
    pub next_pc: C,
    #[opening(LookupOutput)]
    pub lookup_output: C,
    #[opening(ShouldBranch)]
    pub should_branch: C,
    #[opening(RowFlag(RowFlag::LeftIsRs1))]
    pub left_is_rs1: C,
    #[opening(RowFlag(RowFlag::RightIsRs2))]
    pub right_is_rs2: C,
    #[opening(RowFlag(RowFlag::RightIsImm))]
    pub right_is_imm: C,
    #[opening(RowFlag(RowFlag::AddOperands))]
    pub add_operands: C,
    #[opening(RowFlag(RowFlag::SubOperands))]
    pub sub_operands: C,
    #[opening(RowFlag(RowFlag::MulOperands))]
    pub mul_operands: C,
    #[opening(RowFlag(RowFlag::WriteLookupToRd))]
    pub write_lookup_to_rd: C,
    #[opening(RowFlag(RowFlag::Load))]
    pub load: C,
    #[opening(RowFlag(RowFlag::Store))]
    pub store: C,
    #[opening(RowFlag(RowFlag::Jump))]
    pub jump: C,
    #[opening(RowFlag(RowFlag::Branch))]
    pub branch: C,
    #[opening(RowFlag(RowFlag::Assert))]
    pub assert: C,
    #[opening(RowFlag(RowFlag::Halt))]
    pub halt: C,
    #[opening(RowFlag(RowFlag::Trap))]
    pub trap: C,
    #[opening(RowFlag(RowFlag::Advice))]
    pub advice: C,
}

/// The Spartan outer remainder sumcheck: the quadratic R1CS form over the outer
/// R1CS-input openings, weighted by the `SpartanOuterPublic` coefficients.
#[derive(Clone)]
pub struct OuterRemainder {
    shape: SpartanOuterDimensions,
}

impl SymbolicSumcheck for OuterRemainder {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = SpartanOuterDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = OuterRemainderInputClaims<C>;
    type Outputs<C> = OuterRemainderOutputClaims<C>;

    fn new(shape: SpartanOuterDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::SpartanOuter
    }

    fn rounds(&self) -> usize {
        self.shape.remainder_rounds()
    }

    fn degree(&self) -> usize {
        OUTER_REMAINDER_DEGREE
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(outer_uniskip_opening())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        // The factored quadratic form `tau_kernel · Az · Bz` with each linear
        // form expanded over its per-column weights — every derived leaf one
        // multilinear (the weights are linear in the stream variable).
        let mut az = JoltExpr::zero();
        let mut bz = JoltExpr::zero();
        for (index, variable) in self.shape.variables().iter().copied().enumerate() {
            az = az
                + derived(JoltDerivedId::from(SpartanOuterPublic::AzWeight(index)))
                    * opening(outer_opening(variable));
            bz = bz
                + derived(JoltDerivedId::from(SpartanOuterPublic::BzWeight(index)))
                    * opening(outer_opening(variable));
        }
        if self.shape.include_affine_terms() {
            az = az + derived(JoltDerivedId::from(SpartanOuterPublic::AzConstant));
            bz = bz + derived(JoltDerivedId::from(SpartanOuterPublic::BzConstant));
        }

        derived(JoltDerivedId::from(SpartanOuterPublic::TauKernel)) * az * bz
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::JoltVirtualPolynomial;
    use jolt_field::{Fr, Ring};

    /// The expanded `output_expression` reproduces the factored quadratic form
    /// `tau_kernel * (Σ az[i] o[i] + az_c) * (Σ bz[i] o[i] + bz_c)` when fed the
    /// `public_coefficients` expansion of those linear forms. This is the same
    /// expansion `JoltSpartanOuterRemainder::public_coefficients` produces and the
    /// verifier's `derive_output_term` resolves against; equality with the factored
    /// form is the invariant the clear stage-1 path relies on.
    #[test]
    fn output_expression_matches_factored_quadratic_form() {
        let dimensions = match SpartanOuterDimensions::new(
            8,
            vec![
                JoltVirtualPolynomial::PC,
                JoltVirtualPolynomial::LookupOutput,
            ],
            true,
        ) {
            Some(dimensions) => dimensions,
            None => unreachable!("test Spartan outer dimensions should be valid"),
        };
        let relation = OuterRemainder::new(dimensions);

        let openings = [Fr::from_u64(2), Fr::from_u64(3)];
        let tau_kernel = Fr::from_u64(17);
        let az = [Fr::from_u64(5), Fr::from_u64(7)];
        let bz = [Fr::from_u64(11), Fr::from_u64(13)];
        let az_constant = Fr::from_u64(19);
        let bz_constant = Fr::from_u64(23);

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == outer_opening(JoltVirtualPolynomial::PC) => openings[0],
                id if id == outer_opening(JoltVirtualPolynomial::LookupOutput) => openings[1],
                _ => Fr::from_u64(0),
            },
            |_| Fr::from_u64(0),
            |id| match *id {
                JoltDerivedId::SpartanOuter(SpartanOuterPublic::TauKernel) => tau_kernel,
                JoltDerivedId::SpartanOuter(SpartanOuterPublic::AzWeight(index)) => az[index],
                JoltDerivedId::SpartanOuter(SpartanOuterPublic::BzWeight(index)) => bz[index],
                JoltDerivedId::SpartanOuter(SpartanOuterPublic::AzConstant) => az_constant,
                JoltDerivedId::SpartanOuter(SpartanOuterPublic::BzConstant) => bz_constant,
                _ => Fr::from_u64(0),
            },
        );

        let az_form = az[0] * openings[0] + az[1] * openings[1] + az_constant;
        let bz_form = bz[0] * openings[0] + bz[1] * openings[1] + bz_constant;
        assert_eq!(output, tau_kernel * az_form * bz_form);
    }

    /// Pins the row-flag coverage of the outer-remainder output claims: every
    /// `RowFlag` has a field (a newly added flag missing its field would leave
    /// an R1CS input opening unresolvable — and desynchronize the canonical
    /// `SPARTAN_OUTER_R1CS_INPUTS` append order this struct encodes).
    #[test]
    fn output_claims_cover_row_flags() {
        let claims = OuterRemainderOutputClaims::<Fr>::default();
        for flag in RowFlag::ALL {
            let id = outer_opening(JoltVirtualPolynomial::RowFlag(flag));
            assert!(
                claims.resolve_output(&id).is_some(),
                "missing outer-remainder output field for RowFlag({flag:?})",
            );
        }
    }
}

//! The stage 3 `SpartanShift` sumcheck instance.
//!
//! Owns the shift opening-point derivation and the `EqPlusOne` public-value
//! computations (against the product uni-skip `tau_low` and the product-remainder
//! opening point).

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::spartan::{
    SpartanShiftChallenges, SpartanShiftInputClaims, SpartanShiftOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::dimensions::TraceDimensions, JoltDerivedId, SpartanShiftPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;
use jolt_poly::EqPlusOnePolynomial;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage1::Stage1BatchOutputClaims;
use crate::VerifierError;

/// Wire shift's consumed opening *value* (`NextPC` from stage 1's outer
/// sumcheck). Takes the ZK-agnostic upstream output-claims aggregate.
pub fn spartan_shift_input_values_from_upstream<F: JoltField>(
    stage1: &Stage1BatchOutputClaims<F>,
) -> SpartanShiftInputClaims<F> {
    SpartanShiftInputClaims {
        next_pc: stage1.outer_remainder.next_pc,
    }
}

#[derive(Clone)]
pub struct SpartanShift<F: JoltField> {
    symbolic: relations::spartan::Shift,
    product_uniskip_tau_low: Vec<F>,
    product_remainder_opening_point: Vec<F>,
}

impl<F: JoltField> SpartanShift<F> {
    pub fn new(
        trace_dimensions: TraceDimensions,
        product_uniskip_tau_low: Vec<F>,
        product_remainder_opening_point: Vec<F>,
    ) -> Self {
        Self {
            symbolic: relations::spartan::Shift::new(trace_dimensions),
            product_uniskip_tau_low,
            product_remainder_opening_point,
        }
    }

    pub fn product_uniskip_tau_low(&self) -> &[F] {
        &self.product_uniskip_tau_low
    }

    pub fn product_remainder_opening_point(&self) -> &[F] {
        &self.product_remainder_opening_point
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for SpartanShift<F> {
    type Symbolic = relations::spartan::Shift;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &SpartanShiftInputClaims<Vec<F>>,
    ) -> Result<SpartanShiftOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        Ok(SpartanShiftOutputClaims { pc: opening_point })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &SpartanShiftInputClaims<Vec<F>>,
        output_points: &SpartanShiftOutputClaims<Vec<F>>,
        _challenges: &SpartanShiftChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::SpartanShift(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        // Every shift output shares the one shift opening point.
        let opening_point = output_points.pc();
        match public_id {
            SpartanShiftPublic::EqPlusOneOuter => Ok(EqPlusOnePolynomial::new(
                self.product_uniskip_tau_low.clone(),
            )
            .evaluate(opening_point)),
            SpartanShiftPublic::EqPlusOneProduct => Ok(EqPlusOnePolynomial::new(
                self.product_remainder_opening_point.clone(),
            )
            .evaluate(opening_point)),
        }
    }
}

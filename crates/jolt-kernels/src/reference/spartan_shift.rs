//! The Spartan shift (stage 3) kernel: a naive member over the cycle domain.
//!
//! The summand is `eq⁺¹(τ_low, j) · pc(j)` — the shift is carried entirely
//! by the `eq+1` factor (the pc table is unshifted, defined at every cycle
//! including `T − 1`); the `eq+1` table is one multilinear whose MLE is the
//! verifier's closed-form `EqPlusOnePolynomial::evaluate`.

use std::collections::BTreeMap;

use crate::ProverInputs;
use jolt_claims::protocols::jolt::geometry::spartan::pc_shift;
use jolt_claims::protocols::jolt::{JoltDerivedId, SpartanShiftPublic};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, EqPlusOnePolynomial, Polynomial};
use jolt_verifier::stages::stage3::outputs::SpartanShift;
use jolt_witness::JoltWitnessPlane;

use super::views::dense_view;
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel,
};

impl<F: JoltField> PrepareKernel<F, SpartanShift<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, SpartanShift<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = SpartanShift<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let product_uniskip_tau_low = relation.product_uniskip_tau_low();
        let opening_tables = BTreeMap::from([(
            pc_shift(),
            Polynomial::new(dense_view(witness, pc_shift())?),
        )]);
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(SpartanShiftPublic::EqPlusOneOuter),
            Polynomial::new(EqPlusOnePolynomial::evals(product_uniskip_tau_low, None).1),
        )]);

        Ok(Box::new(NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}

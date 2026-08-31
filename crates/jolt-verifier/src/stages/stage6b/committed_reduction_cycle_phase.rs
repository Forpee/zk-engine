//! The stage 6b committed-program claim-reduction cycle phases.
//!
//! In committed-program mode the program-image and per-chunk bytecode
//! polynomials are reduced over the shared precommitted schedule. Each reduction's
//! cycle phase runs here (stage 6b); when the polynomial still has active
//! address-phase rounds it stages an intermediate opening consumed by stage 7,
//! otherwise it completes here, opening the committed polynomial under a
//! `FinalScale` / `ChunkOutputWeight` public.
//!
//! Because the produced opening point is the (reverse-ordered) cycle opening
//! point while the scale is evaluated at the Dory-permuted point, each relation
//! OVERRIDES [`ConcreteSumcheck::expected_output`] to recover the scale from the
//! produced opening point via the layout's `cycle_phase_*_at_opening_point`
//! helpers — see [`PrecommittedClaimReduction::cycle_phase_permuted_from_opening_point`].
//! The output expression is bypassed (not the `derive_output_term` path) because the
//! produced opening id is dynamic in `has_address_phase`; the override computes
//! exactly the formula value, so the clear path and BlindFold stay in sync.

use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::{
    lane_weights, BytecodeLaneWeightInputs,
};
use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::claim_reductions::bytecode::{
    BytecodeReductionCyclePhaseChallenges, BytecodeReductionCyclePhaseInputClaims,
    BytecodeReductionCyclePhaseOutputClaims,
};
pub use jolt_claims::protocols::jolt::relations::claim_reductions::program_image::{
    ProgramImageReductionCyclePhaseInputClaims, ProgramImageReductionCyclePhaseOutputClaims,
};
use jolt_claims::protocols::jolt::{
    BytecodeClaimReductionLayout, JoltRelationId, PrecommittedReductionLayout,
    ProgramImageClaimReductionLayout,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::JoltField;

use super::outputs::BytecodeReductionWeights;
use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage4::RamValCheckInitialEvaluation;
use crate::VerifierError;

/// Wire the consumed RAM value-check program-image contribution *value*.
/// Clear-only.
pub fn program_image_reduction_cycle_phase_input_values_from_upstream<F: JoltField>(
    ram_val_check_init: &RamValCheckInitialEvaluation<F>,
) -> Result<ProgramImageReductionCyclePhaseInputClaims<F>, VerifierError> {
    let (_, value) = ram_val_check_init
        .program_image_contribution
        .as_ref()
        .ok_or_else(|| {
            program_image_public_failed("missing RAM value-check program-image contribution")
        })?;
    Ok(ProgramImageReductionCyclePhaseInputClaims {
        contribution: *value,
    })
}

#[derive(Clone)]
pub struct ProgramImageReductionCyclePhase<F: JoltField> {
    symbolic: relations::claim_reductions::program_image::CyclePhase,
    layout: ProgramImageClaimReductionLayout,
    /// The RAM address component of the `RamVal` opening from RAM read-write
    /// checking; the `FinalScale` public compares the produced opening point
    /// against it.
    r_addr_rw: Vec<F>,
}

impl<F: JoltField> ProgramImageReductionCyclePhase<F> {
    pub fn new(layout: &ProgramImageClaimReductionLayout, r_addr_rw: Vec<F>) -> Self {
        Self {
            symbolic: relations::claim_reductions::program_image::CyclePhase::new(
                layout.dimensions(),
            ),
            layout: layout.clone(),
            r_addr_rw,
        }
    }

    pub fn layout(&self) -> &ProgramImageClaimReductionLayout {
        &self.layout
    }

    /// The RAM address component of the stage-2 `RamVal` opening the kernel's
    /// shifted eq slice binds against.
    pub fn r_addr_rw(&self) -> &[F] {
        &self.r_addr_rw
    }
}

fn program_image_public_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::ProgramImageClaimReductionCyclePhase,
        reason: reason.to_string(),
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for ProgramImageReductionCyclePhase<F> {
    type Symbolic = relations::claim_reductions::program_image::CyclePhase;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    /// Precommitted cycle-phase reductions are bound on the offset-0 prefix of
    /// the batch challenge vector, not the front-loaded suffix.
    fn instance_point_offset(&self, _batch_num_vars: usize) -> Result<usize, VerifierError> {
        Ok(0)
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &ProgramImageReductionCyclePhaseInputClaims<Vec<F>>,
    ) -> Result<ProgramImageReductionCyclePhaseOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .layout
            .cycle_phase_opening_point(sumcheck_point)
            .map_err(program_image_public_failed)?;
        Ok(ProgramImageReductionCyclePhaseOutputClaims {
            program_image: opening_point,
        })
    }

    fn expected_output(
        &self,
        _input_points: &ProgramImageReductionCyclePhaseInputClaims<Vec<F>>,
        output_values: &ProgramImageReductionCyclePhaseOutputClaims<F>,
        output_points: &ProgramImageReductionCyclePhaseOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let value = output_values.program_image;
        if self.layout.dimensions().has_address_phase() {
            Ok(value)
        } else {
            let scale = self
                .layout
                .cycle_phase_scale_at_opening_point(&self.r_addr_rw, output_points.program_image())
                .map_err(program_image_public_failed)?;
            Ok(scale * value)
        }
    }
}

#[derive(Clone)]
pub struct BytecodeReductionCyclePhase<F: JoltField> {
    symbolic: relations::claim_reductions::bytecode::CyclePhase,
    layout: BytecodeClaimReductionLayout,
    weights: BytecodeReductionWeights<F>,
    chunk_count: usize,
}

impl<F: JoltField> BytecodeReductionCyclePhase<F> {
    pub fn new(
        layout: &BytecodeClaimReductionLayout,
        weights: BytecodeReductionWeights<F>,
    ) -> Self {
        Self {
            symbolic: relations::claim_reductions::bytecode::CyclePhase::new((
                layout.dimensions(),
                layout.chunk_count(),
            )),
            layout: layout.clone(),
            weights,
            chunk_count: layout.chunk_count(),
        }
    }

    /// The public bytecode claim-reduction weights this member was built
    /// with. Pub: the prove-side recipe reads them back off the
    /// `build_from_parts` batch (they also feed the bytecode reduction kernel
    /// and ride the clear carrier), so the weight fold is single-sourced.
    /// Stage 7's bytecode address phase reads them off the stage-6b clear output
    /// rather than recomputing them.
    pub fn weights(&self) -> &BytecodeReductionWeights<F> {
        &self.weights
    }

    pub fn layout(&self) -> &BytecodeClaimReductionLayout {
        &self.layout
    }
}

/// Fold the stage-6a bytecode read-RAF address opening and the per-stage gamma
/// vectors into the public [`BytecodeReductionWeights`] (the per-chunk `r_bc`
/// weights and the gamma-folded lane weights) consumed by the bytecode
/// claim-reduction cycle and address phases.
pub fn bytecode_reduction_weights<F: JoltField>(
    layout: &BytecodeClaimReductionLayout,
    lane_inputs: BytecodeLaneWeightInputs<'_, F>,
    bytecode_r_address: &[F],
) -> Result<BytecodeReductionWeights<F>, VerifierError> {
    let address_point = layout
        .split_address_point(bytecode_r_address)
        .map_err(bytecode_public_failed)?;
    let lane_weights = lane_weights(lane_inputs).map_err(bytecode_public_failed)?;
    Ok(BytecodeReductionWeights {
        r_bc: address_point.r_bc,
        chunk_rbc_weights: address_point.chunk_rbc_weights,
        lane_weights,
    })
}

fn bytecode_public_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeClaimReductionCyclePhase,
        reason: reason.to_string(),
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for BytecodeReductionCyclePhase<F> {
    type Symbolic = relations::claim_reductions::bytecode::CyclePhase;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    /// Precommitted cycle-phase reductions are bound on the offset-0 prefix of
    /// the batch challenge vector, not the front-loaded suffix.
    fn instance_point_offset(&self, _batch_num_vars: usize) -> Result<usize, VerifierError> {
        Ok(0)
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &BytecodeReductionCyclePhaseInputClaims<Vec<F>>,
    ) -> Result<BytecodeReductionCyclePhaseOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .layout
            .cycle_phase_opening_point(sumcheck_point)
            .map_err(bytecode_public_failed)?;
        Ok(if self.layout.dimensions().has_address_phase() {
            BytecodeReductionCyclePhaseOutputClaims {
                intermediate: Some(opening_point),
                chunks: Vec::new(),
            }
        } else {
            BytecodeReductionCyclePhaseOutputClaims {
                intermediate: None,
                chunks: vec![opening_point; self.chunk_count],
            }
        })
    }

    fn expected_output(
        &self,
        _input_points: &BytecodeReductionCyclePhaseInputClaims<Vec<F>>,
        output_values: &BytecodeReductionCyclePhaseOutputClaims<F>,
        output_points: &BytecodeReductionCyclePhaseOutputClaims<Vec<F>>,
        _challenges: &BytecodeReductionCyclePhaseChallenges<F>,
    ) -> Result<F, VerifierError> {
        if self.layout.dimensions().has_address_phase() {
            let intermediate = output_values.intermediate.ok_or_else(|| {
                bytecode_public_failed("bytecode reduction produced no intermediate")
            })?;
            return Ok(intermediate);
        }
        if output_values.chunks.len() != self.chunk_count {
            return Err(bytecode_public_failed(format!(
                "bytecode chunk claim count mismatch: expected {}, got {}",
                self.chunk_count,
                output_values.chunks.len()
            )));
        }
        let opening_point = output_points
            .chunks()
            .first()
            .map(Vec::as_slice)
            .ok_or_else(|| {
                bytecode_public_failed("bytecode reduction produced no chunk openings")
            })?;
        let weights = self
            .layout
            .cycle_phase_final_output_weights_at_opening_point(
                self.weights.as_inputs(),
                opening_point,
            )
            .map_err(bytecode_public_failed)?;
        Ok(output_values
            .chunks
            .iter()
            .zip(weights)
            .map(|(chunk, weight)| *chunk * weight)
            .sum())
    }
}

//! The address phase of the bytecode read-RAF symbolic sumcheck.

use jolt_field::Ring;
use jolt_wasm_ir::RowFlag;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::bytecode::{
    bytecode_read_raf_address_phase_opening, read_raf_address_input_fold,
    BytecodeReadRafDimensions, BYTECODE_STAGE_GAMMA_COUNTS,
};
use crate::protocols::jolt::{
    BytecodeReadRafChallenge, JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId,
    JoltRelationId,
};
use crate::{opening, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck};

/// The address-phase produced openings: the `BytecodeReadRafAddrClaim`
/// intermediate, plus (committed-program mode only) the staged `BytecodeValClaim`
/// openings. In full-program mode `val_stages` is empty.
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(BytecodeReadRaf)]
pub struct BytecodeReadRafAddressPhaseOutputClaims<C> {
    #[opening(BytecodeReadRafAddrClaim)]
    pub intermediate: C,
    #[opening(BytecodeValClaim)]
    pub val_stages: Vec<C>,
}

/// The prior-proof openings the address-phase input claim binds: every stage-1..5
/// opening the `read_raf_address_phase` input `Expr` folds (plus the two PC
/// claims). The generic `input_claim` evaluates the bind from these via that
/// `Expr`, so the gamma-folding formula lives in one place rather than a
/// hand-written resolver. Each Spartan-outer row flag is its own field (the
/// `OuterRemainderOutputClaims` idiom), in `RowFlag::ALL` order; the
/// `lookup_table_flags` family is indexed (`LookupTableFlag(i)`).
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct BytecodeReadRafAddressPhaseInputClaims<C> {
    #[opening(Imm, from = SpartanOuter)]
    pub outer_imm: C,
    #[opening(RowFlag(RowFlag::LeftIsRs1), from = SpartanOuter)]
    pub outer_left_is_rs1: C,
    #[opening(RowFlag(RowFlag::RightIsRs2), from = SpartanOuter)]
    pub outer_right_is_rs2: C,
    #[opening(RowFlag(RowFlag::RightIsImm), from = SpartanOuter)]
    pub outer_right_is_imm: C,
    #[opening(RowFlag(RowFlag::AddOperands), from = SpartanOuter)]
    pub outer_add_operands: C,
    #[opening(RowFlag(RowFlag::SubOperands), from = SpartanOuter)]
    pub outer_sub_operands: C,
    #[opening(RowFlag(RowFlag::MulOperands), from = SpartanOuter)]
    pub outer_mul_operands: C,
    #[opening(RowFlag(RowFlag::WriteLookupToRd), from = SpartanOuter)]
    pub outer_write_lookup_to_rd: C,
    #[opening(RowFlag(RowFlag::Load), from = SpartanOuter)]
    pub outer_load: C,
    #[opening(RowFlag(RowFlag::Store), from = SpartanOuter)]
    pub outer_store: C,
    #[opening(RowFlag(RowFlag::Jump), from = SpartanOuter)]
    pub outer_jump: C,
    #[opening(RowFlag(RowFlag::Branch), from = SpartanOuter)]
    pub outer_branch: C,
    #[opening(RowFlag(RowFlag::Assert), from = SpartanOuter)]
    pub outer_assert: C,
    #[opening(RowFlag(RowFlag::Halt), from = SpartanOuter)]
    pub outer_halt: C,
    #[opening(RowFlag(RowFlag::Trap), from = SpartanOuter)]
    pub outer_trap: C,
    #[opening(RowFlag(RowFlag::Advice), from = SpartanOuter)]
    pub outer_advice: C,
    #[opening(PC, from = SpartanOuter)]
    pub outer_pc: C,
    #[opening(RowFlag(RowFlag::Branch), from = SpartanProductVirtualization)]
    pub product_branch: C,
    #[opening(Imm, from = InstructionInputVirtualization)]
    pub instruction_input_imm: C,
    #[opening(RowFlag(RowFlag::LeftIsRs1), from = InstructionInputVirtualization)]
    pub left_operand_is_rs1: C,
    #[opening(RowFlag(RowFlag::RightIsRs2), from = InstructionInputVirtualization)]
    pub right_operand_is_rs2: C,
    #[opening(RowFlag(RowFlag::RightIsImm), from = InstructionInputVirtualization)]
    pub right_operand_is_imm: C,
    #[opening(PC, from = SpartanShift)]
    pub shift_pc: C,
    #[opening(RdWa, from = RegistersReadWriteChecking)]
    pub rd_wa_read_write: C,
    #[opening(Rs1Ra, from = RegistersReadWriteChecking)]
    pub rs1_ra: C,
    #[opening(Rs2Ra, from = RegistersReadWriteChecking)]
    pub rs2_ra: C,
    #[opening(RdWa, from = RegistersValEvaluation)]
    pub rd_wa_val_evaluation: C,
    #[opening(InstructionRafFlag, from = InstructionReadRaf)]
    pub instruction_raf_flag: C,
    #[opening(LookupTableFlag, from = InstructionReadRaf)]
    pub lookup_table_flags: Vec<C>,
}

/// Fiat-Shamir challenges drawn by the address phase of the bytecode read-RAF
/// sumcheck: the batching `gamma` plus the five per-stage gammas (the same set
/// the full monolith folds).
#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct BytecodeReadRafAddressPhaseChallenges<F> {
    #[challenge(BytecodeReadRafChallenge::Gamma)]
    pub gamma: F,
    #[challenge(BytecodeReadRafChallenge::Stage1Gamma)]
    pub stage1_gamma: F,
    #[challenge(BytecodeReadRafChallenge::Stage2Gamma)]
    pub stage2_gamma: F,
    #[challenge(BytecodeReadRafChallenge::Stage3Gamma)]
    pub stage3_gamma: F,
    #[challenge(BytecodeReadRafChallenge::Stage4Gamma)]
    pub stage4_gamma: F,
    #[challenge(BytecodeReadRafChallenge::Stage5Gamma)]
    pub stage5_gamma: F,
}

impl<F: jolt_field::JoltField> BytecodeReadRafAddressPhaseChallenges<F> {
    /// Expand the five drawn per-stage scalars into the gamma-power vectors the
    /// bytecode folds consume (`[1, γ, γ², …]` — the recurrence the prover's
    /// `challenge_scalar_powers` applies to its single squeezed scalar), sized
    /// by [`BYTECODE_STAGE_GAMMA_COUNTS`].
    pub fn stage_gamma_powers(&self) -> [Vec<F>; 5] {
        let stage_gammas = [
            self.stage1_gamma,
            self.stage2_gamma,
            self.stage3_gamma,
            self.stage4_gamma,
            self.stage5_gamma,
        ];
        core::array::from_fn(|stage| {
            let mut powers = vec![F::one(); BYTECODE_STAGE_GAMMA_COUNTS[stage]];
            for index in 1..powers.len() {
                powers[index] = powers[index - 1] * stage_gammas[stage];
            }
            powers
        })
    }
}

/// The address phase of the bytecode read-RAF sumcheck: the same folded input
/// claim, reduced to the staged address-phase opening.
#[derive(Clone)]
pub struct ReadRafAddressPhase {
    shape: BytecodeReadRafDimensions,
}

impl SymbolicSumcheck for ReadRafAddressPhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReadRafDimensions;
    type Challenges<F> = BytecodeReadRafAddressPhaseChallenges<F>;
    type Inputs<C> = BytecodeReadRafAddressPhaseInputClaims<C>;
    type Outputs<C> = BytecodeReadRafAddressPhaseOutputClaims<C>;

    fn new(shape: BytecodeReadRafDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeReadRaf
    }

    fn rounds(&self) -> usize {
        self.shape.log_k()
    }

    fn degree(&self) -> usize {
        self.shape.num_committed_ra_polys() + 1
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        read_raf_address_input_fold(Vec::new())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(bytecode_read_raf_address_phase_opening())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;

    fn dimensions(num_committed_ra_polys: usize) -> BytecodeReadRafDimensions {
        BytecodeReadRafDimensions::new(5, 10, num_committed_ra_polys)
    }

    #[test]
    fn read_raf_address_phase_symbolic_matches_dependencies() {
        let relation = ReadRafAddressPhase::new(dimensions(2));
        assert_eq!(ReadRafAddressPhase::id(), JoltRelationId::BytecodeReadRaf);
        assert_eq!(relation.rounds(), dimensions(2).log_k());
        assert_eq!(
            relation.degree(),
            dimensions(2).num_committed_ra_polys() + 1
        );
    }

    /// Pins the row-flag coverage of the input claims struct: every `RowFlag`
    /// has a `SpartanOuter` field (a newly added flag missing its field would
    /// make the input `Expr` reference an unresolvable opening).
    #[test]
    fn input_claims_cover_row_flags() {
        let claims = BytecodeReadRafAddressPhaseInputClaims::<Fr>::default();
        for flag in RowFlag::ALL {
            let outer = JoltOpeningId::virtual_polynomial(
                crate::protocols::jolt::JoltVirtualPolynomial::RowFlag(flag),
                JoltRelationId::SpartanOuter,
            );
            assert!(
                claims.resolve_input(&outer).is_some(),
                "missing SpartanOuter input field for OpFlags({flag:?})",
            );
        }
    }
}

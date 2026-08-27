//! Spartan shift symbolic sumcheck relation.

use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::spartan::{next_pc_outer, pc_shift, SHIFT_DEGREE};
use crate::protocols::jolt::{
    JoltExpr, JoltRelationId, SpartanShiftChallenge, SpartanShiftPublic, TraceDimensions,
};
use crate::{derived, opening, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck};

/// Produced Spartan shift openings (the shifted unexpanded-PC / PC / virtual /
/// first-in-sequence / noop columns), all sharing the single shift opening point.
/// Generic over the cell.
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(SpartanShift)]
pub struct SpartanShiftOutputClaims<C> {
    #[opening(PC)]
    pub pc: C,
}

/// Consumed shift openings: the `Next*` PC/flag columns from stage 1's outer
/// sumcheck and `next_is_noop` from stage 2's product remainder. Shift reads only
/// these values, so the input points are left empty. Generic over the cell.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct SpartanShiftInputClaims<C> {
    #[opening(NextPC, from = SpartanOuter)]
    pub next_pc: C,
}

/// Fiat-Shamir challenge drawn by the Spartan shift sumcheck.
#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct SpartanShiftChallenges<F> {
    #[challenge(SpartanShiftChallenge::Gamma)]
    pub gamma: F,
}

/// The Spartan shift sumcheck: relates each `Next*` column from the outer
/// sumcheck (and `next_is_noop` from the product remainder) to the shifted
/// column at the same cycle, folded by `gamma` and weighted by the `EqPlusOne`
/// publics.
#[derive(Clone)]
pub struct Shift {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for Shift {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = TraceDimensions;
    type Challenges<F> = SpartanShiftChallenges<F>;
    type Inputs<C> = SpartanShiftInputClaims<C>;
    type Outputs<C> = SpartanShiftOutputClaims<C>;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::SpartanShift
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        SHIFT_DEGREE
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(next_pc_outer())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        derived(SpartanShiftPublic::EqPlusOneOuter) * opening(pc_shift())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::{JoltChallengeId, JoltDerivedId};
    use jolt_field::{Fr, Ring};

    #[test]
    fn shift_evaluates_like_core_formula() {
        let relation = Shift::new(TraceDimensions::new(5));

        let next_pc = Fr::from_u64(5);
        let pc = Fr::from_u64(19);
        let gamma = Fr::from_u64(37);
        let eq_outer = Fr::from_u64(41);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == next_pc_outer() => next_pc,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::SpartanShift(SpartanShiftChallenge::Gamma) => gamma,
                _ => zero,
            },
            |_| zero,
        );
        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == pc_shift() => pc,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::SpartanShift(SpartanShiftChallenge::Gamma) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltDerivedId::SpartanShift(SpartanShiftPublic::EqPlusOneOuter) => eq_outer,
                _ => zero,
            },
        );

        assert_eq!(input, next_pc);
        assert_eq!(output, eq_outer * pc);
    }

    #[test]
    fn shift_symbolic_matches_dependencies() {
        let relation = Shift::new(TraceDimensions::new(5));
        assert_eq!(Shift::id(), JoltRelationId::SpartanShift);
        assert_eq!(relation.rounds(), TraceDimensions::new(5).log_t());
        assert_eq!(relation.degree(), SHIFT_DEGREE);
    }
}

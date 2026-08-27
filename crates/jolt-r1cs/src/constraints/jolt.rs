//! Compile-time Jolt R1CS composition.

use jolt_field::JoltField;
use jolt_poly::{
    lagrange::{centered_lagrange_evals, centered_lagrange_kernel, CenteredIntegerDomainError},
    EqPolynomial,
};
use thiserror::Error as ThisError;

use crate::{ConstraintMatrices, ConstraintMatrixEvalError};

use super::wasm;

pub const NUM_VARS_PER_CYCLE: usize = wasm::NUM_VARS_PER_CYCLE;

pub const NUM_CONSTRAINTS_PER_CYCLE: usize = wasm::NUM_CONSTRAINTS_PER_CYCLE;

pub const SPARTAN_OUTER_ROW_COUNT: usize = wasm::NUM_EQ_CONSTRAINTS;

pub const SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE: usize = SPARTAN_OUTER_ROW_COUNT.div_ceil(2);
pub const SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE: usize =
    3 * SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE - 3;
pub const SPARTAN_OUTER_REMAINDER_DEGREE: usize = 3;
pub const SPARTAN_OUTER_SECOND_GROUP_ROW_COUNT: usize =
    SPARTAN_OUTER_ROW_COUNT - SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE;
pub const SPARTAN_PRODUCT_BASE_LANES: usize = 2;

pub const SPARTAN_PRODUCT_FIELD_INLINE_LANES: usize = 0;

pub const SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE: usize =
    SPARTAN_PRODUCT_BASE_LANES + SPARTAN_PRODUCT_FIELD_INLINE_LANES;
pub const SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE: usize =
    3 * (SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE - 1);

pub const SPARTAN_OUTER_FIRST_GROUP_ROWS: [usize; SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE] =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

pub const SPARTAN_OUTER_SECOND_GROUP_ROWS: [usize; SPARTAN_OUTER_SECOND_GROUP_ROW_COUNT] =
    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];

pub fn spartan_outer_constraints<F: JoltField>() -> ConstraintMatrices<F> {
    let constraints = wasm::wasm_spartan_outer_constraints();
    {
        constraints
    }
}

pub fn trace_constraints<F: JoltField>() -> ConstraintMatrices<F> {
    let constraints = wasm::wasm_trace_constraints();
    {
        constraints
    }
}

pub fn spartan_outer_row_weights<F: JoltField>(
    uniskip: F,
    stream: F,
) -> Result<Vec<F>, CenteredIntegerDomainError> {
    let lagrange_weights = centered_lagrange_evals(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, uniskip)?;
    // The row-group arrays are typed to the domain size, so only a short
    // weight vector could make the zips below drop rows silently.
    debug_assert_eq!(lagrange_weights.len(), SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE);
    let mut weights = vec![F::zero(); SPARTAN_OUTER_ROW_COUNT];

    #[expect(
        clippy::indexing_slicing,
        reason = "SPARTAN_OUTER_FIRST_GROUP_ROWS entries are compile-time constants below SPARTAN_OUTER_ROW_COUNT"
    )]
    for (&row, &lagrange_weight) in SPARTAN_OUTER_FIRST_GROUP_ROWS.iter().zip(&lagrange_weights) {
        weights[row] += (F::one() - stream) * lagrange_weight;
    }
    #[expect(
        clippy::indexing_slicing,
        reason = "SPARTAN_OUTER_SECOND_GROUP_ROWS entries are compile-time constants below SPARTAN_OUTER_ROW_COUNT"
    )]
    for (&row, &lagrange_weight) in SPARTAN_OUTER_SECOND_GROUP_ROWS
        .iter()
        .zip(&lagrange_weights)
    {
        weights[row] += stream * lagrange_weight;
    }

    Ok(weights)
}

pub fn spartan_outer_opening_columns() -> Vec<usize> {
    (0..wasm::NUM_R1CS_INPUTS)
        .map(|index| wasm::V_LEFT_INPUT + index)
        .collect::<Vec<_>>()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoltSpartanOuterPublic {
    TauKernel,
    AzWeight(usize),
    BzWeight(usize),
    AzConstant,
    BzConstant,
}

#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum JoltSpartanOuterRemainderError {
    #[error("missing Spartan outer remainder stream challenge")]
    MissingStreamChallenge,
    #[error("{0}")]
    InvalidUniskipDomain(#[from] CenteredIntegerDomainError),
    #[error("challenge length mismatch: expected {expected}, got {got}")]
    ChallengeLengthMismatch { expected: usize, got: usize },
    #[error("{0}")]
    Matrix(#[from] ConstraintMatrixEvalError),
    #[error("opening length mismatch: expected {expected}, got {got}")]
    OpeningLengthMismatch { expected: usize, got: usize },
    #[error("Spartan outer rows unexpectedly contribute to the C linear form")]
    UnexpectedCContribution,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JoltSpartanOuterRemainder<F: JoltField> {
    tau_kernel: F,
    az_coefficients: Vec<F>,
    bz_coefficients: Vec<F>,
    az_constant: F,
    bz_constant: F,
}

#[derive(Clone, Copy, Debug)]
pub struct JoltSpartanOuterRemainderChallenges<'a, F> {
    pub tau: &'a [F],
    pub uniskip: F,
    pub remainder: &'a [F],
}

impl<F: JoltField> JoltSpartanOuterRemainder<F> {
    pub fn new(
        challenges: JoltSpartanOuterRemainderChallenges<'_, F>,
    ) -> Result<Self, JoltSpartanOuterRemainderError> {
        let Some((&r_stream, _)) = challenges.remainder.split_first() else {
            return Err(JoltSpartanOuterRemainderError::MissingStreamChallenge);
        };

        let row_weights = spartan_outer_row_weights(challenges.uniskip, r_stream)?;
        let columns = spartan_outer_opening_columns();
        let matrices = spartan_outer_constraints::<F>();
        let weighted = matrices.weighted_columns(&row_weights, &columns)?;
        if weighted.c.iter().any(|coefficient| !coefficient.is_zero()) {
            return Err(JoltSpartanOuterRemainderError::UnexpectedCContribution);
        }

        let constant_contributions =
            matrices.public_column_contributions(&row_weights, wasm::const_column(), F::one())?;
        if !constant_contributions.c.is_zero() {
            return Err(JoltSpartanOuterRemainderError::UnexpectedCContribution);
        }

        Ok(Self {
            tau_kernel: spartan_outer_tau_kernel(
                challenges.tau,
                challenges.uniskip,
                challenges.remainder,
            )?,
            az_coefficients: weighted.a,
            bz_coefficients: weighted.b,
            az_constant: constant_contributions.a,
            bz_constant: constant_contributions.b,
        })
    }

    pub fn expected_output_claim(
        &self,
        openings: &[F],
    ) -> Result<F, JoltSpartanOuterRemainderError> {
        let expected = self.az_coefficients.len();
        if openings.len() != expected {
            return Err(JoltSpartanOuterRemainderError::OpeningLengthMismatch {
                expected,
                got: openings.len(),
            });
        }

        Ok(self.tau_kernel
            * eval_linear_form(&self.az_coefficients, self.az_constant, openings)
            * eval_linear_form(&self.bz_coefficients, self.bz_constant, openings))
    }

    pub fn public_coefficients(&self) -> Vec<(JoltSpartanOuterPublic, F)> {
        let count = self.az_coefficients.len();
        let mut coefficients = Vec::with_capacity(2 * count + 3);
        coefficients.push((JoltSpartanOuterPublic::TauKernel, self.tau_kernel));
        for (index, &weight) in self.az_coefficients.iter().enumerate() {
            coefficients.push((JoltSpartanOuterPublic::AzWeight(index), weight));
        }
        for (index, &weight) in self.bz_coefficients.iter().enumerate() {
            coefficients.push((JoltSpartanOuterPublic::BzWeight(index), weight));
        }
        coefficients.push((JoltSpartanOuterPublic::AzConstant, self.az_constant));
        coefficients.push((JoltSpartanOuterPublic::BzConstant, self.bz_constant));
        coefficients
    }
}

fn spartan_outer_tau_kernel<F: JoltField>(
    tau: &[F],
    uniskip: F,
    remainder_challenges: &[F],
) -> Result<F, JoltSpartanOuterRemainderError> {
    let expected = remainder_challenges.len() + 1;
    if tau.len() != expected {
        return Err(JoltSpartanOuterRemainderError::ChallengeLengthMismatch {
            expected,
            got: tau.len(),
        });
    }

    // `tau` is non-empty: `tau.len() == expected >= 1` is checked above.
    let Some((&tau_high, tau_low)) = tau.split_last() else {
        return Err(JoltSpartanOuterRemainderError::ChallengeLengthMismatch { expected, got: 0 });
    };
    let tau_high_bound_r0 =
        centered_lagrange_kernel(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, tau_high, uniskip)?;
    let mut reversed_challenges = remainder_challenges.to_vec();
    reversed_challenges.reverse();
    Ok(tau_high_bound_r0 * EqPolynomial::<F>::mle(tau_low, &reversed_challenges))
}

fn eval_linear_form<F: JoltField>(coefficients: &[F], constant: F, inputs: &[F]) -> F {
    coefficients
        .iter()
        .zip(inputs)
        .fold(constant, |acc, (&coefficient, &input)| {
            acc + coefficient * input
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, Ring};

    #[test]
    fn default_composed_constraints_match_wasm_shape() {
        let composed = trace_constraints::<Fr>();
        let wasm = wasm::wasm_trace_constraints::<Fr>();

        assert_eq!(composed.num_constraints, wasm.num_constraints);
        assert_eq!(composed.num_vars, wasm.num_vars);
        assert_eq!(composed.a, wasm.a);
        assert_eq!(composed.b, wasm.b);
        assert_eq!(composed.c, wasm.c);
    }

    #[test]
    fn default_spartan_outer_geometry_matches_wasm() {
        assert_eq!(SPARTAN_OUTER_ROW_COUNT, wasm::NUM_EQ_CONSTRAINTS);
        assert_eq!(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, 11);
        assert_eq!(SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE, 30);
        assert_eq!(SPARTAN_OUTER_REMAINDER_DEGREE, 3);
        assert_eq!(
            SPARTAN_OUTER_FIRST_GROUP_ROWS,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        );
        assert_eq!(
            SPARTAN_OUTER_SECOND_GROUP_ROWS,
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        );
        assert_eq!(
            spartan_outer_row_weights(Fr::from_u64(2), Fr::from_u64(3))
                .map(|weights| weights.len()),
            Ok(wasm::NUM_EQ_CONSTRAINTS)
        );
        assert_eq!(
            spartan_outer_opening_columns(),
            (wasm::V_LEFT_INPUT..=wasm::NUM_R1CS_INPUTS).collect::<Vec<_>>()
        );
    }
}

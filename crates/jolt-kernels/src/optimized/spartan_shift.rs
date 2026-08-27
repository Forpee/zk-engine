//! Optimized stage-3 Spartan shift kernel.
//!
//! The summand is `eq⁺¹(τ_low, j) · pc(j)` over the cycle domain: the shift
//! lives entirely in the `eq+1` factor (the pc table is unshifted). Two
//! phases, mirroring the legacy prover:
//!
//! - **Prefix–suffix rounds** (the first half of the cycle variables): the
//!   `eq+1` table factors as `P₀(y_hi)·S₀(y_lo) + P₁(y_hi)·S₁(y_lo)`
//!   ([`EqPlusOnePrefixSuffix`]), so the round messages are inner products
//!   of the bound prefix tables against `Q_b(y_lo) = Σ_{y_hi} S_b(y_hi) ·
//!   pc(y_hi ‖ y_lo)` — built once from the native rows with the u32-split
//!   accumulation (`fmadd_u64_split`), no per-cycle field conversion.
//! - **Dense rounds**: on entering the second half the kernel regenerates
//!   the `eq+1` table at its bound prefix and folds the pc column by
//!   `eq(r_prefix)` (its exact partial bind), then binds both densely.

use jolt_claims::protocols::jolt::{JoltDerivedId, SpartanShiftPublic};
use jolt_field::{Accumulator, JoltField};
use jolt_poly::{EqPlusOnePrefixSuffix, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage3::spartan_shift::{SpartanShift, SpartanShiftOutputClaims};
use jolt_witness::witnesses::Pc;
use jolt_witness::{JoltWitnessPlane, WitnessBundle};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::{
    bind_pairs, collect_rows, fmadd_u64_split, pin_derived_term, RoundChallenges,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// The per-cycle shift column as a native scalar: the pc at cycle `j`,
/// unshifted (the shift lives in the `eq+1` factor).
#[derive(Clone, Copy, Debug, WitnessBundle)]
struct SpartanShiftRow {
    #[opening(PC)]
    pc: Pc,
}

pub struct OptimizedSpartanShift;

impl<F: JoltField> PrepareKernel<F, SpartanShift<F>> for OptimizedSpartanShift {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, SpartanShift<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = SpartanShift<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let log_t = relation.rounds();
        if log_t == 0 {
            return Err(KernelError::Unsupported {
                reason: "optimized Spartan shift requires at least one cycle round",
            });
        }
        let r_outer: &[F] = relation.product_uniskip_tau_low();
        if r_outer.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan shift eq+1 point has the wrong variable count",
            });
        }
        let cycles = 1usize << log_t;
        let rows: Vec<SpartanShiftRow> = collect_rows(witness, cycles)?;

        let outer = EqPlusOnePrefixSuffix::new(r_outer);
        let prefix_vars = outer.prefix_0.len().trailing_zeros() as usize;

        // Q_b(y_lo) = Σ_{y_hi} S_b(y_hi) · pc(y_hi ‖ y_lo).
        const BLOCK: usize = 32;
        let suffix_rows: Vec<[F; 2]> = (0..outer.suffix_0.len())
            .map(|x_hi| [outer.suffix_0[x_hi], outer.suffix_1[x_hi]])
            .collect();
        let build_q_block = |(block_index, q_block): (usize, &mut [[F; 2]])| {
            let mut folds = vec![[F::Accumulator::default(); 2]; q_block.len()];
            for (x_hi, suffix) in suffix_rows.iter().enumerate() {
                let base = x_hi << prefix_vars;
                for (i, fold) in folds.iter_mut().enumerate() {
                    let x_lo = block_index * BLOCK + i;
                    let v = F::from_u64(rows[base + x_lo].pc.0);
                    fold[0].fmadd(v, suffix[0]);
                    fold[1].fmadd(v, suffix[1]);
                }
            }
            for (q, fold) in q_block.iter_mut().zip(folds) {
                q[0] = fold[0].reduce();
                q[1] = fold[1].reduce();
            }
        };
        let mut q_rows = vec![[F::zero(); 2]; 1 << prefix_vars];
        #[cfg(feature = "parallel")]
        q_rows
            .par_chunks_mut(BLOCK)
            .enumerate()
            .for_each(build_q_block);
        #[cfg(not(feature = "parallel"))]
        q_rows.chunks_mut(BLOCK).enumerate().for_each(build_q_block);

        let [q_0, q_1]: [Vec<F>; 2] =
            core::array::from_fn(|pair| q_rows.iter().map(|q| q[pair]).collect());
        let pairs = [(outer.prefix_0, q_0), (outer.prefix_1, q_1)];

        Ok(Box::new(ShiftKernel {
            log_t,
            r_outer: r_outer.to_vec(),
            rows,
            phase: Phase::PrefixSuffix { pairs },
            challenges: RoundChallenges::new(log_t),
        }))
    }
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F")
)]
enum Phase<F> {
    /// First half of the rounds: the two `(P, Q)` pairs over the prefix
    /// variables.
    PrefixSuffix {
        #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_scalar_pairs))]
        pairs: [(Vec<F>, Vec<F>); 2],
    },
    /// Remaining rounds: the `eq+1` table and the pc column, dense.
    Dense {
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
        eq_plus_one_outer: Vec<F>,
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
        pc: Vec<F>,
    },
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct ShiftKernel<F: JoltField> {
    log_t: usize,
    /// The `eq+1` point (big-endian) the summand factor fixes.
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    r_outer: Vec<F>,
    /// Raw per-cycle values, kept for the phase-2 regeneration.
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    rows: Vec<SpartanShiftRow>,
    phase: Phase<F>,
    challenges: RoundChallenges<F>,
}
impl<F: JoltField> ShiftKernel<F> {
    /// Regenerate the dense phase from the raw values: the pc column folded
    /// by `eq(r_prefix)` (its exact partial bind) and the `eq+1` table
    /// recombined from its suffix pair and bound-prefix evaluations.
    fn transition_to_dense(&mut self) {
        let bound = self.challenges.bound();
        let r_prefix: Vec<F> = self.challenges.as_slice().iter().rev().copied().collect();
        let eq_prefix = EqPolynomial::<F>::evals(&r_prefix, None);
        let eq_prefix_shifted: Vec<F> = eq_prefix.iter().map(|eq| eq.mul_pow_2(32)).collect();
        let chunk = eq_prefix.len();
        let remaining = 1usize << (self.log_t - bound);

        let fold_chunk = |rows: &[SpartanShiftRow]| -> F {
            let mut fold = F::SmallScalarAccumulator::default();
            for (row, (&eq, &eq_shifted)) in rows
                .iter()
                .zip(eq_prefix.iter().zip(eq_prefix_shifted.iter()))
            {
                fmadd_u64_split(&mut fold, eq, eq_shifted, row.pc.0);
            }
            fold.reduce()
        };
        #[cfg(feature = "parallel")]
        let pc: Vec<F> = self.rows.par_chunks(chunk).map(fold_chunk).collect();
        #[cfg(not(feature = "parallel"))]
        let pc: Vec<F> = self.rows.chunks(chunk).map(fold_chunk).collect();
        debug_assert_eq!(pc.len(), remaining);

        // The raw values only feed this regeneration; free them now.
        self.rows = Vec::new();

        let split = EqPlusOnePrefixSuffix::new(&self.r_outer);
        let prefix_0_eval = Polynomial::new(split.prefix_0).evaluate(&r_prefix);
        let prefix_1_eval = Polynomial::new(split.prefix_1).evaluate(&r_prefix);
        let eq_plus_one_outer = split
            .suffix_0
            .iter()
            .zip(split.suffix_1.iter())
            .map(|(&suffix_0, &suffix_1)| prefix_0_eval * suffix_0 + prefix_1_eval * suffix_1)
            .collect();

        self.phase = Phase::Dense {
            eq_plus_one_outer,
            pc,
        };
    }

    fn bind(&mut self, r: F) {
        self.challenges.push(r);
        // Last prefix variable: regenerate the dense phase from the raw
        // values instead of binding the exhausted P·Q pairs.
        if matches!(&self.phase, Phase::PrefixSuffix { pairs } if pairs[0].0.len() == 2) {
            self.transition_to_dense();
            return;
        }
        match &mut self.phase {
            Phase::PrefixSuffix { pairs } => {
                for (p, q) in pairs {
                    bind_pairs(p, r);
                    bind_pairs(q, r);
                }
            }
            Phase::Dense {
                eq_plus_one_outer,
                pc,
            } => {
                bind_pairs(eq_plus_one_outer, r);
                bind_pairs(pc, r);
            }
        }
    }
}

impl<F: JoltField> ProveRounds<F> for ShiftKernel<F> {
    fn num_rounds(&self) -> usize {
        self.log_t
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }

        // Degree-2 member: evals at t = 0 and t = 2; s(1) from the hint.
        let evals: [F; 2] = match &self.phase {
            Phase::PrefixSuffix { pairs } => {
                let mut acc = [F::Accumulator::default(); 2];
                for (p, q) in pairs {
                    for y in 0..p.len() / 2 {
                        let (p_0, p_1) = (p[2 * y], p[2 * y + 1]);
                        let (q_0, q_1) = (q[2 * y], q[2 * y + 1]);
                        acc[0].fmadd(p_0, q_0);
                        acc[1].fmadd(p_1 + p_1 - p_0, q_1 + q_1 - q_0);
                    }
                }
                acc.map(F::Accumulator::reduce)
            }
            Phase::Dense {
                eq_plus_one_outer,
                pc,
            } => {
                let mut acc = [F::Accumulator::default(); 2];
                let pair = |table: &[F], y: usize| (table[2 * y], table[2 * y + 1]);
                let extend = |(lo, hi): (F, F)| hi + hi - lo;
                for y in 0..eq_plus_one_outer.len() / 2 {
                    let eq1o = pair(eq_plus_one_outer, y);
                    let pcs = pair(pc, y);
                    acc[0].fmadd(eq1o.0, pcs.0);
                    acc[1].fmadd(extend(eq1o), extend(pcs));
                }
                acc.map(F::Accumulator::reduce)
            }
        };

        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for ShiftKernel<F> {
    type Relation = SpartanShift<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SpartanShiftOutputClaims<F>, SumcheckKernelError<F>> {
        self.challenges.require_complete()?;
        let Phase::Dense { pc, .. } = &self.phase else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "Spartan shift must finish in the dense phase",
            });
        };
        Ok(SpartanShiftOutputClaims { pc: pc[0] })
    }

    /// Pin the regenerated `eq+1` table to the verifier's scalar path: its
    /// fully bound value must equal `derive_output_term` at the bound point.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.challenges.require_complete()?;
        let Phase::Dense {
            eq_plus_one_outer, ..
        } = &self.phase
        else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "Spartan shift must finish in the dense phase",
            });
        };
        let id = JoltDerivedId::from(SpartanShiftPublic::EqPlusOneOuter);
        pin_derived_term(
            relation,
            id,
            input_points,
            output_points,
            challenges,
            eq_plus_one_outer[0],
        )
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_field::{Fr, Ring};
    use jolt_poly::EqPlusOnePolynomial;
    use jolt_verifier::stages::stage3::spartan_shift::{
        SpartanShift, SpartanShiftChallenges, SpartanShiftInputClaims,
    };
    use jolt_witness::{JoltWitnessOracle, JoltWitnessPlane, TraceBackend};

    use super::super::registers_read_write::test_support::{
        assert_kernel_parity, assert_nontrivial, challenge_sequence,
    };
    use super::super::testing::{empty_io, with_trace_plane, TraceBuilder};
    use super::OptimizedSpartanShift;

    /// A pc-varied trace: `2^log_t − 1` real rows (`log_t = 1`: both cycles
    /// real — the summand weights only cycle 1, and a padding row there
    /// zeroes the input claim) ending on the jump back to the halt row.
    fn with_shift_plane<R>(log_t: usize, f: impl FnOnce(&TraceBackend) -> R) -> R {
        let real_rows = if log_t == 1 { 2 } else { (1 << log_t) - 1 };
        let mut builder = TraceBuilder::new();
        while builder.len() + 1 < real_rows {
            builder.nop();
        }
        builder.jump(0);
        let trace = builder.finish(1 << log_t);
        with_trace_plane(log_t, 64, 4, trace, empty_io(), f)
    }

    fn run_parity(log_t: usize, seed: u64) {
        with_shift_plane(log_t, |backend| {
            let r_outer = challenge_sequence(log_t, seed ^ 0x07E5);
            let r_product = challenge_sequence(log_t, seed ^ 0xFACE);
            let gamma = Fr::from_u64(0x5EED_0F0F_1234_5678);
            let relation = SpartanShift::<Fr>::new(
                TraceDimensions::new(log_t),
                r_outer.clone(),
                r_product.clone(),
            );

            let pc: Vec<Fr> = JoltWitnessOracle::<Fr>::oracle_table(
                backend,
                JoltPolynomialId::Virtual(JoltVirtualPolynomial::PC),
            )
            .unwrap();
            let eq1_outer = EqPlusOnePolynomial::evals(&r_outer, None).1;
            let input_claim: Fr = (0..1usize << log_t).map(|j| eq1_outer[j] * pc[j]).sum();
            assert_nontrivial(input_claim);

            let round_challenges = challenge_sequence(log_t, seed);
            assert_kernel_parity(
                &OptimizedSpartanShift,
                backend as &dyn JoltWitnessPlane<Fr>,
                &relation,
                &SpartanShiftInputClaims::default(),
                &SpartanShiftInputClaims::<Vec<Fr>>::default(),
                &SpartanShiftChallenges { gamma },
                input_claim,
                &round_challenges,
            );
        });
    }

    #[test]
    fn parity_even_log_t() {
        run_parity(4, 211);
    }

    #[test]
    fn parity_odd_log_t() {
        run_parity(3, 223);
    }

    #[test]
    fn parity_deep_phase2() {
        run_parity(5, 227);
    }

    #[test]
    fn parity_minimal_single_round() {
        // log_t = 1: the P·Q phase covers the single round and the dense
        // phase materializes inside `finish_rounds`.
        run_parity(1, 229);
    }
}

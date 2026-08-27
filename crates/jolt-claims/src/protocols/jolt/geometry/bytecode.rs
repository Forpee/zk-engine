use jolt_field::{JoltField, Ring};
use jolt_poly::{EqPolynomial, IdentityPolynomial, MultilinearEvaluation};
use jolt_wasm_ir::{BytecodeRow, RowFlag, REGISTER_NONE};
use jolt_wasm_tables::WasmTable;

use crate::{challenge, derived, opening};

use super::super::{
    BytecodeReadRafChallenge, BytecodeReadRafPublic, JoltCommittedPolynomial, JoltExpr,
    JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
};
use super::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
use super::dimensions::JoltFormulaPointError;
use super::error::require_len;
use super::instruction::{imm, instruction_raf_flag, lookup_table_flag, row_flag_input};
use super::registers::{
    rd_wa_read_write, rd_wa_val_evaluation, rs1_ra_read_write, rs2_ra_read_write,
};
use super::spartan::pc_shift;

/// Per-stage (1..=5) gamma-power vector lengths for the bytecode read-RAF stage
/// folds — the arities of the prover's `challenge_scalar_powers` draws. The
/// verifier stores each stage's single drawn scalar and expands it with
/// [`stage_gamma_powers`], so these lengths are single-sourced with the
/// fold-side `require_len` guards.
///
/// [`stage_gamma_powers`]: crate::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges::stage_gamma_powers
pub const BYTECODE_STAGE_GAMMA_COUNTS: [usize; 5] = [
    // Stage 1: Imm, then one per row flag (all Spartan outer).
    1 + RowFlag::COUNT,
    // Stage 2: the Branch product-virtualization flag.
    1,
    // Stage 3: Imm (instruction input) and the three operand-source flags.
    4,
    // Stage 4: the RdWa, Rs1Ra, Rs2Ra register read-write openings.
    3,
    // Stage 5: RdWa (registers val evaluation), InstructionRafFlag, then one
    // per catalog table flag.
    2 + WasmTable::COUNT,
];

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafDimensions {
    log_t: usize,
    log_k: usize,
    committed_ra_polys: usize,
}

impl BytecodeReadRafDimensions {
    pub const fn new(log_t: usize, log_k: usize, committed_ra_polys: usize) -> Self {
        Self {
            log_t,
            log_k,
            committed_ra_polys,
        }
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    pub const fn log_k(self) -> usize {
        self.log_k
    }

    pub const fn num_committed_ra_polys(self) -> usize {
        self.committed_ra_polys
    }

    pub const fn sumcheck_rounds(self) -> usize {
        self.log_t + self.log_k
    }
}

/// The staged input fold shared by the bytecode read-RAF monolith and its
/// address phase: the five base staged claims at `γ^0..4`, then one raw claim
/// per `extra_stage_claims` entry at the following powers (the lattice
/// fused-inc consumer stages), then the Spartan outer/shift PC openings and
/// the constant entry term at the next three powers.
pub(crate) fn read_raf_address_input_fold<F>(extra_stage_claims: Vec<JoltExpr<F>>) -> JoltExpr<F>
where
    F: Ring,
{
    let gamma = challenge(BytecodeReadRafChallenge::Gamma);
    let base_stages = BYTECODE_STAGE_GAMMA_COUNTS.len();
    let num_val_stages = base_stages + extra_stage_claims.len();

    let mut fold = gamma.clone().pow(num_val_stages + 2)
        + stage1_claim()
        + gamma.clone() * stage2_claim()
        + gamma.clone().pow(2) * stage3_claim()
        + gamma.clone().pow(3) * stage4_claim()
        + gamma.clone().pow(4) * stage5_claim::<F>();
    for (index, claim) in extra_stage_claims.into_iter().enumerate() {
        fold = fold + gamma.clone().pow(base_stages + index) * claim;
    }
    fold + gamma.clone().pow(num_val_stages) * opening(pc_spartan_outer())
        + gamma.pow(num_val_stages + 1) * opening(pc_shift())
}

pub(crate) fn read_raf_cycle_output<F>(
    dimensions: BytecodeReadRafDimensions,
    num_val_stages: usize,
) -> JoltExpr<F>
where
    F: Ring,
{
    let gamma = challenge(BytecodeReadRafChallenge::Gamma);
    let mut output_coeff = JoltExpr::zero();
    for stage in 0..num_val_stages {
        output_coeff = output_coeff
            + gamma.clone().pow(stage) * derived(BytecodeReadRafPublic::StageValue(stage));
    }
    output_coeff = output_coeff
        + gamma.clone().pow(num_val_stages) * derived(BytecodeReadRafPublic::SpartanOuterRaf)
        + gamma.clone().pow(num_val_stages + 1) * derived(BytecodeReadRafPublic::SpartanShiftRaf)
        + gamma.pow(num_val_stages + 2) * derived(BytecodeReadRafPublic::Entry);

    output_coeff * bytecode_ra_product(dimensions)
}

pub(crate) fn read_raf_cycle_output_committed<F>(
    dimensions: BytecodeReadRafDimensions,
    num_val_stages: usize,
) -> JoltExpr<F>
where
    F: Ring,
{
    let gamma = challenge(BytecodeReadRafChallenge::Gamma);
    // The staged Val factor multiplies after the RA product so the lowered
    // R1CS auxiliary chain matches core's `[ra..., val_stage]` factor order.
    let mut output = JoltExpr::zero();
    for stage in 0..num_val_stages {
        output = output
            + gamma.clone().pow(stage)
                * derived(BytecodeReadRafPublic::StageCycleEq(stage))
                * bytecode_ra_product(dimensions)
                * opening(super::claim_reductions::bytecode::bytecode_val_stage_opening(stage));
    }
    let raf_coeff = gamma.clone().pow(num_val_stages)
        * derived(BytecodeReadRafPublic::SpartanOuterRaf)
        + gamma.clone().pow(num_val_stages + 1) * derived(BytecodeReadRafPublic::SpartanShiftRaf)
        + gamma.pow(num_val_stages + 2) * derived(BytecodeReadRafPublic::Entry);

    output + raf_coeff * bytecode_ra_product(dimensions)
}

pub const READ_RAF_CYCLE_STAGES: usize = BYTECODE_STAGE_GAMMA_COUNTS.len();

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafOutputOpenings {
    pub bytecode_ra: Vec<JoltOpeningId>,
}

pub fn bytecode_read_raf_address_phase_opening() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::BytecodeReadRafAddrClaim,
        JoltRelationId::BytecodeReadRaf,
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafPublicValues<F: JoltField> {
    pub stage_values: [F; 5],
    pub spartan_outer_raf: F,
    pub spartan_shift_raf: F,
    pub entry: F,
}

impl<F: JoltField> BytecodeReadRafPublicValues<F> {
    /// Returns `None` for committed-mode publics (`StageCycleEq`) and
    /// out-of-range stage indices so a wrong-mode formula fails loudly at the
    /// source instead of evaluating with a silently zeroed term.
    pub fn value(&self, id: BytecodeReadRafPublic) -> Option<F> {
        match id {
            BytecodeReadRafPublic::StageValue(index) => self.stage_values.get(index).copied(),
            BytecodeReadRafPublic::StageCycleEq(_) => None,
            BytecodeReadRafPublic::SpartanOuterRaf => Some(self.spartan_outer_raf),
            BytecodeReadRafPublic::SpartanShiftRaf => Some(self.spartan_shift_raf),
            BytecodeReadRafPublic::Entry => Some(self.entry),
        }
    }
}

/// Committed-program read-RAF publics: the bytecode table is not available,
/// so only the table-independent factors are computed. The per-stage Val
/// factors are openings; their cycle-eq coefficients are public. One cycle eq
/// per relation stage — five in base mode, nine in lattice mode (the four
/// fused-inc consumer stages follow the base five).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafCommittedPublicValues<F: JoltField> {
    pub stage_cycle_eqs: [F; READ_RAF_CYCLE_STAGES],
    pub spartan_outer_raf: F,
    pub spartan_shift_raf: F,
    pub entry: F,
}

impl<F: JoltField> BytecodeReadRafCommittedPublicValues<F> {
    /// Returns `None` for full-mode publics (`StageValue`) and out-of-range
    /// stage indices so a wrong-mode formula fails loudly at the source
    /// instead of evaluating with a silently zeroed term.
    pub fn value(&self, id: BytecodeReadRafPublic) -> Option<F> {
        match id {
            BytecodeReadRafPublic::StageValue(_) => None,
            BytecodeReadRafPublic::StageCycleEq(index) => self.stage_cycle_eqs.get(index).copied(),
            BytecodeReadRafPublic::SpartanOuterRaf => Some(self.spartan_outer_raf),
            BytecodeReadRafPublic::SpartanShiftRaf => Some(self.spartan_shift_raf),
            BytecodeReadRafPublic::Entry => Some(self.entry),
        }
    }
}

pub struct BytecodeReadRafCommittedEvaluationInputs<'a, F> {
    pub r_address: &'a [F],
    pub r_cycle: &'a [F],
    /// One cycle point per relation stage (five base, nine lattice — the four
    /// fused-inc consumer points follow the base five).
    pub stage_cycle_points: [&'a [F]; READ_RAF_CYCLE_STAGES],
    pub entry_bytecode_index: usize,
}

pub fn read_raf_committed_public_values<F>(
    inputs: BytecodeReadRafCommittedEvaluationInputs<'_, F>,
) -> BytecodeReadRafCommittedPublicValues<F>
where
    F: JoltField,
{
    let stage_cycle_eqs = inputs
        .stage_cycle_points
        .map(|stage_cycle_point| EqPolynomial::<F>::mle(stage_cycle_point, inputs.r_cycle));
    let (spartan_outer_raf, spartan_shift_raf, entry) = read_raf_raf_entry_publics(
        inputs.r_address,
        inputs.r_cycle,
        stage_cycle_eqs[0],
        stage_cycle_eqs[2],
        inputs.entry_bytecode_index,
    );

    BytecodeReadRafCommittedPublicValues {
        stage_cycle_eqs,
        spartan_outer_raf,
        spartan_shift_raf,
        entry,
    }
}

/// Table-independent read-RAF publics shared by the full and committed
/// evaluation paths: `(SpartanOuterRaf, SpartanShiftRaf, Entry)`, where the
/// RAF terms scale `Int(r_address)` by the stage-1/stage-3 cycle-eq factors.
fn read_raf_raf_entry_publics<F>(
    r_address: &[F],
    r_cycle: &[F],
    outer_stage_cycle_eq: F,
    shift_stage_cycle_eq: F,
    entry_bytecode_index: usize,
) -> (F, F, F)
where
    F: JoltField,
{
    let identity = IdentityPolynomial::new(r_address.len()).evaluate(r_address);
    let spartan_outer_raf = identity * outer_stage_cycle_eq;
    let spartan_shift_raf = identity * shift_stage_cycle_eq;

    let entry_bits = (0..r_address.len())
        .map(|i| F::from_u64(((entry_bytecode_index >> (r_address.len() - 1 - i)) & 1) as u64))
        .collect::<Vec<_>>();
    let zero_cycle = vec![F::zero(); r_cycle.len()];
    let entry = EqPolynomial::<F>::mle(&entry_bits, r_address)
        * EqPolynomial::<F>::mle(&zero_cycle, r_cycle);

    (spartan_outer_raf, spartan_shift_raf, entry)
}

pub struct BytecodeReadRafEvaluationInputs<'a, F> {
    pub bytecode: &'a [BytecodeRow],
    pub r_address: &'a [F],
    pub r_cycle: &'a [F],
    pub stage_cycle_points: [&'a [F]; 5],
    pub register_read_write_point: &'a [F],
    pub register_val_evaluation_point: &'a [F],
    pub entry_bytecode_index: usize,
    pub stage1_gammas: &'a [F],
    pub stage2_gammas: &'a [F],
    pub stage3_gammas: &'a [F],
    pub stage4_gammas: &'a [F],
    pub stage5_gammas: &'a [F],
}

pub struct BytecodeReadRafStageValueInputs<'a, F> {
    pub bytecode: &'a [BytecodeRow],
    pub register_read_write_point: &'a [F],
    pub register_val_evaluation_point: &'a [F],
    pub stage1_gammas: &'a [F],
    pub stage2_gammas: &'a [F],
    pub stage3_gammas: &'a [F],
    pub stage4_gammas: &'a [F],
    pub stage5_gammas: &'a [F],
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BytecodeReadRafRegisterEqEvals<F> {
    pub read_write: Vec<F>,
    pub val_evaluation: Vec<F>,
}

fn read_raf_register_eq_evals<F>(
    register_read_write_point: &[F],
    register_val_evaluation_point: &[F],
) -> BytecodeReadRafRegisterEqEvals<F>
where
    F: JoltField,
{
    BytecodeReadRafRegisterEqEvals {
        read_write: EqPolynomial::<F>::evals(register_read_write_point, None),
        val_evaluation: EqPolynomial::<F>::evals(register_val_evaluation_point, None),
    }
}

/// Every bytecode row's staged values: the five gamma-folded stages, plus
/// (lattice) the store circuit flag as the sixth staged value, folded like
/// the others by the read-raf consumers.
pub fn read_raf_stage_values<F>(
    inputs: BytecodeReadRafStageValueInputs<'_, F>,
) -> Vec<[F; NUM_BYTECODE_VAL_STAGES]>
where
    F: JoltField,
{
    let register_eq = read_raf_register_eq_evals(
        inputs.register_read_write_point,
        inputs.register_val_evaluation_point,
    );
    inputs
        .bytecode
        .iter()
        .map(|instruction| {
            read_raf_row_values::<F>(
                instruction,
                &register_eq.read_write,
                &register_eq.val_evaluation,
                inputs.stage1_gammas,
                inputs.stage2_gammas,
                inputs.stage3_gammas,
                inputs.stage4_gammas,
                inputs.stage5_gammas,
            )
        })
        .collect()
}

pub fn read_raf_public_values<F>(
    inputs: BytecodeReadRafEvaluationInputs<'_, F>,
) -> Result<BytecodeReadRafPublicValues<F>, JoltFormulaPointError>
where
    F: JoltField,
{
    require_len(inputs.stage1_gammas, BYTECODE_STAGE_GAMMA_COUNTS[0])?;
    require_len(inputs.stage2_gammas, BYTECODE_STAGE_GAMMA_COUNTS[1])?;
    require_len(inputs.stage3_gammas, BYTECODE_STAGE_GAMMA_COUNTS[2])?;
    require_len(inputs.stage4_gammas, BYTECODE_STAGE_GAMMA_COUNTS[3])?;
    require_len(inputs.stage5_gammas, BYTECODE_STAGE_GAMMA_COUNTS[4])?;

    let expected_domain = 1usize << inputs.r_address.len();
    if inputs.bytecode.len() != expected_domain {
        return Err(JoltFormulaPointError::EvaluationDomainLengthMismatch {
            expected: expected_domain,
            got: inputs.bytecode.len(),
        });
    }

    let register_eq = read_raf_register_eq_evals(
        inputs.register_read_write_point,
        inputs.register_val_evaluation_point,
    );
    let address_eq_evals = EqPolynomial::<F>::evals(inputs.r_address, None);

    // The base monolith publics carry the five gamma'd stages only; the
    // lattice sixth (store) row value never flows through this path, so the
    // zip below is deliberately driven by the five-slot accumulator.
    let mut stage_values = [F::zero(); 5];
    for (instruction, eq_address) in inputs.bytecode.iter().zip(address_eq_evals) {
        let row_values = read_raf_row_values::<F>(
            instruction,
            &register_eq.read_write,
            &register_eq.val_evaluation,
            inputs.stage1_gammas,
            inputs.stage2_gammas,
            inputs.stage3_gammas,
            inputs.stage4_gammas,
            inputs.stage5_gammas,
        );
        for (stage_value, row_value) in stage_values.iter_mut().zip(row_values) {
            *stage_value += row_value * eq_address;
        }
    }

    let stage_cycle_eqs = inputs
        .stage_cycle_points
        .iter()
        .map(|stage_cycle_point| EqPolynomial::<F>::mle(stage_cycle_point, inputs.r_cycle))
        .collect::<Vec<_>>();
    for (stage_value, stage_cycle_eq) in stage_values.iter_mut().zip(&stage_cycle_eqs) {
        *stage_value *= *stage_cycle_eq;
    }

    let (spartan_outer_raf, spartan_shift_raf, entry) = read_raf_raf_entry_publics(
        inputs.r_address,
        inputs.r_cycle,
        stage_cycle_eqs[0],
        stage_cycle_eqs[2],
        inputs.entry_bytecode_index,
    );

    Ok(BytecodeReadRafPublicValues {
        stage_values,
        spartan_outer_raf,
        spartan_shift_raf,
        entry,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Each gamma slice corresponds to one protocol subexpression."
)]
fn read_raf_row_values<F>(
    row: &BytecodeRow,
    register_read_write_eq: &[F],
    register_val_evaluation_eq: &[F],
    stage1_gammas: &[F],
    stage2_gammas: &[F],
    stage3_gammas: &[F],
    stage4_gammas: &[F],
    stage5_gammas: &[F],
) -> [F; NUM_BYTECODE_VAL_STAGES]
where
    F: JoltField,
{
    let flags = row.flags;
    let imm = F::from_i128(row.imm_signed());
    let mut stage1 = imm;
    for (index, flag) in RowFlag::ALL.into_iter().enumerate() {
        if flags.has(flag) {
            stage1 += stage1_gammas[index + 1];
        }
    }
    let mut stage2 = F::zero();
    if flags.has(RowFlag::Branch) {
        stage2 += stage2_gammas[0];
    }
    let mut stage3 = imm;
    if flags.has(RowFlag::LeftIsRs1) {
        stage3 += stage3_gammas[1];
    }
    if flags.has(RowFlag::RightIsRs2) {
        stage3 += stage3_gammas[2];
    }
    if flags.has(RowFlag::RightIsImm) {
        stage3 += stage3_gammas[3];
    }
    let rd = register(row.rd);
    let stage4 = register_eq(rd, register_read_write_eq) * stage4_gammas[0]
        + register_eq(register(row.rs1), register_read_write_eq) * stage4_gammas[1]
        + register_eq(register(row.rs2), register_read_write_eq) * stage4_gammas[2];
    let mut stage5 = register_eq(rd, register_val_evaluation_eq);
    if row.raf_flag() {
        stage5 += stage5_gammas[1];
    }
    if let Some(op) = row.table_op() {
        stage5 += stage5_gammas[2 + WasmTable::of(op).index()];
    }
    [stage1, stage2, stage3, stage4, stage5]
}
fn register(id: u8) -> Option<u8> {
    (id != REGISTER_NONE).then_some(id)
}
fn register_eq<F: JoltField>(register: Option<u8>, eq: &[F]) -> F {
    register
        .and_then(|register| eq.get(register as usize))
        .copied()
        .unwrap_or_else(F::zero)
}

pub fn read_raf_output_openings(
    dimensions: BytecodeReadRafDimensions,
) -> BytecodeReadRafOutputOpenings {
    BytecodeReadRafOutputOpenings {
        bytecode_ra: (0..dimensions.num_committed_ra_polys())
            .map(bytecode_ra)
            .collect(),
    }
}

/// Openings that must agree across relations. WebAssembly rows have no
/// unexpanded-pc value column, so there are none.
pub fn read_raf_consistency_openings() -> [(JoltOpeningId, JoltOpeningId); 0] {
    []
}

pub(crate) fn stage1_claim<F>() -> JoltExpr<F>
where
    F: Ring,
{
    let beta = challenge(BytecodeReadRafChallenge::Stage1Gamma);
    let mut claim = opening(imm_spartan_outer());
    for (i, flag) in RowFlag::ALL.into_iter().enumerate() {
        claim = claim + beta.clone().pow(i + 1) * opening(row_flag_spartan_outer(flag));
    }
    claim
}

pub(crate) fn stage2_claim<F>() -> JoltExpr<F>
where
    F: Ring,
{
    opening(row_flag_product(RowFlag::Branch))
}

pub(crate) fn stage3_claim<F>() -> JoltExpr<F>
where
    F: Ring,
{
    let beta = challenge(BytecodeReadRafChallenge::Stage3Gamma);
    opening(imm())
        + beta.clone() * opening(row_flag_input(RowFlag::LeftIsRs1))
        + beta.clone().pow(2) * opening(row_flag_input(RowFlag::RightIsRs2))
        + beta.pow(3) * opening(row_flag_input(RowFlag::RightIsImm))
}

pub(crate) fn stage4_claim<F>() -> JoltExpr<F>
where
    F: Ring,
{
    let beta = challenge(BytecodeReadRafChallenge::Stage4Gamma);

    opening(rd_wa_read_write())
        + beta.clone() * opening(rs1_ra_read_write())
        + beta.pow(2) * opening(rs2_ra_read_write())
}

pub(crate) fn stage5_claim<F>() -> JoltExpr<F>
where
    F: Ring,
{
    let beta = challenge(BytecodeReadRafChallenge::Stage5Gamma);
    let mut claim =
        opening(rd_wa_val_evaluation()) + beta.clone() * opening(instruction_raf_flag());

    for (i, table) in WasmTable::iter().enumerate() {
        claim = claim + beta.clone().pow(i + 2) * opening(lookup_table_flag(table));
    }

    claim
}

fn bytecode_ra_product<F>(dimensions: BytecodeReadRafDimensions) -> JoltExpr<F>
where
    F: Ring,
{
    let mut product = JoltExpr::one();
    for i in 0..dimensions.num_committed_ra_polys() {
        product = product * opening(bytecode_ra(i));
    }
    product
}

pub(crate) fn imm_spartan_outer() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::Imm, JoltRelationId::SpartanOuter)
}

fn row_flag_spartan_outer(flag: RowFlag) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RowFlag(flag),
        JoltRelationId::SpartanOuter,
    )
}

pub(crate) fn row_flag_product(flag: RowFlag) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RowFlag(flag),
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub(crate) fn pc_spartan_outer() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::PC, JoltRelationId::SpartanOuter)
}

pub fn bytecode_ra(index: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::BytecodeRa(index),
        JoltRelationId::BytecodeReadRaf,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, Ring};
    use jolt_poly::EqPolynomial;
    use jolt_wasm_ir::{AluOp, Ir, Operand, Reg, Width};

    #[test]
    fn read_raf_register_eq_evals_builds_register_address_tables() {
        let read_write = vec![Fr::from_u64(2), Fr::from_u64(3)];
        let val_evaluation = vec![Fr::from_u64(5), Fr::from_u64(7)];
        let eq = read_raf_register_eq_evals(&read_write, &val_evaluation);

        assert_eq!(
            eq,
            BytecodeReadRafRegisterEqEvals {
                read_write: EqPolynomial::<Fr>::evals(&read_write, None),
                val_evaluation: EqPolynomial::<Fr>::evals(&val_evaluation, None),
            }
        );
    }

    fn gammas(base: u64, count: usize) -> Vec<Fr> {
        (0..count)
            .map(|value| Fr::from_u64(value as u64 + base))
            .collect()
    }

    #[test]
    fn read_raf_stage_values_match_row_formula() {
        let bytecode = vec![
            BytecodeRow::of(Ir::alu(
                AluOp::Add(Width::W64),
                Reg::T0,
                Reg::T1,
                Operand::Reg(Reg::T2),
            )),
            BytecodeRow::of(Ir::Halt),
        ];
        let register_read_write_point = vec![Fr::from_u64(2); 4];
        let register_val_evaluation_point = vec![Fr::from_u64(3); 4];
        let stage1_gammas = gammas(1, BYTECODE_STAGE_GAMMA_COUNTS[0]);
        let stage2_gammas = gammas(11, BYTECODE_STAGE_GAMMA_COUNTS[1]);
        let stage3_gammas = gammas(17, BYTECODE_STAGE_GAMMA_COUNTS[2]);
        let stage4_gammas = gammas(29, BYTECODE_STAGE_GAMMA_COUNTS[3]);
        let stage5_gammas = gammas(37, BYTECODE_STAGE_GAMMA_COUNTS[4]);
        let register_eq =
            read_raf_register_eq_evals(&register_read_write_point, &register_val_evaluation_point);

        let stage_values = read_raf_stage_values(BytecodeReadRafStageValueInputs {
            bytecode: &bytecode,
            register_read_write_point: &register_read_write_point,
            register_val_evaluation_point: &register_val_evaluation_point,
            stage1_gammas: &stage1_gammas,
            stage2_gammas: &stage2_gammas,
            stage3_gammas: &stage3_gammas,
            stage4_gammas: &stage4_gammas,
            stage5_gammas: &stage5_gammas,
        });
        let expected = bytecode
            .iter()
            .map(|row| {
                read_raf_row_values(
                    row,
                    &register_eq.read_write,
                    &register_eq.val_evaluation,
                    &stage1_gammas,
                    &stage2_gammas,
                    &stage3_gammas,
                    &stage4_gammas,
                    &stage5_gammas,
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(stage_values, expected);
    }
}

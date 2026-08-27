//! Stage 0: input validation, the Fiat-Shamir preamble, and witness
//! commitment.
//!
//! The transcript work is the verifier's own exported code
//! ([`validate_inputs_from_parts`], [`absorb_transcript_preamble`],
//! [`absorb_transcript_commitments`]) — the two sides share the absorb
//! sequence structurally, so stage-0 Fiat-Shamir drift is impossible by
//! construction. The commitment compute is delegated to the `jolt-kernels`
//! witness-commitment kernel; only the absorbs happen here.

use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_claims::protocols::jolt::JoltPolynomialId;
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_kernels::reference::bytecode_read_raf::BytecodeReadRafWitness;
use jolt_kernels::reference::instruction_read_raf::InstructionReadRafWitness;
use jolt_kernels::{CommitmentGrid, JoltBackend, ProofSession, WitnessCommitment};
use jolt_openings::CommitmentScheme;
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::proof::JoltCommitments;
use jolt_verifier::{
    absorb_committed_program_commitments, absorb_transcript_commitments,
    absorb_transcript_preamble, validate_inputs_from_parts, CheckedInputs, ProofTranscriptConfig,
};
use jolt_wasm_program::PublicIo;
use jolt_witness::{validate_servable, JoltWitnessOracle, RowSource, WitnessBundle};

use crate::{CommittedProgramCandidates, JoltProverPreprocessing, ProverConfig, ProverError};

/// The externally supplied trusted-advice commitment (produced at
/// preprocessing time, before any proving) and its opening hint. Mirrors
/// legacy's prover-constructor pair: the commitment is absorbed in stage 0 and
/// batched in stage 8, and the hint joins the stage-8 hint combination.
pub struct TrustedAdviceCommitment<PCS: CommitmentScheme> {
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
}

/// Stage 0's outputs: the validated inputs, the seeded transcript (positioned
/// exactly where the verifier's `verify_until_stage1` leaves its own), the
/// witness commitments in wire form, the untrusted-advice commitment (proved
/// at prove time, carried on the proof), and the per-polynomial opening hints
/// the stage-8 joint opening will consume (advice hints included).
pub struct Stage0Output<PCS, T>
where
    PCS: CommitmentScheme,
{
    pub checked: CheckedInputs,
    pub transcript: T,
    pub commitments: JoltCommitments<PCS::Output>,
    pub untrusted_advice_commitment: Option<PCS::Output>,
    pub hints: Vec<(JoltCommittedPolynomial, PCS::OpeningHint)>,
}

/// Validate inputs, seed the transcript, commit the witness (the untrusted
/// advice polynomial in its own balanced grid), and absorb the commitments
/// (main, untrusted advice, trusted advice, then the preprocessing-held
/// committed-program chunk/image commitments — the verifier's own absorb
/// order).
#[tracing::instrument(skip_all)]
pub fn prove_stage0<F, PCS, VC, T, W>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&TrustedAdviceCommitment<PCS>>,
    witness: &W,
    public_io: &PublicIo,
) -> Result<Stage0Output<PCS, T>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    PCS::Output: AppendToTranscript,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
    W: JoltWitnessOracle<F> + RowSource,
{
    // Committed-program mode needs the prover-retained full program + hints;
    // require presence to agree with the verifier preprocessing's mode.
    if preprocessing.verifier.program.committed().is_some()
        != preprocessing.committed_program.is_some()
    {
        return Err(ProverError::Unsupported {
            reason: "committed-program prover data presence disagrees with the preprocessing mode",
        });
    }
    // The chunk commitments bake their trace order in at preprocessing time;
    // a disagreeing proof config would transpose the rebuilt chunk tables
    // against the absorbed commitments and fail only at verification.
    if preprocessing
        .committed_program
        .as_ref()
        .is_some_and(|committed| committed.trace_order != config.trace_polynomial_order)
    {
        return Err(ProverError::Unsupported {
            reason: "committed-program preprocessing was built for a different trace layout",
        });
    }
    // The verifier's own input validation doubles as the prover's self-check
    // and produces the normalized `CheckedInputs` the preamble absorbs. The
    // zk axis is the compiled feature — the co-compiled verifier's
    // `SELECTED_ZK_CONFIG` flips with the same feature, so both sides always
    // agree.
    let checked = validate_inputs_from_parts(
        &preprocessing.verifier,
        public_io,
        config.trace_length,
        config.ram_K,
        config.trace_polynomial_order,
        config.one_hot_config,
        trusted_advice.is_some(),
        false,
        cfg!(feature = "zk"),
    )?;

    let mut transcript = T::new(b"Jolt");
    absorb_transcript_preamble(
        &checked,
        ProofTranscriptConfig {
            rw_config: config.rw_config,
            one_hot_config: config.one_hot_config,
            trace_polynomial_order: config.trace_polynomial_order,
        },
        &mut transcript,
    );

    let ids: Vec<JoltCommittedPolynomial> = witness
        .committed_order()?
        .into_iter()
        .filter(|id| {
            !matches!(
                id,
                JoltCommittedPolynomial::TrustedAdvice | JoltCommittedPolynomial::UntrustedAdvice
            )
        })
        .collect();
    // Stage-0 validation: every id the proof will request — the committed
    // set and each bundle's annotated set — must be servable by the backend
    // before witness generation starts.
    let requested = ids
        .iter()
        .map(|&id| JoltPolynomialId::Committed(id))
        .chain(InstructionReadRafWitness::annotated_ids())
        .chain(BytecodeReadRafWitness::annotated_ids());
    validate_servable(witness as &dyn JoltWitnessOracle<F>, requested)?;

    let grid = CommitmentGrid {
        total_vars: config.commitment_total_vars(CommittedProgramCandidates::from_schedule(
            &checked.precommitted,
        )),
        log_t: config.trace_length.ilog2() as usize,
        log_k_chunk: config.one_hot_config.committed_chunk_bits(),
        order: config.trace_polynomial_order,
    };
    // The `commit_witness` kernel-seam span sits at this call boundary, not
    // on any one backend impl, so every `CommitWitness` backend inherits it
    // (the taxonomy advertises it as backend-neutral).
    let committed = tracing::info_span!(
        "commit_witness",
        columns = ids.len(),
        total_vars = grid.total_vars
    )
    .in_scope(|| {
        backend.commit.commit_witness(
            session,
            witness as &dyn RowSource,
            &ids,
            grid,
            &preprocessing.pcs_setup,
        )
    })?;
    let (commitments, mut hints) = assemble_commitments::<PCS>(committed)?;

    // The WASM layout has no advice regions: `validate_inputs` rejects
    // advice commitments, so no untrusted advice polynomial is ever committed.
    let untrusted_advice_commitment = None;
    if let Some(trusted) = trusted_advice {
        hints.push((JoltCommittedPolynomial::TrustedAdvice, trusted.hint.clone()));
    }
    // The committed-program hints ride from preprocessing (the chunk/image
    // commitments were produced there, before any proving).
    if let Some(committed) = &preprocessing.committed_program {
        let expected_chunks = checked
            .precommitted
            .bytecode
            .as_ref()
            .map_or(0, |layout| layout.chunk_count());
        if committed.bytecode_chunk_hints.len() != expected_chunks {
            return Err(ProverError::Unsupported {
                reason: "committed-program chunk hint count disagrees with the bytecode schedule",
            });
        }
        for (index, hint) in committed.bytecode_chunk_hints.iter().enumerate() {
            hints.push((JoltCommittedPolynomial::BytecodeChunk(index), hint.clone()));
        }
        hints.push((
            JoltCommittedPolynomial::ProgramImageInit,
            committed.program_image_hint.clone(),
        ));
    }

    absorb_transcript_commitments(
        &commitments,
        untrusted_advice_commitment.as_ref(),
        trusted_advice.map(|trusted| &trusted.commitment),
        &mut transcript,
    );
    if let Some(committed) = preprocessing.verifier.program.committed() {
        absorb_committed_program_commitments(
            &committed.bytecode_chunk_commitments,
            &committed.program_image_commitment,
            &mut transcript,
        );
    }

    Ok(Stage0Output {
        checked,
        transcript,
        commitments,
        untrusted_advice_commitment,
        hints,
    })
}

/// Split the kernel's flat id-ordered output into the proof's wire shape.
#[expect(
    clippy::type_complexity,
    reason = "the wire aggregate paired with its opening hints"
)]
fn assemble_commitments<PCS: CommitmentScheme>(
    committed: Vec<WitnessCommitment<PCS>>,
) -> Result<
    (
        JoltCommitments<PCS::Output>,
        Vec<(JoltCommittedPolynomial, PCS::OpeningHint)>,
    ),
    ProverError<PCS::Field>,
> {
    let mut rd_inc = None;
    let mut ram_inc = None;
    let mut instruction = Vec::new();
    let mut ram = Vec::new();
    let mut bytecode = Vec::new();
    let mut hints = Vec::with_capacity(committed.len());

    for entry in committed {
        let WitnessCommitment {
            id,
            commitment,
            hint,
        } = entry;
        match id {
            JoltCommittedPolynomial::RdInc => rd_inc = Some(commitment),
            JoltCommittedPolynomial::RamInc => ram_inc = Some(commitment),
            JoltCommittedPolynomial::InstructionRa(_) => instruction.push(commitment),
            JoltCommittedPolynomial::RamRa(_) => ram.push(commitment),
            JoltCommittedPolynomial::BytecodeRa(_) => bytecode.push(commitment),
            other => {
                return Err(ProverError::InvariantViolation {
                    reason: match other {
                        JoltCommittedPolynomial::TrustedAdvice
                        | JoltCommittedPolynomial::UntrustedAdvice => {
                            "advice polynomials are absorbed separately, not as main commitments"
                        }
                        _ => "precommitted polynomials are not main witness commitments",
                    },
                });
            }
        }
        hints.push((id, hint));
    }

    let (Some(rd_inc), Some(ram_inc)) = (rd_inc, ram_inc) else {
        return Err(ProverError::InvariantViolation {
            reason: "witness did not produce the RdInc/RamInc commitments",
        });
    };
    Ok((
        JoltCommitments::new(rd_inc, ram_inc, instruction, ram, bytecode),
        hints,
    ))
}

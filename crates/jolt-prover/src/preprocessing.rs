//! Prover preprocessing: the shared program tables, their digest, and the
//! prover-retained committed-program data.

use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::JoltVerifierPreprocessing;
use jolt_wasm_ir::{IrProgram, PreprocessingError};
use jolt_wasm_program::WasmProgramPreprocessing;

/// Preprocess an IR program for proving: the bytecode table and program
/// memory both sides agree on, plus the digest the Fiat-Shamir preamble
/// binds them with.
pub fn preprocess_program(
    program: &IrProgram,
    max_trace_length: usize,
) -> Result<(WasmProgramPreprocessing, [u8; 32]), PreprocessingError> {
    let preprocessing = WasmProgramPreprocessing::new(program, max_trace_length)?;
    let digest = preprocessing_digest(&preprocessing);
    Ok((preprocessing, digest))
}

/// The preprocessing digest: a Blake2b transcript over the encoded bytecode
/// table, the program memory words, the memory limits, and the trace budget.
pub fn preprocessing_digest(preprocessing: &WasmProgramPreprocessing) -> [u8; 32] {
    let mut transcript = Blake2bTranscript::<jolt_field::Fr>::new(b"jolt-wasm-preprocessing");
    transcript.append_bytes(&preprocessing.bytecode.encode());
    for word in &preprocessing.program_memory {
        transcript.append_bytes(&word.address.to_le_bytes());
        transcript.append_bytes(&word.value.to_le_bytes());
    }
    transcript.append_bytes(&preprocessing.memory.initial_pages.to_le_bytes());
    transcript.append_bytes(&preprocessing.memory.max_pages.to_le_bytes());
    transcript.append_bytes(&(preprocessing.max_trace_length as u64).to_le_bytes());
    transcript.state()
}

/// The prover-retained committed-program data: the verifier's preprocessing
/// carries only the program COMMITMENTS in committed mode, but the prover
/// still needs the full program (witness generation, the bytecode stage-value
/// folds, the reduction chunk grids, the stage-8 materialization) and the
/// commitments' opening material (the stage-8 openings).
#[derive(Clone)]
pub struct CommittedProgramProverData<PCS: CommitmentScheme> {
    pub full: WasmProgramPreprocessing,
    /// One opening hint per committed bytecode chunk, in chunk order.
    pub bytecode_chunk_hints: Vec<PCS::OpeningHint>,
    pub program_image_hint: PCS::OpeningHint,
    /// The trace order the chunk commitments' coefficient grids were built
    /// under at preprocessing time. Stage 0 rejects a proof config whose
    /// order disagrees — the chunk tables stages 6b/8 rebuild would transpose
    /// against the absorbed commitments and fail only at verification.
    pub trace_order: TracePolynomialOrder,
}

/// The prover's preprocessing is a strict superset of the verifier's: the
/// embedded [`JoltVerifierPreprocessing`] carries the program view, the
/// preprocessing digest, the PCS verifier setup, and the ZK
/// vector-commitment setup; the prover adds its PCS prover setup and, in
/// committed-program mode, the retained full program and opening hints.
/// Witness generation reads the full program through
/// [`program`](Self::program).
#[derive(Clone)]
pub struct JoltProverPreprocessing<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub verifier: JoltVerifierPreprocessing<PCS, VC>,
    pub pcs_setup: PCS::ProverSetup,
    /// Present exactly when the verifier preprocessing is committed-program.
    pub committed_program: Option<CommittedProgramProverData<PCS>>,
}

impl<PCS, VC> JoltProverPreprocessing<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    /// The full program preprocessing witness generation and the bytecode
    /// folds consume: the verifier's own full view, or the prover-retained
    /// copy in committed-program mode.
    pub fn program(&self) -> Option<&WasmProgramPreprocessing> {
        self.verifier
            .program
            .as_full()
            .or_else(|| self.committed_program.as_ref().map(|data| &data.full))
    }
}

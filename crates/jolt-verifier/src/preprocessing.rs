//! Verifier preprocessing inputs.

use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_wasm_ir::{MemoryLimits, Pc};
use jolt_wasm_program::{WasmProgramPreprocessing, PROGRAM_IMAGE_START_INDEX};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::VerifierError;

/// The program facts a committed-program verifier binds without holding the
/// bytecode table or program image.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramMetadata {
    /// Padded bytecode table length (the bytecode address domain).
    pub bytecode_len: usize,
    /// Exported function name → entry-stub pc.
    pub entries: BTreeMap<String, Pc>,
    /// Dense program-image length in words from [`PROGRAM_IMAGE_START_INDEX`].
    pub program_image_len_words: usize,
    pub memory: MemoryLimits,
}

impl ProgramMetadata {
    pub fn of(preprocessing: &WasmProgramPreprocessing) -> Self {
        Self {
            bytecode_len: preprocessing.bytecode.rows().len(),
            entries: preprocessing.bytecode.entries().clone(),
            program_image_len_words: preprocessing.program_image().words.len(),
            memory: preprocessing.memory,
        }
    }
}

/// Committed-program verifier inputs: trusted bytecode-chunk and program-image
/// commitments plus the program metadata they bind to. The chunk count is
/// implied by `bytecode_chunk_commitments.len()`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "PCS::Output: Serialize",
    deserialize = "PCS::Output: serde::de::DeserializeOwned"
))]
pub struct CommittedProgramPreprocessing<PCS: CommitmentScheme> {
    pub meta: ProgramMetadata,
    pub max_trace_length: usize,
    pub bytecode_chunk_commitments: Vec<PCS::Output>,
    pub program_image_commitment: PCS::Output,
}

impl<PCS: CommitmentScheme> CommittedProgramPreprocessing<PCS> {
    pub fn bytecode_chunk_count(&self) -> usize {
        self.bytecode_chunk_commitments.len()
    }
}

/// Program preprocessing in one of two modes, detected at runtime from the
/// deserialized preprocessing: `Full` carries the bytecode table and program
/// memory, `Committed` replaces them with trusted commitments plus metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "PCS::Output: Serialize",
    deserialize = "PCS::Output: serde::de::DeserializeOwned"
))]
pub enum ProgramPreprocessing<PCS: CommitmentScheme> {
    /// `Arc` so witness backends take an owning handle without deep-cloning
    /// the program-sized tables (serde `rc`: serializes as the contents).
    Full(Arc<WasmProgramPreprocessing>),
    Committed(CommittedProgramPreprocessing<PCS>),
}

impl<PCS: CommitmentScheme> ProgramPreprocessing<PCS> {
    pub fn as_full(&self) -> Option<&WasmProgramPreprocessing> {
        match self {
            Self::Full(full) => Some(full),
            Self::Committed(_) => None,
        }
    }

    /// The owning counterpart of [`as_full`](Self::as_full) — a refcount
    /// bump, never a copy.
    pub fn as_full_arc(&self) -> Option<Arc<WasmProgramPreprocessing>> {
        match self {
            Self::Full(full) => Some(Arc::clone(full)),
            Self::Committed(_) => None,
        }
    }

    pub fn committed(&self) -> Option<&CommittedProgramPreprocessing<PCS>> {
        match self {
            Self::Full(_) => None,
            Self::Committed(committed) => Some(committed),
        }
    }

    pub fn memory(&self) -> MemoryLimits {
        match self {
            Self::Full(full) => full.memory,
            Self::Committed(committed) => committed.meta.memory,
        }
    }

    pub fn max_trace_length(&self) -> usize {
        match self {
            Self::Full(full) => full.max_trace_length,
            Self::Committed(committed) => committed.max_trace_length,
        }
    }

    /// The entry-stub pc of an exported function.
    pub fn entry_pc(&self, export: &str) -> Option<Pc> {
        match self {
            Self::Full(full) => full.bytecode.entry(export),
            Self::Committed(committed) => committed.meta.entries.get(export).copied(),
        }
    }

    /// [`entry_pc`](Self::entry_pc), attributing an unknown export to the
    /// consuming `stage`.
    pub fn entry_pc_checked(
        &self,
        export: &str,
        stage: JoltRelationId,
    ) -> Result<Pc, VerifierError> {
        self.entry_pc(export)
            .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                stage,
                reason: format!("export {export:?} was not found in bytecode preprocessing"),
            })
    }

    /// Padded bytecode table length (the bytecode address domain).
    pub fn bytecode_len(&self) -> usize {
        match self {
            Self::Full(full) => full.bytecode.rows().len(),
            Self::Committed(committed) => committed.meta.bytecode_len,
        }
    }

    pub fn program_image_len_words(&self) -> usize {
        match self {
            Self::Full(full) => full.program_image().words.len(),
            Self::Committed(committed) => committed.meta.program_image_len_words,
        }
    }

    /// One past the last program-image word's RAM index.
    pub fn program_image_end_index(&self) -> u64 {
        PROGRAM_IMAGE_START_INDEX
            .saturating_add(crate::num::u64_from_usize(self.program_image_len_words()))
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "ProgramPreprocessing<PCS>: Serialize, PCS::VerifierSetup: Serialize, VC::Setup: Serialize",
    deserialize = "ProgramPreprocessing<PCS>: serde::de::DeserializeOwned, PCS::VerifierSetup: serde::de::DeserializeOwned, VC::Setup: serde::de::DeserializeOwned"
))]
pub struct JoltVerifierPreprocessing<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub program: ProgramPreprocessing<PCS>,
    pub preprocessing_digest: [u8; 32],
    pub pcs_setup: PCS::VerifierSetup,
    pub vc_setup: Option<VC::Setup>,
}

impl<PCS, VC> JoltVerifierPreprocessing<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub fn new(
        program: ProgramPreprocessing<PCS>,
        preprocessing_digest: [u8; 32],
        pcs_setup: PCS::VerifierSetup,
        vc_setup: Option<VC::Setup>,
    ) -> Self {
        Self {
            program,
            preprocessing_digest,
            pcs_setup,
            vc_setup,
        }
    }
}

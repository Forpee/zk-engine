//! The trace-backed witness backend: derives every served oracle from an
//! execution trace via the atomic extractors in [`crate::witnesses`].

use std::sync::Arc;

use jolt_claims::protocols::jolt::{
    geometry::{committed_openings, dimensions::REGISTER_ADDRESS_BITS, ra::JoltRaPolynomialLayout},
    JoltCommittedPolynomial, JoltFormulaDimensions, JoltOneHotConfig, JoltVirtualPolynomial,
};
use jolt_field::JoltField;
use jolt_wasm_program::{PublicIo, WasmProgramPreprocessing};

use crate::backend::ProgramSource;
use crate::witnesses::ram_access_address;
use crate::{TraceRow, WitnessError, JOLT_VM_LABEL, LOOKUP_ADDRESS_BITS};

mod cycle;
mod oracle;
mod ram;
mod registers;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JoltVmWitnessConfig {
    pub log_t: usize,
    pub ram_k: usize,
    pub one_hot: JoltOneHotConfig,
}

impl Default for JoltVmWitnessConfig {
    fn default() -> Self {
        Self::new(
            0,
            1,
            JoltOneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
        )
    }
}

impl JoltVmWitnessConfig {
    pub fn new(log_t: usize, ram_k: usize, one_hot: JoltOneHotConfig) -> Self {
        Self {
            log_t,
            ram_k,
            one_hot,
        }
    }

    pub const fn with_log_t(mut self, log_t: usize) -> Self {
        self.log_t = log_t;
        self
    }
}

/// The trace side of a witness: the program view, the compact proof rows,
/// and the run's public I/O (whose inputs seed the initial memory).
pub struct JoltVmWitnessInputs {
    pub preprocessing: Arc<WasmProgramPreprocessing>,
    pub rows: Arc<Vec<TraceRow>>,
    pub io: PublicIo,
}

impl JoltVmWitnessInputs {
    pub fn new(
        preprocessing: &Arc<WasmProgramPreprocessing>,
        rows: Arc<Vec<TraceRow>>,
        io: PublicIo,
    ) -> Self {
        Self {
            preprocessing: Arc::clone(preprocessing),
            rows,
            io,
        }
    }
}

/// Proof witness backed by shared compact rows.
pub struct TraceBackend {
    pub config: JoltVmWitnessConfig,
    pub preprocessing: Arc<WasmProgramPreprocessing>,
    pub rows: Arc<Vec<TraceRow>>,
    pub io: PublicIo,
}

impl ProgramSource for TraceBackend {
    fn program_preprocessing(&self) -> &WasmProgramPreprocessing {
        &self.preprocessing
    }
}

impl TraceBackend {
    /// Constructs a backend from proof rows, retaining their allocation.
    /// Trailing padding rows are trimmed (the walk re-pads).
    ///
    /// Panics when the rows exceed the cycle domain; use [`Self::try_new`]
    /// when the trace is not trusted.
    #[expect(
        clippy::panic,
        reason = "compatibility constructor for trusted prover-generated traces"
    )]
    pub fn new(config: JoltVmWitnessConfig, inputs: JoltVmWitnessInputs) -> Self {
        match Self::try_new(config, inputs) {
            Ok(backend) => backend,
            Err(error) => panic!("invalid proof-facing trace: {error}"),
        }
    }

    pub fn try_new(
        config: JoltVmWitnessConfig,
        inputs: JoltVmWitnessInputs,
    ) -> Result<Self, WitnessError> {
        let cycles = checked_pow2(config.log_t)?;
        let mut rows = inputs.rows;
        let trailing = rows
            .iter()
            .rev()
            .take_while(|row| **row == TraceRow::default())
            .count();
        if trailing > 0 {
            let kept = rows.len() - trailing;
            rows = Arc::new(rows[..kept].to_vec());
        }
        if rows.len() > cycles {
            return Err(WitnessError::InvalidWitnessData {
                label: JOLT_VM_LABEL,
                reason: format!(
                    "physical trace has {} rows but the cycle domain has {cycles}",
                    rows.len()
                ),
            });
        }
        Ok(Self {
            config,
            preprocessing: inputs.preprocessing,
            rows,
            io: inputs.io,
        })
    }

    pub fn committed_polynomial_order(&self) -> Result<Vec<JoltCommittedPolynomial>, WitnessError> {
        Ok(committed_openings::proof_commitment_order(
            self.ra_layout()?,
        ))
    }

    fn ra_layout(&self) -> Result<JoltRaPolynomialLayout, WitnessError> {
        self.formula_dimensions()
            .map(|dimensions| dimensions.ra_layout)
    }

    fn formula_dimensions(&self) -> Result<JoltFormulaDimensions, WitnessError> {
        let dimensions = self.config.one_hot.dimensions(
            self.config.log_t,
            LOOKUP_ADDRESS_BITS,
            self.preprocessing.bytecode.code_size(),
            self.config.ram_k,
        );
        JoltFormulaDimensions::try_from(dimensions).map_err(|error| {
            WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: error.to_string(),
            }
        })
    }

    fn trace_log_rows(&self) -> usize {
        self.config.log_t
    }

    fn ram_log_k(&self) -> Result<usize, WitnessError> {
        if self.config.ram_k == 0 || !self.config.ram_k.is_power_of_two() {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: format!(
                    "ram_k must be a nonzero power of two, got {}",
                    self.config.ram_k
                ),
            });
        }
        Ok(self.config.ram_k.ilog2() as usize)
    }

    fn ram_read_write_log_rows(&self) -> Result<usize, WitnessError> {
        self.config
            .log_t
            .checked_add(self.ram_log_k()?)
            .ok_or_else(|| WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: "RAM read-write rows overflow".to_owned(),
            })
    }

    fn register_read_write_log_rows(&self) -> Result<usize, WitnessError> {
        self.config
            .log_t
            .checked_add(REGISTER_ADDRESS_BITS)
            .ok_or_else(|| WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: "register read-write rows overflow".to_owned(),
            })
    }

    fn one_hot_log_rows(&self) -> Result<usize, WitnessError> {
        self.config
            .log_t
            .checked_add(self.config.one_hot.committed_chunk_bits())
            .ok_or_else(|| WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: "one-hot committed rows overflow".to_owned(),
            })
    }

    fn instruction_virtual_ra_log_rows(&self) -> Result<usize, WitnessError> {
        self.config
            .log_t
            .checked_add(self.config.one_hot.lookup_virtual_chunk_bits())
            .ok_or_else(|| WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: "instruction virtual RA rows overflow".to_owned(),
            })
    }

    fn instruction_virtual_ra_count(&self) -> Result<usize, WitnessError> {
        let chunk_bits = self.config.one_hot.lookup_virtual_chunk_bits();
        if chunk_bits == 0 || !LOOKUP_ADDRESS_BITS.is_multiple_of(chunk_bits) {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: format!(
                    "lookup virtual chunk bits {chunk_bits} must evenly divide {LOOKUP_ADDRESS_BITS}"
                ),
            });
        }
        Ok(LOOKUP_ADDRESS_BITS / chunk_bits)
    }
}

pub(crate) fn checked_pow2(log: usize) -> Result<usize, WitnessError> {
    1usize
        .checked_shl(log as u32)
        .filter(|_| log < usize::BITS as usize)
        .ok_or_else(|| WitnessError::InvalidDimensions {
            label: JOLT_VM_LABEL,
            reason: format!("2^{log} overflows usize"),
        })
}

pub(crate) fn require_index(index: usize, count: usize) -> Result<(), WitnessError> {
    if index < count {
        Ok(())
    } else {
        Err(WitnessError::UnknownOracle {
            label: JOLT_VM_LABEL,
        })
    }
}

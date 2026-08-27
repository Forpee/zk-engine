//! Per-proof configuration, derived from the execution trace.
//!
//! These five values are exactly the proof's wire config block
//! (`JoltProof::{trace_length, ram_K, rw_config, one_hot_config,
//! trace_polynomial_order}`) plus the Fiat-Shamir preamble inputs.

use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig, TracePolynomialOrder};
use jolt_field::JoltField;
use jolt_wasm_ir::layout::remap_word_address;
use jolt_wasm_ir::{MemoryLimits, REGISTER_COUNT};
use jolt_wasm_program::{max_ram_k, min_ram_k, WasmTraceRow};
use jolt_wasm_tables::XLEN;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::ProverError;

/// The full instruction lookup key width: two `XLEN`-bit operands.
const LOOKUP_ADDRESS_BITS: usize = 2 * XLEN;
#[cfg(feature = "parallel")]
const PARALLEL_DERIVE_MIN_ROWS: usize = 1 << 16;

/// The minimum padded trace length — Dory needs `T >= K^(1/D)` (256).
const MIN_PADDED_TRACE_LENGTH: usize = 256;

/// Trace length (log2) at which the one-hot chunking switches to the wide
/// policy.
const ONEHOT_CHUNK_THRESHOLD_LOG_T: usize = 25;

/// The proof-shape configuration for one proving run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[expect(non_snake_case)]
pub struct ProverConfig {
    /// Padded trace length (a power of two, at least 256).
    pub trace_length: usize,
    /// RAM address-space size (a power of two).
    pub ram_K: usize,
    pub rw_config: JoltReadWriteConfig,
    pub one_hot_config: JoltOneHotConfig,
    /// Coefficient placement of the trace polynomials in the commitment
    /// matrix. [`ProverConfig::derive`] always picks cycle-major;
    /// address-major is chosen by overwriting this field after derivation.
    /// Committed-program preprocessing bakes this order into its chunk
    /// commitments — it must be chosen before preprocessing and match here
    /// (stage 0 checks).
    pub trace_polynomial_order: TracePolynomialOrder,
}

impl ProverConfig {
    /// Derive the proof shape from an unpadded trace: pad the length (minimum
    /// 256 so `T >= K^(1/D)`, else next power of two past the trace plus its
    /// final no-op), size RAM to the highest touched word, the program image
    /// extent, or the public I/O window (whichever is largest), and pick the
    /// chunking policies from `log_T`.
    #[tracing::instrument(skip_all, name = "ProverConfig::derive", fields(rows = rows.len()))]
    #[expect(non_snake_case)]
    pub fn derive<F: JoltField>(
        rows: &[WasmTraceRow],
        program_image_end_index: u64,
        memory: MemoryLimits,
        max_trace_length: usize,
    ) -> Result<Self, ProverError<F>> {
        let trace_length = if rows.len() < MIN_PADDED_TRACE_LENGTH {
            MIN_PADDED_TRACE_LENGTH
        } else {
            (rows.len() + 1).next_power_of_two()
        };
        if trace_length > max_trace_length {
            return Err(ProverError::Unsupported {
                reason: "trace exceeds the preprocessing's maximum padded trace length",
            });
        }

        let touched_word = |row: &WasmTraceRow| {
            (row.is_load() || row.is_store())
                .then(|| remap_word_address(row.ram_address()))
                .flatten()
        };
        #[cfg(feature = "parallel")]
        let touched = if rows.len() >= PARALLEL_DERIVE_MIN_ROWS {
            rows.par_iter().filter_map(touched_word).max()
        } else {
            rows.iter().filter_map(touched_word).max()
        };
        #[cfg(not(feature = "parallel"))]
        let touched = rows.iter().filter_map(touched_word).max();
        let floor = min_ram_k(program_image_end_index).map_err(|_| ProverError::Unsupported {
            reason: "the program image does not fit a RAM domain",
        })?;
        let ram_K = touched
            .map_or(0, |word| word as usize + 1)
            .next_power_of_two()
            .max(floor);
        let ceiling = max_ram_k(memory).map_err(|_| ProverError::Unsupported {
            reason: "the linear memory limit does not fit a RAM domain",
        })?;
        if ram_K > ceiling {
            return Err(ProverError::Unsupported {
                reason: "the trace touches RAM beyond the program's memory limit",
            });
        }

        let log_T = trace_length.ilog2() as usize;
        Ok(Self {
            trace_length,
            ram_K,
            rw_config: read_write_config(log_T, ram_K.ilog2() as usize),
            one_hot_config: one_hot_config(log_T),
            trace_polynomial_order: TracePolynomialOrder::CycleMajor,
        })
    }

    /// The shared commitment-embedding variable count: the one-hot main matrix
    /// (`log_k_chunk + log_T`) maxed with the committed-program candidates
    /// when present.
    pub fn commitment_total_vars(
        &self,
        committed_program: Option<CommittedProgramCandidates>,
    ) -> usize {
        let mut total_vars =
            self.one_hot_config.committed_chunk_bits() + self.trace_length.ilog2() as usize;
        if let Some(committed) = committed_program {
            total_vars = total_vars
                .max(committed.bytecode_chunk_vars)
                .max(committed.program_image_vars);
        }
        total_vars
    }
}

/// Read-write checking phase splits: cycle variables in phase 1, address
/// variables in phase 2 (registers have a fixed 2^7 address space).
#[expect(non_snake_case)]
fn read_write_config(log_T: usize, ram_log_K: usize) -> JoltReadWriteConfig {
    JoltReadWriteConfig {
        ram_rw_phase1_num_rounds: log_T as u8,
        ram_rw_phase2_num_rounds: ram_log_K as u8,
        registers_rw_phase1_num_rounds: log_T as u8,
        registers_rw_phase2_num_rounds: REGISTER_COUNT.ilog2() as u8,
    }
}

/// One-hot chunking policy: below the trace-length threshold (`log_T < 25`),
/// 4-bit committed chunks and `LOG_K/8 = 16`-bit virtual-RA chunks; at or
/// above it, 8-bit committed chunks and `LOG_K/4 = 32`-bit virtual-RA chunks.
#[expect(non_snake_case)]
fn one_hot_config(log_T: usize) -> JoltOneHotConfig {
    if log_T < ONEHOT_CHUNK_THRESHOLD_LOG_T {
        JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: (LOOKUP_ADDRESS_BITS / 8) as u8,
        }
    } else {
        JoltOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: (LOOKUP_ADDRESS_BITS / 4) as u8,
        }
    }
}

/// The committed-program precommitted candidates' variable counts, folded
/// into the shared commitment grid.
#[derive(Clone, Copy, Debug)]
pub struct CommittedProgramCandidates {
    pub bytecode_chunk_vars: usize,
    pub program_image_vars: usize,
}

impl CommittedProgramCandidates {
    /// Read the candidates off the validated precommitted schedule: present
    /// exactly when committed-program layouts are.
    pub fn from_schedule(schedule: &jolt_verifier::stages::PrecommittedSchedule) -> Option<Self> {
        match (&schedule.bytecode, &schedule.program_image) {
            (Some(bytecode), Some(image)) => Some(Self {
                bytecode_chunk_vars: bytecode.chunk_shape().total_vars(),
                program_image_vars: image.image_shape().total_vars(),
            }),
            _ => None,
        }
    }
}

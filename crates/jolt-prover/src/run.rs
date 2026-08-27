//! One WebAssembly execution prepared for proving: the program executed on
//! the [`Machine`], its records materialized as proof rows, the derived
//! proof shape, and the run's public I/O — everything `prove` consumes
//! besides the preprocessing and the PCS setup.

use std::sync::Arc;

use jolt_field::JoltField;
use jolt_wasm_backend::Machine;
use jolt_wasm_ir::IrProgram;
use jolt_wasm_program::{build_trace_rows, PublicIo, WasmProgramPreprocessing, WasmTraceRow};
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

use crate::{ProverConfig, ProverError};

/// An executed run, ready to prove.
pub struct PreparedRun {
    pub rows: Arc<Vec<WasmTraceRow>>,
    pub io: PublicIo,
    pub config: ProverConfig,
}

impl PreparedRun {
    /// Execute `entry(args)` and derive the proof shape.
    pub fn execute<F: JoltField>(
        program: &IrProgram,
        preprocessing: &WasmProgramPreprocessing,
        entry: &str,
        args: &[u64],
    ) -> Result<Self, ProverError<F>> {
        let execution = Machine::new(program)
            .and_then(|machine| machine.invoke(entry, args))
            .map_err(|error| ProverError::Execution {
                reason: error.to_string(),
            })?;
        let rows =
            build_trace_rows(&execution.records).map_err(|error| ProverError::Execution {
                reason: error.to_string(),
            })?;
        let config = ProverConfig::derive::<F>(
            &rows,
            preprocessing.program_image().end_index(),
            preprocessing.memory,
            preprocessing.max_trace_length,
        )?;
        Ok(Self {
            rows: Arc::new(rows),
            io: PublicIo {
                entry: entry.to_owned(),
                inputs: args.to_vec(),
                outputs: execution.results,
            },
            config,
        })
    }

    /// The trace-backed witness plane over this run.
    pub fn witness(&self, preprocessing: &Arc<WasmProgramPreprocessing>) -> TraceBackend {
        TraceBackend::new(
            JoltVmWitnessConfig::new(
                self.config.trace_length.ilog2() as usize,
                self.config.ram_K,
                self.config.one_hot_config,
            ),
            JoltVmWitnessInputs::new(preprocessing, Arc::clone(&self.rows), self.io.clone()),
        )
    }
}

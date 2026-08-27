//! Sample-trace fixtures for the derive-generated bundle consistency tests.

use std::collections::BTreeMap;
use std::sync::Arc;

use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltPolynomialId};
use jolt_field::Fr;
use jolt_wasm_backend::Machine;
use jolt_wasm_ir::layout::LINEAR_MEMORY_BASE;
use jolt_wasm_ir::{AluOp, Ir, IrFunction, IrProgram, MemoryLimits, Operand, Reg, Width};
use jolt_wasm_program::{build_trace_rows, PublicIo, WasmProgramPreprocessing};

use crate::backend::trace::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
use crate::{BundleSource, JoltWitnessOracle, WitnessBundle};

/// The padded cycle domain of the sample trace.
pub const SAMPLE_CYCLES: usize = 8;

/// A hand-built program: `T0 = 5`, `T1 = T0 + 3`, `mem[LINEAR + 8] = T1`,
/// jump to the halt trampoline. No entry stub, no arguments.
#[expect(clippy::unwrap_used, reason = "test fixture construction")]
pub fn sample_program() -> IrProgram {
    let code = vec![
        Ir::Halt,
        Ir::const_(Reg::T0, 5),
        Ir::alu(AluOp::Add(Width::W64), Reg::T1, Reg::T0, Operand::Imm(3)),
        Ir::Store {
            base: Reg::ZERO,
            value: Reg::T1,
            offset: (LINEAR_MEMORY_BASE + 8) as i64,
        },
        Ir::Jump { target: 0 },
    ];
    let mut exports = BTreeMap::new();
    let _ = exports.insert("main".to_owned(), 0);
    let mut entries = BTreeMap::new();
    let _ = entries.insert("main".to_owned(), 1);
    IrProgram {
        code,
        functions: vec![IrFunction {
            entry: 1,
            params: 0,
            results: 0,
            frame_slots: 0,
        }],
        exports,
        entries,
        memory: MemoryLimits {
            initial_pages: 1,
            max_pages: 1,
        },
        globals: Vec::new(),
        data: Vec::new(),
        table: Vec::new(),
    }
    .validate()
    .unwrap()
}

trait Validate: Sized {
    fn validate(self) -> Result<Self, jolt_wasm_ir::PreprocessingError>;
}

impl Validate for IrProgram {
    fn validate(self) -> Result<Self, jolt_wasm_ir::PreprocessingError> {
        let _ = jolt_wasm_ir::WasmBytecode::preprocess(&self)?;
        Ok(self)
    }
}

/// Runs `f` against a small canned backend: the sample program's four rows,
/// padded to [`SAMPLE_CYCLES`].
#[expect(clippy::unwrap_used, reason = "test fixture construction")]
pub fn with_sample_backend<R>(f: impl FnOnce(&TraceBackend) -> R) -> R {
    let program = sample_program();
    let preprocessing = Arc::new(WasmProgramPreprocessing::new(&program, SAMPLE_CYCLES).unwrap());
    let execution = Machine::new(&program).unwrap().invoke("main", &[]).unwrap();
    let rows = Arc::new(build_trace_rows(&execution.records).unwrap());
    let io = PublicIo {
        entry: "main".to_owned(),
        inputs: Vec::new(),
        outputs: Vec::new(),
    };
    let config = JoltVmWitnessConfig::new(
        SAMPLE_CYCLES.ilog2() as usize,
        1 << 20,
        JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        },
    );
    let backend = TraceBackend::new(config, JoltVmWitnessInputs::new(&preprocessing, rows, io));
    f(&backend)
}

/// Asserts that one annotated bundle field's column (extracted by `value`)
/// equals the backend's `oracle_table` for `id` — the typed path and the id
/// path meeting at the `Extract` impls. Driven by the derive-generated
/// per-field consistency tests.
#[expect(clippy::unwrap_used, reason = "test assertion helper")]
pub fn assert_bundle_column_matches<B>(id: JoltPolynomialId, value: impl Fn(&B) -> Fr)
where
    B: WitnessBundle + Clone + Send + Sync,
{
    with_sample_backend(|backend| {
        let bundles: Vec<B> = backend.bundles().unwrap();
        let typed: Vec<Fr> = bundles.iter().map(&value).collect();
        let table: Vec<Fr> = backend.oracle_table(id).unwrap();
        assert_eq!(
            typed, table,
            "column {id:?} differs between the typed and id paths"
        );
    });
}

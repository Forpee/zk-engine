//! Synthetic-trace fixtures and the reference/optimized lockstep parity
//! harness shared by the optimized kernel tests.
//!
//! [`TraceBuilder`] assembles a hand-built [`IrProgram`] and its records in
//! lockstep — every row's register reads come from a running register file
//! and every RAM access from a running memory, so the rows satisfy the
//! record contracts by construction — and [`with_trace_plane`] replays them
//! into a real [`TraceBackend`].

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test support module: fail loudly"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use jolt_claims::protocols::jolt::{JoltChallengeId, JoltOneHotConfig};
use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
use jolt_field::{Field, Fr, Ring};
use jolt_poly::UnivariatePoly;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_wasm_backend::{RamAccess, RamRead, RamWrite, Record, RegisterRead, RegisterWrite};
use jolt_wasm_ir::layout::{RAM_BASE, WORD_BYTES};
use jolt_wasm_ir::row::RowModel;
use jolt_wasm_ir::{
    AluOp, Ir, IrFunction, IrProgram, MemoryLimits, Operand, Pc, Reg, Width, REGISTER_COUNT,
};
use jolt_wasm_program::{
    build_trace_rows, MemoryWord, PublicIo, WasmBytecode, WasmProgramPreprocessing,
};
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessPlane, TraceBackend};
use rand_core::SeedableRng;

use crate::{ProverInputs, SumcheckKernel};

/// The fixture's word `w` lives at this guest address; the RAM argument's
/// dense index of that word is `w`.
pub(crate) fn word_address(word: u64) -> u64 {
    RAM_BASE + WORD_BYTES * word
}

/// The lowest RAM address the RAF relation's unmap constant expects.
pub(crate) fn fixture_lowest_address() -> u64 {
    RAM_BASE
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FixtureShape {
    pub log_t: usize,
    pub ram_k: usize,
}

impl FixtureShape {
    pub fn log_k(self) -> usize {
        assert!(self.ram_k.is_power_of_two());
        self.ram_k.trailing_zeros() as usize
    }
}

/// One scripted cycle. Pre-values are replayed from the running RAM state,
/// so scripts stay trace-consistent by construction.
#[derive(Clone, Copy, Debug)]
pub(crate) enum RamOp {
    Read { word: u64 },
    Write { word: u64, post: u64 },
    None,
}

/// A program and its records, built one row at a time. The code starts with
/// the `Halt` trampoline at pc 0; a trace whose pc chain must end on the
/// padding row closes with [`jump`](Self::jump)`(0)`.
pub(crate) struct TraceBuilder {
    code: Vec<Ir>,
    records: Vec<Record>,
    regs: [u64; REGISTER_COUNT],
    ram: BTreeMap<u64, u64>,
    program_memory: Vec<MemoryWord>,
}

impl TraceBuilder {
    pub(crate) fn new() -> Self {
        Self {
            code: vec![Ir::Halt],
            records: Vec::new(),
            regs: [0; REGISTER_COUNT],
            ram: BTreeMap::new(),
            program_memory: Vec::new(),
        }
    }

    /// Seed a nonzero initial RAM word (part of the program memory image).
    pub(crate) fn initial_word(&mut self, address: u64, value: u64) {
        let _ = self.ram.insert(address, value);
        self.program_memory.push(MemoryWord { address, value });
    }

    /// Set a register without a row (the fixture's "value already held").
    pub(crate) fn set_register(&mut self, register: Reg, value: u64) {
        self.regs[register.id() as usize] = value;
    }

    fn pc(&self) -> Pc {
        self.code.len() as Pc
    }

    /// Append `ir` at the next pc with reads off the running register file;
    /// `rd_post` is the written value (applied to the file), `ram` the row's
    /// access, `next_pc` the successor.
    fn push(&mut self, ir: Ir, rd_post: u64, ram: RamAccess, next_pc: Pc) {
        let pc = self.pc();
        let spec = ir.row_spec();
        let read = |register: Reg| RegisterRead {
            register,
            value: self.regs[register.id() as usize],
        };
        let rs1 = spec.rs1.map(read);
        let rs2 = spec.rs2.map(read);
        let rd = spec.rd.map(|register| RegisterWrite {
            register,
            pre_value: self.regs[register.id() as usize],
            post_value: rd_post,
        });
        if let Some(write) = rd {
            self.regs[write.register.id() as usize] = write.post_value;
        }
        self.code.push(ir);
        self.records.push(Record {
            pc,
            next_pc,
            instruction: ir,
            rs1,
            rs2,
            rd,
            ram,
        });
    }

    pub(crate) fn nop(&mut self) {
        let next = self.pc() + 1;
        self.push(Ir::Nop, 0, RamAccess::NoOp, next);
    }

    /// `rd = mem[address]`.
    pub(crate) fn load(&mut self, rd: Reg, address: u64) {
        let value = self.ram.get(&address).copied().unwrap_or(0);
        let next = self.pc() + 1;
        self.push(
            Ir::Load {
                rd,
                base: Reg::ZERO,
                offset: address as i64,
            },
            value,
            RamAccess::Read(RamRead { address, value }),
            next,
        );
    }

    /// `mem[address] = value`, the value staged in `T0`.
    pub(crate) fn store(&mut self, address: u64, value: u64) {
        self.set_register(Reg::T0, value);
        let pre_value = self.ram.insert(address, value).unwrap_or(0);
        let next = self.pc() + 1;
        self.push(
            Ir::Store {
                base: Reg::ZERO,
                value: Reg::T0,
                offset: address as i64,
            },
            0,
            RamAccess::Write(RamWrite {
                address,
                pre_value,
                post_value: value,
            }),
            next,
        );
    }

    /// A register-only row touching the given operands: an `Add` writing
    /// `post` to `rd`, or (without `rd`) an equality `Branch` to the halt
    /// row reading both. Absent read operands read `ZERO`.
    pub(crate) fn register_op(
        &mut self,
        rd: Option<Reg>,
        rs1: Option<Reg>,
        rs2: Option<Reg>,
        post: u64,
    ) {
        let fallthrough = self.pc() + 1;
        let rs1 = rs1.unwrap_or(Reg::ZERO);
        let (ir, next) = if let Some(rd) = rd {
            (
                Ir::Alu {
                    op: AluOp::Add(Width::W64),
                    rd,
                    rs1,
                    rs2: rs2.map_or(Operand::Imm(3), Operand::Reg),
                },
                fallthrough,
            )
        } else {
            let rs2 = rs2.unwrap_or(Reg::ZERO);
            let taken = self.regs[rs1.id() as usize] == self.regs[rs2.id() as usize];
            (
                Ir::Branch {
                    op: AluOp::Eq,
                    rs1,
                    rs2,
                    target: 0,
                },
                if taken { 0 } else { fallthrough },
            )
        };
        self.push(ir, post, RamAccess::NoOp, next);
    }

    pub(crate) fn jump(&mut self, target: Pc) {
        self.push(Ir::Jump { target }, 0, RamAccess::NoOp, target);
    }

    pub(crate) fn len(&self) -> usize {
        self.records.len()
    }

    /// Package the program, its preprocessing, and the proof rows.
    pub(crate) fn finish(self, max_trace_length: usize) -> BuiltTrace {
        let mut exports = BTreeMap::new();
        let _ = exports.insert("main".to_owned(), 0);
        let mut entries = BTreeMap::new();
        let _ = entries.insert("main".to_owned(), 1);
        let program = IrProgram {
            code: self.code,
            functions: vec![IrFunction {
                entry: 1,
                params: 0,
                results: 0,
                frame_slots: 0,
            }],
            exports,
            entries,
            memory: MemoryLimits {
                initial_pages: 0,
                max_pages: 0,
            },
            globals: Vec::new(),
            data: Vec::new(),
        };
        let mut program_memory = self.program_memory;
        program_memory.sort_by_key(|word| word.address);
        let preprocessing = WasmProgramPreprocessing {
            bytecode: WasmBytecode::preprocess(&program).unwrap(),
            program_memory,
            memory: program.memory,
            max_trace_length,
        };
        let rows = build_trace_rows(&self.records).unwrap();
        BuiltTrace {
            preprocessing: Arc::new(preprocessing),
            rows,
        }
    }
}

pub(crate) struct BuiltTrace {
    pub preprocessing: Arc<WasmProgramPreprocessing>,
    pub rows: Vec<jolt_wasm_program::WasmTraceRow>,
}

pub(crate) fn one_hot_config(log_k_chunk: u8) -> JoltOneHotConfig {
    JoltOneHotConfig {
        log_k_chunk,
        lookups_ra_virtual_log_k_chunk: 16,
    }
}

/// Run `f` against a trace backend over `trace`, padded to `2^log_t`
/// cycles over a `ram_k`-word RAM domain, with `io` as the run's public I/O.
pub(crate) fn with_trace_plane<R>(
    log_t: usize,
    ram_k: usize,
    log_k_chunk: u8,
    trace: BuiltTrace,
    io: PublicIo,
    f: impl FnOnce(&TraceBackend) -> R,
) -> R {
    assert!(trace.rows.len() <= 1 << log_t, "fixture overflows 2^log_t");
    let config = JoltVmWitnessConfig::new(log_t, ram_k, one_hot_config(log_k_chunk));
    let inputs = JoltVmWitnessInputs::new(&trace.preprocessing, Arc::new(trace.rows), io);
    let backend = TraceBackend::new(config, inputs);
    f(&backend)
}

pub(crate) fn empty_io() -> PublicIo {
    PublicIo {
        entry: "main".to_owned(),
        inputs: Vec::new(),
        outputs: Vec::new(),
    }
}

/// Run `f` against a trace backend replaying `ops` (plus the jump back to
/// the halt row), padded to `2^log_t` cycles.
pub(crate) fn with_ram_fixture<R>(
    shape: FixtureShape,
    ops: Vec<RamOp>,
    f: impl FnOnce(&dyn JoltWitnessPlane<Fr>) -> R,
) -> R {
    with_ram_fixture_init(shape, Vec::new(), ops, f)
}

/// [`with_ram_fixture`] with nonzero initial RAM values: `init_words[i]`
/// seeds word `2 + i` of the program memory image, so untouched nonzero
/// words keep `RamValFinal` consistent with `val_init`.
pub(crate) fn with_ram_fixture_init<R>(
    shape: FixtureShape,
    init_words: Vec<u64>,
    ops: Vec<RamOp>,
    f: impl FnOnce(&dyn JoltWitnessPlane<Fr>) -> R,
) -> R {
    assert!(ops.len() < 1usize << shape.log_t, "script too long");
    assert!(
        init_words.is_empty() || 2 + init_words.len() <= shape.ram_k,
        "init words exceed the RAM domain"
    );
    let mut builder = TraceBuilder::new();
    for (i, &value) in init_words.iter().enumerate() {
        builder.initial_word(word_address(2 + i as u64), value);
    }
    for op in ops {
        match op {
            RamOp::Read { word } => builder.load(Reg::T1, word_address(word)),
            RamOp::Write { word, post } => builder.store(word_address(word), post),
            RamOp::None => builder.nop(),
        }
    }
    builder.jump(0);
    let trace = builder.finish(1 << shape.log_t);
    with_trace_plane(shape.log_t, shape.ram_k, 4, trace, empty_io(), |backend| {
        f(backend)
    })
}

/// Deterministic scalars for fixture points and challenges.
pub(crate) fn random_scalars(count: usize, seed: u64) -> Vec<Fr> {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
    (0..count).map(|_| Fr::random(&mut rng)).collect()
}

/// Trailing-zero-insensitive round-polynomial coefficients: the engine sums
/// members into `max_degree + 1` slots and trims the batched polynomial, so
/// a member's trailing zeros never reach the wire.
fn trimmed(poly: &UnivariatePoly<Fr>) -> Vec<Fr> {
    let mut coefficients = poly.coefficients().to_vec();
    while coefficients.last() == Some(&Fr::from_u64(0)) {
        let _ = coefficients.pop();
    }
    coefficients
}

/// Drive both kernels through the fused round loop in lockstep with the
/// same deterministic challenges, asserting per-round polynomial equality
/// (up to trailing zeros) and output-claim equality; returns the drawn
/// challenges for the caller's post-loop checks.
pub(crate) fn drive_parity_rounds<R>(
    reference: &mut dyn SumcheckKernel<Fr, Relation = R>,
    optimized: &mut dyn SumcheckKernel<Fr, Relation = R>,
    input_claim: Fr,
    inputs: &ProverInputs<'_, Fr, R>,
    challenge_seed: u64,
) -> Vec<Fr>
where
    R: ConcreteSumcheck<Fr>,
    SumcheckInputClaims<Fr, R>: InputClaims<Fr>,
    SumcheckOutputClaims<Fr, R>: OutputClaims<Fr> + PartialEq + core::fmt::Debug,
    ConcreteSumcheckChallenges<Fr, R>: SumcheckChallenges<Fr, JoltChallengeId>,
{
    let rounds = reference.num_rounds();
    assert_eq!(optimized.num_rounds(), rounds, "round count diverged");
    assert_eq!(inputs.relation.rounds(), rounds, "relation rounds diverged");

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(challenge_seed);
    let mut reference_claim = input_claim;
    let mut optimized_claim = input_claim;
    let mut challenges = Vec::with_capacity(rounds);
    let mut bind = None;
    for round in 0..rounds {
        // The reference (naive) member self-checks s(0) + s(1) against the
        // running claim, so a drifting optimized claim fails loudly here.
        let reference_poly = reference
            .prove_round(bind, round, reference_claim)
            .unwrap_or_else(|error| panic!("reference round {round}: {error}"));
        let optimized_poly = optimized
            .prove_round(bind, round, optimized_claim)
            .unwrap_or_else(|error| panic!("optimized round {round}: {error}"));
        assert_eq!(
            trimmed(&reference_poly),
            trimmed(&optimized_poly),
            "round {round} polynomial diverged"
        );
        let challenge = Fr::random(&mut rng);
        reference_claim = reference_poly.evaluate(challenge);
        optimized_claim = optimized_poly.evaluate(challenge);
        challenges.push(challenge);
        bind = Some(challenge);
    }
    if let Some(challenge) = bind {
        reference.finish_rounds(challenge).unwrap();
        optimized.finish_rounds(challenge).unwrap();
    }

    let reference_outputs = reference.output_claims(inputs.claims).unwrap();
    let optimized_outputs = optimized.output_claims(inputs.claims).unwrap();
    assert_eq!(
        reference_outputs, optimized_outputs,
        "output claims diverged"
    );
    challenges
}

/// [`drive_parity_rounds`] plus both kernels' derived-table self-checks.
pub(crate) fn assert_parity<R>(
    mut reference: Box<dyn SumcheckKernel<Fr, Relation = R>>,
    mut optimized: Box<dyn SumcheckKernel<Fr, Relation = R>>,
    input_claim: Fr,
    inputs: &ProverInputs<'_, Fr, R>,
    challenge_seed: u64,
) where
    R: ConcreteSumcheck<Fr>,
    SumcheckInputClaims<Fr, R>: InputClaims<Fr>,
    SumcheckOutputClaims<Fr, R>: OutputClaims<Fr> + PartialEq + core::fmt::Debug,
    ConcreteSumcheckChallenges<Fr, R>: SumcheckChallenges<Fr, JoltChallengeId>,
{
    let challenges = drive_parity_rounds(
        reference.as_mut(),
        optimized.as_mut(),
        input_claim,
        inputs,
        challenge_seed,
    );
    let output_points = inputs
        .relation
        .derive_opening_points(&challenges, inputs.points)
        .unwrap();
    reference
        .validate_derived_tables(
            inputs.relation,
            inputs.points,
            &output_points,
            inputs.challenges,
        )
        .unwrap();
    optimized
        .validate_derived_tables(
            inputs.relation,
            inputs.points,
            &output_points,
            inputs.challenges,
        )
        .unwrap();
}

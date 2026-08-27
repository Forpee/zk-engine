//! End-to-end over the WASM stack: a WAT module through the frontend,
//! executed on the backend, proved by the modular prover, and accepted by
//! `jolt-verifier` — plus a tampered public output the verifier rejects.
//! Under the `zk` feature the same run produces a BlindFold proof.

#![expect(clippy::expect_used, reason = "integration tests")]

use std::sync::Arc;

use jolt_crypto::{Bn254G1, DeriveSetup, Pedersen, PedersenSetup};
use jolt_dory::DoryScheme;
use jolt_field::Fr;
use jolt_prover::{preprocess_program, JoltBackend, JoltProverPreprocessing, PreparedRun};
use jolt_transcript::Blake2bTranscript;
use jolt_verifier::config::MAX_BLINDFOLD_GENERATORS;
use jolt_verifier::preprocessing::ProgramPreprocessing;
use jolt_verifier::proof::JoltProof;
use jolt_verifier::JoltVerifierPreprocessing;
use jolt_wasm_frontend::WasmModule;
use jolt_wasm_program::PublicIo;

const MAX_TRACE_LENGTH: usize = 1 << 12;

/// Iterative fibonacci over `n` steps: the `n`-th term modulo 2^64.
const FIB: &str = r#"
(module
  (func (export "fib") (param $n i64) (result i64)
    (local $a i64) (local $b i64) (local $i i64) (local $t i64)
    (local.set $b (i64.const 1))
    (block $done
      (loop $top
        (br_if $done (i64.ge_u (local.get $i) (local.get $n)))
        (local.set $t (i64.add (local.get $a) (local.get $b)))
        (local.set $a (local.get $b))
        (local.set $b (local.get $t))
        (local.set $i (i64.add (local.get $i) (i64.const 1)))
        (br $top)))
    (local.get $a)))
"#;

type Proof = JoltProof<DoryScheme, Pedersen<Bn254G1>>;
type VerifierPreprocessing = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;

fn prove_fib(
    backend: JoltBackend<Fr, DoryScheme>,
    n: u64,
) -> (VerifierPreprocessing, PublicIo, Proof) {
    let bytes = wat::parse_str(FIB).expect("wat");
    let program = WasmModule::decode(&bytes)
        .expect("decode")
        .lower()
        .expect("lower");
    let (preprocessing, digest) =
        preprocess_program(&program, MAX_TRACE_LENGTH).expect("preprocess");
    let preprocessing = Arc::new(preprocessing);
    let run = PreparedRun::execute::<Fr>(&program, &preprocessing, "fib", &[n]).expect("execute");
    assert_eq!(run.io.outputs, vec![fib(n)]);

    // The SRS covers the largest trace the preprocessing admits.
    let total_vars = 4 + MAX_TRACE_LENGTH.ilog2() as usize;
    let prover_setup = DoryScheme::setup_prover(total_vars);
    let vc_setup = PedersenSetup::<Bn254G1>::derive(&prover_setup, MAX_BLINDFOLD_GENERATORS);
    let verifier = JoltVerifierPreprocessing::new(
        ProgramPreprocessing::Full(Arc::clone(&preprocessing)),
        digest,
        DoryScheme::setup_verifier(total_vars),
        Some(vc_setup),
    );
    let prover_preprocessing = JoltProverPreprocessing {
        verifier,
        pcs_setup: prover_setup,
        committed_program: None,
    };
    let witness = run.witness(&preprocessing);
    let proof = jolt_prover::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
        &backend,
        &prover_preprocessing,
        &run.config,
        None,
        &witness,
        &run.io,
    )
    .expect("prove");
    (prover_preprocessing.verifier, run.io, proof)
}

fn fib(n: u64) -> u64 {
    let (mut a, mut b) = (0u64, 1u64);
    for _ in 0..n {
        let t = a.wrapping_add(b);
        a = b;
        b = t;
    }
    a
}

fn verify(
    preprocessing: &VerifierPreprocessing,
    io: &PublicIo,
    proof: &Proof,
) -> Result<(), jolt_verifier::VerifierError> {
    jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
        preprocessing,
        io,
        proof,
        None,
    )
}

/// BlindFold verification recurses over a large folded R1CS — run on a wide
/// stack.
fn with_wide_stack<R: Send + 'static>(body: impl FnOnce() -> R + Send + 'static) -> R {
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(body)
        .expect("spawn test thread")
        .join()
        .expect("test thread panicked")
}

#[test]
fn fibonacci_proof_is_accepted_by_the_optimized_backend() {
    with_wide_stack(|| {
        let (preprocessing, io, proof) = prove_fib(JoltBackend::optimized(), 40);
        verify(&preprocessing, &io, &proof).expect("proof must verify");
    });
}

#[test]
fn fibonacci_proof_is_accepted_by_the_reference_backend() {
    with_wide_stack(|| {
        let (preprocessing, io, proof) = prove_fib(JoltBackend::reference(), 12);
        verify(&preprocessing, &io, &proof).expect("proof must verify");
    });
}

#[test]
fn tampered_public_output_is_rejected() {
    with_wide_stack(|| {
        let (preprocessing, mut io, proof) = prove_fib(JoltBackend::optimized(), 40);
        io.outputs[0] += 1;
        assert!(verify(&preprocessing, &io, &proof).is_err());
    });
}

#[test]
fn wrong_entry_is_rejected() {
    with_wide_stack(|| {
        let (preprocessing, mut io, proof) = prove_fib(JoltBackend::optimized(), 40);
        io.entry = "missing".to_owned();
        assert!(verify(&preprocessing, &io, &proof).is_err());
    });
}

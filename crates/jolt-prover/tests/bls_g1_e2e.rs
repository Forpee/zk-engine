//! The MVP use case: blst's BLS12-381 G1 scalar multiplication compiled to
//! WebAssembly (`guests/bls-g1`, fixture `fixtures/bls_g1.wasm`) proved
//! end to end. The public inputs are the scalar's four limbs, the public
//! outputs the six words of the compressed point `[s]·G1`, checked against
//! native blst; a full-width scalar is executed (not proved) to pin the
//! cycle budget the profiling workload is sized from.

#![expect(clippy::expect_used, reason = "integration tests")]

use std::sync::Arc;

use blst::{blst_p1, blst_p1_compress, blst_p1_generator, blst_p1_mult};
use jolt_crypto::{Bn254G1, DeriveSetup, Pedersen, PedersenSetup};
use jolt_dory::DoryScheme;
use jolt_field::Fr;
use jolt_prover::{preprocess_program, JoltBackend, JoltProverPreprocessing, PreparedRun};
use jolt_transcript::Blake2bTranscript;
use jolt_verifier::config::MAX_BLINDFOLD_GENERATORS;
use jolt_verifier::preprocessing::ProgramPreprocessing;
use jolt_verifier::JoltVerifierPreprocessing;
use jolt_wasm_backend::Machine;
use jolt_wasm_frontend::WasmModule;
use jolt_wasm_ir::IrProgram;

const BLS_G1_WASM: &[u8] = include_bytes!("fixtures/bls_g1.wasm");
const ENTRY: &str = "g1_mul";
/// The proved run: a 3-bit scalar keeps the trace at 2^21.
const PROVED_SCALAR: [u64; 4] = [5, 0, 0, 0];
const MAX_TRACE_LENGTH: usize = 1 << 21;

fn program() -> IrProgram {
    WasmModule::decode(BLS_G1_WASM)
        .expect("decode")
        .lower()
        .expect("lower")
}

/// `[s]·G1` compressed, as the six little-endian words the guest returns.
fn expected_words(scalar: [u64; 4]) -> Vec<u64> {
    let mut bytes = [0u8; 32];
    for (chunk, limb) in bytes.chunks_exact_mut(8).zip(scalar) {
        chunk.copy_from_slice(&limb.to_le_bytes());
    }
    let mut point = blst_p1::default();
    let mut compressed = [0u8; 48];
    // SAFETY: valid static generator; buffers of the sizes blst requires.
    unsafe {
        blst_p1_mult(&raw mut point, blst_p1_generator(), bytes.as_ptr(), 256);
        blst_p1_compress(compressed.as_mut_ptr(), &raw const point);
    }
    compressed
        .chunks_exact(8)
        .map(|w| u64::from_le_bytes(w.try_into().expect("8 bytes")))
        .collect()
}

#[test]
fn g1_mul_matches_native_blst_for_full_width_scalars() {
    let program = program();
    for scalar in [
        [12345, 0, 0, 0],
        [u64::MAX, u64::MAX, u64::MAX, u64::MAX >> 1],
        [
            0x1234_5678_9abc_def0,
            0xfedc_ba98_7654_3210,
            0xdead_beef,
            0x0123_4567,
        ],
    ] {
        let run = Machine::new(&program)
            .expect("machine")
            .invoke(ENTRY, &scalar)
            .expect("execute");
        assert_eq!(run.results, expected_words(scalar), "scalar {scalar:x?}");
        assert!(run.terminated);
    }
}

#[test]
fn g1_mul_proof_is_accepted() {
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(|| {
            let program = program();
            let (preprocessing, digest) =
                preprocess_program(&program, MAX_TRACE_LENGTH).expect("preprocess");
            let preprocessing = Arc::new(preprocessing);
            let run = PreparedRun::execute::<Fr>(&program, &preprocessing, ENTRY, &PROVED_SCALAR)
                .expect("execute");
            assert_eq!(run.io.outputs, expected_words(PROVED_SCALAR));

            let total_vars = run.config.commitment_total_vars(None);
            let prover_setup = DoryScheme::setup_prover(total_vars);
            let vc_setup =
                PedersenSetup::<Bn254G1>::derive(&prover_setup, MAX_BLINDFOLD_GENERATORS);
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
            let proof =
                jolt_prover::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
                    &JoltBackend::optimized(),
                    &prover_preprocessing,
                    &run.config,
                    &witness,
                    &run.io,
                )
                .expect("prove");
            jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
                &prover_preprocessing.verifier,
                &run.io,
                &proof,
            )
            .expect("proof must verify");

            // The outputs are bound: a forged point is rejected.
            let mut forged = run.io.clone();
            forged.outputs[0] ^= 1;
            assert!(
                jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
                    &prover_preprocessing.verifier,
                    &forged,
                    &proof,
                )
                .is_err()
            );
        })
        .expect("spawn")
        .join()
        .expect("test thread panicked");
}

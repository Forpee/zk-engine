# Jolt zkWASM

![imgs/jolt_alpha.png](imgs/jolt_alpha.png)

A **zkWASM** built on the Jolt proving stack. A WebAssembly module is decoded
and lowered to a register-machine IR (`crates/jolt-wasm-frontend`), executed
into one record per instruction (`crates/jolt-wasm-backend`), and proved with
Jolt's sumcheck-based protocols — Spartan over a uniform R1CS, the Twist/Shout
memory and lookup arguments, Dory commitments, and BlindFold for zero
knowledge (`crates/jolt-prover`, `crates/jolt-verifier`).

The design is described in [`specs/zkwasm-frontend.md`](specs/zkwasm-frontend.md);
the crate map, row model, and guest layout are in [`CLAUDE.md`](CLAUDE.md).

## Papers

- [Jolt: SNARKs for Virtual Machines via Lookups](https://eprint.iacr.org/2023/1217) — Arasu Arun, Srinath Setty, Justin Thaler
- [Twist and Shout: Faster memory checking arguments via one-hot addressing and increments](https://eprint.iacr.org/2025/105) — Srinath Setty, Justin Thaler
- [Unlocking the lookup singularity with Lasso](https://eprint.iacr.org/2023/1216) — Srinath Setty, Justin Thaler, Riad Wahby

## Building

The Rust toolchain is pinned in [rust-toolchain.toml](./rust-toolchain.toml);
`rustup` picks it up on the first `cargo` invocation.

```bash
cargo build --release -p jolt-prover
```

## Testing

Always `cargo nextest`, never `cargo test`:

```bash
cargo nextest run --workspace --cargo-quiet

# The end-to-end check: fibonacci WAT → IR → execution → prove → verify,
# plus tamper rejection, in the clear and in ZK
cargo nextest run -p jolt-prover --cargo-quiet -E 'binary(wasm_e2e)'
cargo nextest run -p jolt-prover --features zk --cargo-quiet -E 'binary(wasm_e2e)'
```

## The MVP use case

`guests/bls-g1` compiles blst's BLS12-381 G1 scalar multiplication to
WebAssembly; `[s]·G1` is proved with the scalar as public input and the
compressed point as public output, checked against native blst:

```bash
cargo nextest run -p jolt-prover --cargo-quiet -E 'binary(bls_g1_e2e)'
cargo run --release -p jolt-prover --features profiling -- profile --name bls-g1 --scale 25
```

A 255-bit scalar is a 2^25 trace: ~2 minutes and ~7 GB to prove on a laptop,
0.2 s to verify. `scripts/build_guests.sh` rebuilds the guest fixture.

## Profiling

```bash
cargo run --release -p jolt-prover --features profiling -- profile --name fibonacci --format chrome
```

writes `benchmark-runs/<timestamp>_modular_fibonacci_16/` with `trace.json`
(open in [Perfetto](https://ui.perfetto.dev/)), `summary.json`, and
`memory.html`. `--scale <log2 trace length>` sets the trace size;
`--backend optimized` selects the performance kernels over the reference
oracle. `--features profiling,allocative` adds per-batch heap snapshots.
See `CLAUDE.md` for the multi-scale sweep and the summary schema.

## Acknowledgements

Built on [a16z/jolt](https://github.com/a16z/jolt), which started as a fork of
<https://github.com/arkworks-rs/spartan> (original Spartan
[code](https://github.com/microsoft/Spartan) by Srinath Setty).

## Licensing

Dual licensed under the MIT License ([LICENSE-MIT](LICENSE-MIT)) and the
Apache License ([LICENSE-APACHE](LICENSE-APACHE)) at your discretion.
Jolt is Copyright (c) a16z 2023; portions of the codebase are modifications
or ports of third-party code as indicated in the applicable code headers.

## Disclaimer

_This code is being provided as is. No guarantee, representation or warranty is being made, express or implied, as to the safety or correctness of the code. It has not been audited and as such there can be no assurance it will work as intended, and users may experience delays, failures, errors, omissions or loss of transmitted information. Use at your own risk._

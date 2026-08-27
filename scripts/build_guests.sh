#!/usr/bin/env bash
# Rebuild the WebAssembly guest fixtures from `guests/` (needs
# `rustup target add wasm32-unknown-unknown`; blst's C sources need a clang
# with a wasm32 target, e.g. Apple's or Homebrew LLVM).
set -euo pipefail
cd "$(dirname "$0")/.."
(cd guests/bls-g1 && cargo build --release)
cp guests/bls-g1/target/wasm32-unknown-unknown/release/bls_g1_guest.wasm \
   crates/jolt-prover/tests/fixtures/bls_g1.wasm
ls -la crates/jolt-prover/tests/fixtures/

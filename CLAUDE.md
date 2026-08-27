# CLAUDE.md

## Project Overview

This repository is a **zkWASM** built on the Jolt proving stack. A WebAssembly
module is decoded and lowered to a register-machine IR, executed into one
record per instruction, and proved with Jolt's sumcheck-based protocols
(Spartan over a uniform R1CS, the Twist/Shout memory and lookup arguments,
Dory commitments, BlindFold for zero knowledge). The RISC-V target is gone:
every crate speaks the WASM row model.

## Essential Commands

### Linting and Formatting

```bash
# Must pass in the default build and in the ZK/profiling build
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p jolt-verifier -p jolt-kernels -p jolt-prover --all-targets \
  --features jolt-prover/zk,jolt-prover/profiling,jolt-prover/allocative -- -D warnings
cargo fmt --all
```

### Testing

```bash
# Always cargo nextest, never cargo test
cargo nextest run --workspace --cargo-quiet

# Run specific test in specific package
cargo nextest run -p [package_name] [test_name] --cargo-quiet

# Primary correctness check — the WASM end-to-end (fibonacci WAT → IR →
# execution → prove → verify, plus tamper rejection), clear and ZK
cargo nextest run -p jolt-prover --cargo-quiet -E 'binary(wasm_e2e)'
cargo nextest run -p jolt-prover --features zk --cargo-quiet -E 'binary(wasm_e2e)'

# Kernel parity suites (reference vs optimized tiers) and the verifier/claims units
cargo nextest run -p jolt-kernels -p jolt-verifier -p jolt-claims -p jolt-witness --cargo-quiet
```

### Profiling

```bash
# Emits benchmark-runs/{timestamp}_modular_{name}_{scale}/ containing trace.json
# (Perfetto UI / trace_processor SQL), summary.json (machine-queryable), and memory.html,
# with benchmark-runs/latest_modular_{name}_{scale} symlinked to the newest successful run.
cargo run --release -p jolt-prover --features profiling -- profile --name fibonacci --format chrome
# --name options: fibonacci (default scale 16; iterative fibonacci WAT, 17 rows/iteration)
# --scale <log2 trace length> overrides; --format none = no-subscriber Instant baseline
# --backend reference (default, naive test oracle) | optimized (performance tier);
# optimized artifacts get an _optimized suffix on the run dir and latest_ symlink

jq '.stages | map({label, s: (.wall_time_ns/1e9)})' benchmark-runs/latest_modular_fibonacci_16/summary.json

# Multi-scale sweep (one profile subprocess per run; results in benchmark-runs/modular_timings.csv)
cargo run --release -p jolt-prover --features profiling -- benchmark --min-scale 18 --max-scale 21 --resume

# Per-batch heap snapshots (*.folded in the run directory; rendered by memory.html)
cargo run --release -p jolt-prover --features profiling,allocative -- profile --name fibonacci --format chrome

# Smoke test of the harness (scale 2^13)
cargo nextest run -p jolt-prover --features profiling -E 'binary(profiling_smoke)'
```

The span taxonomy (versioned, normative) lives in `crates/jolt-profiling/src/taxonomy.rs` — renaming a span is a schema change (summary keys and `telemetry:*` objectives break; the profiling smoke test enforces label presence).

## Architecture

### Crate Structure

**WASM stack** (`specs/zkwasm-frontend.md`):

- `jolt-wasm-ir` — the shared IR contract: `Ir`/`IrProgram` (each instruction reads ≤2 registers, writes ≤1, touches ≤1 aligned RAM word, static immediate), `AluOp` (the table catalog), `RowFlag`/`RowSpec` (the proof-row model), `WasmBytecode`/`BytecodeRow` (the committed bytecode table), and `layout` (the guest address space).
- `jolt-wasm-frontend` — `WasmModule::decode` (wasmparser + validator, static stack heights) and `lower` (instruction selection onto the row model: register-allocated operand stack, shadow call stack, entry stubs per export).
- `jolt-wasm-backend` — `Memory`, the `Machine` emitting one `Record` per instruction, `check_record` (constraint-form row contract).
- `jolt-wasm-tables` — `WasmTable`, the WebAssembly lookup-table catalog over `jolt-lookup-tables`' prefix–suffix tables (`clz`/`ctz`/`popcnt` added for WASM).
- `jolt-wasm-program` — proof-side preprocessing: `WasmProgramPreprocessing`, `WasmTraceRow` (64-byte proof row), `PublicIo`, the public memory model (`PublicIoMemory`, `PublicInitialRam`, `min_ram_k`/`max_ram_k`).

**Proof stack**: `jolt-claims` (protocol vocabulary: ids, geometry, symbolic relations), `jolt-witness` (`TraceBackend` witness plane over `WasmTraceRow`s), `jolt-r1cs` (`constraints::wasm`, the uniform R1CS), `jolt-verifier` (stages 1–8, BlindFold ZK path, `verify`), `jolt-kernels` (reference and optimized prover kernels), `jolt-prover` (orchestration: `preprocess_program`, `PreparedRun`, `prove`, the profiling harness), plus `jolt-sumcheck`, `jolt-poly`, `jolt-blindfold`, `jolt-openings`, `jolt-dory`, `jolt-crypto`, `jolt-field`, `jolt-transcript`, `jolt-utils`, `jolt-profiling`, and the derive crates.

Arkworks dependencies use a fork: `a16z/arkworks-algebra` branch `dev/twist-shout`, pinned in the root `Cargo.toml`.

### Row model and guest layout

- 128 registers: `ZERO=0, SP=1, RA=2, T0..T4=3..7`, frame slots from `FRAME_BASE=8` (`MAX_FRAME_SLOTS=120`, `MAX_RESULTS=5`). `i32` values are canonical zero-extended.
- `RowFlag` (15 flags, bit order = column order): `LeftIsRs1, RightIsRs2, RightIsImm, AddOperands, SubOperands, MulOperands, WriteLookupToRd, Load, Store, Jump, Branch, Assert, Halt, Trap, Advice`. Branch/jump targets are absolute; `Halt` at pc 0 is the padding row.
- Guest RAM, contiguous from `RAM_BASE = 0x9000_0000` (dense word index `(addr − RAM_BASE)/8`): system words (memory size, termination), public inputs, public outputs, 1 MiB shadow stack + guard page, globals, linear memory. Public I/O is memory: `PublicIo{entry, inputs, outputs}`; the verifier binds the initial memory (program image + inputs) and the final I/O window.
- WASM R1CS: 32 variables (const + 16 inputs + 15 flags), 22 eq rows + 2 product rows; uni-skip domain 11. Spartan outer/product/shift and the instruction-input, bytecode read-RAF (5 gamma-folded stages), register/RAM, and instruction-lookup sumchecks are the WASM-vocabulary relations in `jolt-claims`.

### Prover Pipeline

1. **Frontend**: WAT/WASM → `WasmModule::decode` → `lower` → `IrProgram`
2. **Preprocess**: `jolt_prover::preprocess_program` → `WasmProgramPreprocessing` + digest
3. **Execute**: `PreparedRun::execute` (backend `Machine`) → `Vec<WasmTraceRow>`, `PublicIo`, `ProverConfig`
4. **Witness**: `TraceBackend` over the rows (committed polynomials: `RdInc`, `RamInc`, one-hot `InstructionRa`/`BytecodeRa`/`RamRa`)
5. **Stages 0–8**: commitments, Spartan uni-skips + outer/product remainders, shift + instruction input + register reductions, RAM/registers read-write and val checks, instruction read-RAF, bytecode read-RAF, claim reductions, the joint Dory opening
6. **ZK** (`zk` feature): every sumcheck round Pedersen-committed, BlindFold over the verifier R1CS replaces cleartext claims

### ZK Feature Gate

The `zk` Cargo feature selects the zero-knowledge protocol at compile time in `jolt-verifier`, `jolt-kernels`, and `jolt-prover`; the proof self-describes its protocol (`JoltProof::protocol`) and `validate_proof_config` rejects a mismatch fail-closed. Every sumcheck relation defines both its claim computation and its BlindFold constraint (`input_claim` / `input_claim_constraint`, `output_claim` / `output_claim_constraint`); any change to one requires the matching change to the other — the `wasm_e2e` zk test catches drift.

## Development Guidelines

### Performance

- Profile before optimizing
- Benchmark changes to `poly/` code — small regressions multiply across thousands of sumcheck rounds
- Use `#[inline]` judiciously in hot paths
- Pre-allocate vectors unsafely when size is known; avoid clones in hot paths
- Hot trace paths get one pass and one owner of trace-sized storage. Produce or share derived rows during the existing pass instead of walking or materializing the trace again.

### Prover Hot Paths

- Sumcheck inner loop dominates: polynomial bind, sumcheck_evals, eq_poly evals
- `CompactPolynomial` bind converts small scalars to field elements — keep scalars small
- `SharedRaPolynomials` avoids per-polynomial memory duplication for RA indices

### Code Style Invariants

- Use `non_snake_case` for math variables: `log_T`, `ram_K`, `log_K`, etc.
- **Machine-checked, repo-wide:** one `cfg_attr` per predicate per item; fold adjacent `#[cfg_attr(P, A)]` `#[cfg_attr(P, B)]` into `#[cfg_attr(P, A, B)]`.
- **Machine-checked, repo-wide:** `#[allocative(visit = ...)]` never decorates a container of primitives (`Vec<u32>`, `Vec<Vec<usize>>`, `Vec<Option<u8>>`, ...). Native impls report element types and unused capacity; `jolt_poly::visit_scalars`/`visit_scalar_rows` exist only to avoid the `F: Allocative` bound on foreign-scalar containers.
- **Machine-checked on added lines:** import types, traits, enums, constants, and PascalCase macros; reference them by short name. Keep enum variants qualified by the imported enum type (`Kind::ADD`, not bare `ADD`). Import singleton paths directly (`use x::Kind`, never `use x::Kind::{self}`). Lowercase namespace free functions and macros (`std::mem::take`, `tracing::info!`) may stay qualified. Fully qualified paths remain valid for ambiguity, attribute arguments (`allocative(visit = jolt_poly::visit_scalars)`), and macro bodies. Once a path is imported, never spell it qualified in the same file.
- Alias an instruction-kind enum as `Kind` at emitter call sites and write `Kind::INSTRUCTION`; never qualify emitted instructions with `SourceInstructionKind`, `JoltInstructionKind`, or a module path.
- Give each protocol formula, geometry or sizing law, schedule, and state transition one owner. Consumers call the canonical implementation instead of mirroring or open-coding it. Tests use independent ground truth or the production computation; never a second implementation of the same rule as their oracle.
- Encode correlated state and absence with typed requests, structs, enums, and `Option`; never value sentinels, decomposed arguments, or parallel options that can disagree.
- Serialized or ordinal enums are append-only; keep feature-gated and test-only variants last so feature selection cannot shift real discriminants.
- Enforce public-boundary invariants at the point of fault in release builds with typed errors. Use `debug_assert` only for properties already pinned by types or release checks; keep recoverable capability gaps separate from invariant violations.
- Derive trait impls instead of hand-rolling; exhaust derive escapes (`#[allocative(bound = "F: JoltField")]`, `visit = ...`, `skip`) first. Hand-write only what a derive cannot express, keep it local to the one type that needs it, and size buffers by `capacity()`, not `len()`.
- A free function is pure or shared across ≥2 callers. Otherwise make it a method on the type whose state it uses; inline it when no type owns the behavior.
- No public API, abstraction, mode, or state container without an in-repo production caller or documented external contract; add it with its first use. Lazy-init globals and error slots are valid, but speculative lifecycle guards and unreachable transitions are not.
- State enforcement honestly in docs: never describe a property as constraint-enforced when it holds only for the honest encoder or under a `debug_assert`; name the mechanism and location that pins each invariant. If reviewers independently misread a deliberate gap, the missing argument belongs in a comment.
- Make names track current semantics: rename vocabulary when an encoding changes (`UnsignedIncMsb` → `BalancedIncCarry`); keep no compatibility names.
- Add `cfg`/`cfg_attr` gates only where the build requires them.
- Before PR handoff, audit every added test and helper. Remove development-only probes, ignored tests, temporary benchmarks, diagnostic counters or histograms, and one-off fuzz or parity scaffolding. Keep permanent tests only when they add a distinct failure signal beyond existing tests, golden fixtures, or CI. Make a worthwhile manual diagnostic an intentional tool or benchmark with a documented command.

### Testing Guidelines

- Do not add old-vs-new equivalence tests that reimplement the pre-change logic as the oracle. Transition-validation belongs in the PR process (byte-parity CI vs a living reference, one-off scripts), not the permanent suite. Permanent tests must assert against independent ground truth: spec vectors, golden fixtures, live reference paths (e.g. `jolt-kernels`' reference tier, the legacy-prover byte-parity suites), or properties. If the old code is deleted, its reimplementation in a test is dead weight — delete the test rather than keep the old logic alive inside it. A `#[cfg(test)]` copy of superseded production code "kept as the oracle" is the same anti-pattern.

### Lint Policy

- Workspace enforces `allow_attributes = "deny"` — use `#[expect(...)]` instead of `#[allow(...)]`
- The jolt-verifier runtime closure (18 crates, listed in `specs/verifier-closure-lints.md`) carries stricter crate-root lints: panic-source denies (`indexing_slicing` in control-plane crates, `panic_in_result_fn`, `wildcard_enum_match_arm`, ...), `forbid(unsafe_code)` where a crate has no unsafe, and numeric-discipline denies in jolt-verifier itself — which additionally denies `unreachable`, the only abort macro that escapes both `panic` and `panic_in_result_fn`. New code in those crates must fix the lint or add `#[expect(clippy::..., reason = "...")]` at the narrowest scope with a real justification
- `.unwrap()` and `.expect()` are fine in tests. In non-test code, avoid them unless the alternative significantly hurts readability (e.g., infallible fixed-size array conversions). When used, annotate the function with `#[expect(clippy::unwrap_used)]` or `#[expect(clippy::expect_used)]`
- Use `#[expect(clippy::...)]` on test modules to blanket-suppress test-inappropriate lints rather than per-function annotations

### Comments

Match the codebase's low comment density. Worth writing: WHY comments, WARNING for non-obvious gotchas, SAFETY on unsafe blocks, algorithm explanations (link to paper if applicable), public API docs stating behavior or invariants.
Do not narrate code or test assertions. If a comment only restates an expression, make the code self-documenting instead.

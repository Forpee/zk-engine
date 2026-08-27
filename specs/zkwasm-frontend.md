# Spec: zkWASM Frontend (IR and Execution Record)

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Claude, @Forpee                |
| Created     | 2026-08-27                     |
| Status      | implemented (frontend, catalog, bytecode, trace row, R1CS, program/IO model) |
| PR          |                                |

## Summary

Jolt is being re-targeted from RISC-V to WebAssembly. The proving backend
(Twist/Shout memory checking, instruction lookups, Spartan over per-row flags)
consumes a *register-machine row* — `rs1`/`rs2` reads, one `rd` write, one RAM
access, an immediate, and a bytecode pc — not RISC-V per se. The frontend's
job is therefore to turn a WebAssembly module into a program over exactly that
row shape and to produce one execution record per step. This spec covers the
first slices: four crates — `jolt-wasm-ir` (the shared IR contract),
`jolt-wasm-frontend` (decode, validate, lower), `jolt-wasm-backend`
(memory, interpreter emitting `Record`s, proof-row model), and
`jolt-wasm-tables` (the WebAssembly lookup-table catalog, realizing every IR
ALU op as a prefix–suffix-decomposable Jolt table), and `jolt-wasm-program`
(proof-side preprocessing: the committed bytecode table, the initial memory
image, and the compact proof-facing trace row). The modular prover/verifier (`jolt-claims`, `jolt-witness`, `jolt-verifier`,
`jolt-kernels`, `jolt-prover`) speak this row model natively; the RISC-V
crates are gone.

## Intent

### Goal

Build the WASM stack as three crates with one owner each, where the
frontend and backend share only the IR crate:

- `jolt-wasm-ir`: operator enums, `Ir`/`IrProgram` (the lowered **final**
  universe: each instruction reads ≤2 registers, writes ≤1 register, touches
  ≤1 memory location, and has a static immediate — the proof-row shape), and
  the guest address-space layout constants.
- `jolt-wasm-frontend`: `WasmModule` (`module.rs`, `source.rs`) — the
  decoded, validated **source** universe, every operator carrying the
  operand-stack height before it executes (from `wasmparser`'s
  `FuncValidator`) — and `WasmModule::lower` (`lower.rs`).
- `jolt-wasm-backend`: `Memory`, the `Machine` emitting a `Record` per IR
  instruction (`pc`, `next_pc`, the instruction, `rs1`/`rs2` reads, the `rd`
  write with pre/post values, the RAM access with pre/post values), and the
  proof-row model (`row.rs`).

### Invariants

- **Static register assignment.** Frame slot `i` of a function (parameters,
  then declared locals, then operand-stack depth) is register
  `FRAME_BASE + i`. Validation makes the stack height static at every
  instruction, so no IR instruction ever indexes a register dynamically; the
  bytecode row alone determines `rs1`/`rs2`/`rd` ids.
- **Reserved registers.** `r0 = ZERO` (hardwired; the machine drops writes,
  the lowering never emits one), `r1 = SP` (shadow-stack pointer), `r2 = RA`,
  `r3..=r7 = T0..=T4` (temporaries; carry function results across `return`).
  `REGISTER_COUNT = 128` so ids fit the existing one-hot register layout.
- **Frame bound.** A function needs `params + locals + max_stack_height ≤ 120`
  slots and `results ≤ 5`; otherwise lowering fails with a typed
  `LowerError` (register spilling is deferred, see Non-Goals).
- **Canonical values.** Registers hold `i32` values zero-extended to 64 bits
  and `i64` values as-is; every `W32` op produces a zero-extended result. This
  keeps `i64.extend_i32_u` a `Move` and makes linear-memory addressing
  (`LINEAR_MEMORY_BASE + addr + offset`) overflow-free in `u64`.
- **One address space, inside Jolt's RAM window.** From `RAM_BASE`
  (`0x9000_0000`) upward and contiguous: system words (memory size,
  termination), public inputs, public outputs, the 1 MiB shadow stack plus
  its guard page, globals (8 bytes each), the function table (16 bytes per
  slot: the callee's entry pc and its canonical signature id + 1, `0` for
  null), and linear memory. The RAM
  argument's word index is the dense `(address − RAM_BASE) / 8`
  (`layout::remap_word_address`); a record's RAM address is always absolute.
  System and I/O words sit at the bottom so a program that touches little
  memory proves over a small RAM domain (`jolt-wasm-program::min_ram_k`). Shadow-stack exhaustion traps as
  `CallStackExhausted`.
- **Program/IO model.** Public I/O is memory: the host writes the entry's
  arguments to the input words before execution and reads the results from
  the output words and the termination word after; `PublicIo` gives the
  initial image (program words + inputs) and the final public words (outputs
  + termination = 1). Every export gets a synthesized **entry stub**
  (`IrProgram::entries`): from an all-zero register file it sets `SP`, runs
  the `start` function, loads the parameters from the input words, calls the
  function through the normal calling convention, stores the results to the
  output words, sets the termination word, and jumps to `Halt`. A trace is
  therefore one contiguous segment beginning at the stub with zero
  registers — no host-initialized state inside the proven execution.
- **`memory.grow` is plain rows**: load the size word, add `delta` pages,
  `LeU` against the static page cap (an immediate), branch; on success store
  the new size and return the old size in pages, else `u32::MAX`. The
  interpreter's linear backing store follows writes to the size word. There
  is no special row class, so the lattice-mode store/rd-write disjointness
  holds for every row.
- **Doubleword RAM contract.** Every `Record` RAM access is one naturally
  aligned 64-bit word (`Ir::Load`/`Ir::Store`), matching Twist's
  doubleword-addressable argument (`specs/byte-addressable-memory.md` keeps
  it so). A WebAssembly access of any width/alignment lowers to: effective
  address, `AssertBounds` against the size word, floor to the containing
  doubleword, word load/store of that doubleword *and the next*, and
  `Ir::Subword` extract/insert ops (two-input, lookup-table-shaped:
  `ExtractLo/Hi`, `InsertLo/Hi`, `ClearLo/Hi`). For a non-crossing access
  the `Hi` ops are identities, and linear memory carries one zeroed slack
  word past its end so the second word is always in bounds. Cost: 10 rows
  per load, 14 per store — the target for the fused-table tiers of the
  byte-addressable spec. `memory.size` is a load of the size word;
  `memory.grow` is one row whose RAM write rewrites it.
- **Calling convention** (`lower.rs` module docs are the owner): the caller
  spills its live frame slots and the return address to the shadow stack,
  loads the callee's parameters from the spilled argument slots, bumps `SP`,
  and jumps; the callee zeroes its declared locals on entry and on `return`
  moves results to `T0..`, reloads `RA` from `SP - 8`, and `JumpReg`s; the
  caller then restores `SP` and its live slots below the arguments and moves
  the results into place. The callee never touches `SP` net, so `return` is
  caller-agnostic. `call_indirect` runs the same sequence with the table
  index (an operand-stack slot above the arguments) resolved right before
  the jump: bounds-`Assert` against the table size, load the slot's
  signature word and `Assert` it equals the expected canonical signature
  id + 1 (structural equality of function types; a null slot's `0` never
  matches), load the entry pc into `RA`, `JumpReg RA`.
- **Operand forwarding.** `local.get` and `*.const` emit no row: the stack
  slot is marked pending and the consumer reads the local's register (or
  takes the constant as its immediate); an operator followed by
  `local.set`/`tee` writes the local directly. Pending slots are
  materialized before control flow, calls, `return`, and a write to the
  aliased local (`crates/jolt-wasm-frontend/src/lower.rs`, `Pending`).
- **One RAM cell per wasm word.** Linear memory maps wasm address `a` to
  guest address `LINEAR_MEMORY_BASE + 2a` (`layout::linear_address`); each
  8-byte cell holds one 4-byte wasm word zero-extended (the backend rejects
  larger cell writes as a lowering invariant violation). Bounds are one
  `Assert LtU t < LIMIT_w` against the reserved registers `LIMIT_B/H/W/D`
  (`linear_address(size) − 2w + 1`, set by the entry stub and refreshed by
  `memory.grow`). After the assert, `a & 3` decides between the hot path —
  an aligned `i32` is a plain cell load/store, an aligned `i64` two cells,
  a byte/halfword one cell shifted by `8·(a & 3)` — and an out-of-line cold
  block placed after the function body for accesses crossing a wasm word
  (up to three cells). Aligned accesses never execute cold rows.
- **Structured control flow lowers to jumps.** `block`/`loop`/`if` become
  forward/backward `Jump`/`BranchIfZero`/`BranchIfNonZero`; a branch that
  carries values `Move`s them down to the target label's base height first.
  `br_table` is a compare-and-branch chain; `select` is a branch around a
  `Move`.
- **The ALU vocabulary is the table catalog.** `jolt_wasm_ir::AluOp` has one
  variant per lookup table (`Add/Sub/Mul` at 32/64, `And/Andn/Or/Xor`,
  `Eq/Ne/LtU/LtS/GeU/GeS/LeU`, `Srl/Sra/Rotr` over a shift bitmask,
  `NegateIf`, `MulUNoOverflow`, `Pow2`, `ShiftRightBitmask`,
  `SignExtendWord`, `LowerHalfWord`, `Clz/Ctz/Popcnt`) with an
  `OperandMode` (interleaved index, or the raw `left+right` / `left−right+2^64`
  / `left·right` index — Jolt's `Add/Sub/MulOperands` flags). `AluOp::evaluate`
  is the reference semantics; `jolt-wasm-tables::WasmTable::of(op)` is the
  table and `tests/catalog.rs` proves `materialize_entry(lookup_index) ==
  evaluate` on random domain inputs and on real traces. `Clz`, `Ctz`, and
  `Popcnt` are new tables in `jolt-lookup-tables` (with new `Prefixes` /
  `Suffixes` variants) — MSB-first prefix accumulation
  (`Popcnt' = P + popcnt(chunk)`, `Clz' = C + LeftIsZero·lz(chunk)`,
  `Ctz' = chunk≠0 ? tz(chunk) : C + |chunk|`), checked by the standard
  full-hypercube, random-MLE, and phase-by-phase prefix-suffix harnesses.
  They are deliberately *not* added to the RV64 `LookupTableKind` (the legacy
  prover mirrors it variant-for-variant); `WasmTable` is the WASM catalog the
  instruction lookup argument will key on, and a `const` assert keeps it
  under the kernels' 6-bit table-id cap.
- **Every WebAssembly operator is one catalog row or an expansion over
  catalog rows** (`lower.rs` module docs list them): `shl` = `Pow2`·`Mul`;
  `shr_u`/`shr_s`/`rotr` = `ShiftRightBitmask` then `Srl`/`Sra`/`Rotr`;
  `rotl` = `rotr` by `0 − count`; 32-bit variants mask the count and
  canonicalize with `LowerHalfWord`; `gt`/`le` swap operands of `lt`/`ge`,
  32-bit signed compares sign-extend first; `extend8_s`/`extend16_s` =
  `Mul` up then `Sra` by a constant bitmask; `div`/`rem` = `Advice` quotient
  pinned by `MulUNoOverflow`/`LeU`/`LtU` asserts (signed through `NegateIf`
  magnitudes, `div_s` asserting `!(MIN ∧ −1)`); sub-word/misaligned memory
  = word loads/stores of the containing doubleword and the next combined with
  bitmask shifts (`s = 8·(addr & 7)`), the bounds check being `LeU(end,
  size)`. No row ever needs a table outside the catalog.
- **Row classification** (`row.rs` in the backend): `RowModel::row_spec()`
  gives every `Ir` its `RowFlags` (`LEFT_IS_RS1/PC`, `RIGHT_IS_RS2/IMM`,
  `ADD/SUB/MUL_OPERANDS`, `WRITE_LOOKUP_TO_RD`, `LOAD`, `STORE`, `JUMP`,
  `BRANCH`, `ASSERT`, `HALT`, `MEMORY_GROW`, `TRAP`, `ADVICE`), operands,
  immediate, and `Lookup::{Table(AluOp), Advice(hint)}`. The interpreter
  executes rows through it; `check_record` restates the uniform R1CS in
  constraint form (RAM address = rs1 + imm under `LOAD|STORE`, load value =
  rd, store value = rs2, rd = lookup output under `WRITE_LOOKUP_TO_RD`,
  assert rows have output 1, branch/assert ops are boolean, next pc = output
  under `JUMP`, = imm under `BRANCH` with output 1, = pc under `HALT`, else
  pc + 1). Differences from RV64:
  absolute branch/jump targets, no link-register write on jumps, no
  expanded/compressed pc, register-or-immediate right operands on ALU and
  assert rows.
- **Bytecode preprocessing** (`jolt-wasm-program`): `WasmBytecode::preprocess`
  turns `IrProgram::code` into the committed table of packed 24-byte
  `BytecodeRow`s (`imm`, `RowFlags`, `rs1`/`rs2`/`rd` ids, `WasmTable` id) —
  the static half of every proof row — validating that pc 0 is the `Halt`
  trampoline, immediate jump/branch targets are in range, branch/assert ops
  are boolean, and every export's entry stub lies inside the program. The table is padded to a power of two (≥ 2) with the
  `Halt` row: a pc self-loop with no writes, the canonical no-op, so padding
  cycles are `Halt` at pc 0. IR pcs are dense, so a `Record` links to its row
  by `pc` alone (no expanded/unexpanded pc map). `BytecodeColumn` names the
  per-pc columns the bytecode read-RAF argument folds (`Pc`, `Imm`, each
  flag bit, `Rs1/Rs2/Rd`, one `TableFlag` per catalog table, `HasLookup`);
  `column_values` yields a column's hypercube evaluations and `encode` the
  canonical bytes a program commitment is taken over. `initial_memory_words`
  is the RAM argument's initial state: the non-zero aligned words of the data
  segments, globals, the memory-size word, and the run's inputs.
- **Compact trace row** (`jolt-wasm-program::WasmTraceRow`, 64 bytes,
  size-asserted): a `Record` materialized once at the trace boundary with
  the logical-column accessor API the witness/sumcheck code consumes
  (`rs1_value`, `rs2_value`, `rd_pre_value`, `rd_write_value`, `ram_address`,
  `ram_read_value`, `ram_write_value`, `pc`, `next_pc`, `imm`, register and
  table ids, flags). Four aliased value slots per row class — non-memory
  `rs1|rs2|rd_pre|rd_write`; load `rs1|addr|rd_pre|rd_write` (= ram read =
  ram write); store `rs1|rs2(=ram write)|ram_read|addr`; `memory.grow`
  `rs1|old size|rd_pre|new size` with the RAM address in `imm` and `rd_write`
  derived from the size words. `from_record` enforces the class contract
  (load value = rd write, store value = rs2, grow result derivation, operand
  ids match the row spec). The row is self-sufficient: `lookup_index()`
  (from its operand-mode flags) and `lookup_output()` (the `WasmTable` entry
  by id) need no `AluOp`, and `bytecode_row()` recovers the committed static
  half. `WasmTraceRow::default()` is the `Halt` padding row.
- **Uniform R1CS** (`jolt_r1cs::constraints::wasm`): the constraint-form
  `check_record` transcribed into 23 `guard · (left − right) = 0` rows and 2
  product rows over 33 variables per cycle (const, 16 inputs, the 16
  `RowFlags` bits in bit order). Beyond the RV64 set it constrains the
  instruction-input selection (`LEFT_IS_RS1/PC`, `RIGHT_IS_RS2/IMM`, zero
  otherwise) in-row and replaces the pc rules with four disjoint guards (`Jump`: next pc =
  lookup output; `ShouldBranch`: next pc = imm; `Halt`: next pc = pc;
  otherwise pc + 1). `jolt-wasm-program::r1cs::cycle_witness` fills the
  layout from a `WasmTraceRow`; the immediate column is signed
  (`BytecodeRow::imm_signed`: memory rows carry a byte offset, e.g. the
  shadow-stack reload at `SP − 8`), so the bytecode `Imm` column and the
  witness share one definition. Traces are single-segment (the entry stub
  folds `start` in), so the cross-cycle pc chain holds end to end.
- **Traps are typed and total.** Division by zero, signed overflow, OOB
  memory, `unreachable`, invalid jumps, and stack exhaustion surface as
  `ExecutionError::Trap { pc, trap }`; nothing panics on guest input.

### Non-Goals

- Floats (`f32`/`f64`), SIMD, imports/host calls, multi-memory, memory64,
  bulk memory, exceptions, GC types, and table mutation (`table.get/set/
  grow`, multiple tables, passive/declarative element segments). Each is a
  typed `DecodeError` today. One `funcref` table with active element
  segments and `call_indirect` is supported.
- Register spilling for frames larger than 120 slots.
- Proof-side integration: re-pointing `jolt-prover` (Spartan over
  `constraints::wasm`, bytecode read-RAF over `WasmBytecode`,
  instruction lookups over `WasmTable`, RAM over the WASM address space,
  witness generation over `WasmTraceRow`), and deleting the RISC-V crates.
  The modular prover's protocol vocabulary (`jolt-claims` ids such as
  `UnexpandedPC`, `NextIsVirtual`, `OpFlags(CircuitFlags)`,
  `LookupTableFlag` sized by `LookupTableKind::COUNT`) is RV64's; this is a
  vocabulary rewrite across claims/witness/prover/verifier/kernels, not a
  drop-in.
- Proving a trapping execution (a trap ends the trace without the
  termination word; the public output is then "no result").
- Tracing performance (chunked/parallel execution) — the interpreter is the
  reference oracle.

## Evaluation

### Acceptance Criteria

- [x] `WasmModule::decode` accepts the MVP integer core and rejects everything
      else with a typed error (`unsupported_operators_are_typed_errors`).
- [x] Recursive calls, loops, `br_table`, `select`, block results carried by
      branches, multi-value calls, globals + `start`, data segments, narrow
      loads/stores, `memory.grow`/`memory.size`, and `call_indirect` (table
      dispatch; null-slot, signature-mismatch, and out-of-bounds traps)
      execute to the spec-defined result
      (`crates/jolt-wasm-backend/tests/execute.rs`).
- [x] Every record's instruction is the one at its pc, `ZERO` is never
      written, and the pc chains within a segment (`check_records`).
- [x] Traps: divide-by-zero, signed overflow, OOB, `unreachable`, unbounded
      recursion.
- [x] Misaligned and doubleword-crossing loads/stores, and accesses to the
      last byte of memory, are correct; every record RAM access is an aligned
      word (`misaligned_and_word_crossing_accesses`, `check_records`).
- [x] Every record of every test trace satisfies `check_record`, and
      tampering rd, RAM address, or next pc is rejected
      (`tampered_records_fail_the_row_constraints`).
- [x] Signed/unsigned `div`/`rem` (negative operands, `MIN % -1 == 0`,
      `MIN / -1` traps), `shl`/`rotl` by counts ≥ width, all through the
      expansions (`division_shift_expansions`).
- [x] `Clz`/`Ctz`/`Popcnt` pass `mle_full_hypercube` (XLEN = 8),
      `mle_random`, and `prefix_suffix` (16- and 8-round phases, plus 2-round
      phases for `clz`/`ctz`); the rest of `jolt-lookup-tables` is unchanged
      (410/410).
- [x] Every `AluOp`'s table matches `AluOp::evaluate` on 2,000 random
      in-domain inputs plus corner cases, and on every lookup row of a real
      trace exercising every expansion (`jolt-wasm-tables/tests/catalog.rs`).
- [x] Every record of a real trace addresses the bytecode row of its own
      instruction; padding rows are the halt row; exports resolve to entry
      pcs; the initial image is word-for-word what the machine starts from
      (`jolt-wasm-program/tests/preprocess.rs`).
- [x] Every `WasmTraceRow` accessor reproduces its record (registers, RAM,
      pc chain), its static half equals the committed bytecode row, its
      self-computed lookup output equals the rd write / assert value, and
      contract violations are rejected (`jolt-wasm-program/tests/trace_row.rs`).
- [x] Every row of a real trace (all six row classes) and the padding row
      satisfy `wasm_trace_constraints`; tampering rd, RAM address, or next pc
      is rejected by the owning constraint (`jolt-wasm-program/tests/r1cs.rs`).
- [x] `cargo clippy -p jolt-wasm-ir -p jolt-wasm-frontend -p jolt-wasm-backend
      -p jolt-wasm-tables -p jolt-wasm-program -p jolt-r1cs --all-targets
      -- -D warnings` clean; the frontend and backend do not depend on each
      other.

### Testing Strategy

WAT fixtures compiled with the `wat` crate in integration tests; results are
independent ground truth from the WebAssembly spec. Neither the `host` nor
`zk` feature applies to this crate.

### Performance

None targeted in this slice; the interpreter exists for correctness. Record
size is 136 bytes (`Ir` is 16); a compact proof row will be a separate type
built at the trace boundary, as `JoltTraceRow` is for RISC-V today.

## Design

### Architecture

```
jolt-wasm-frontend   .wasm ──wasmparser+validator──▶ WasmModule { functions: [SourceOp{op, height}], … }
                                                          │  WasmModule::lower
jolt-wasm-ir                                              ▼
                                  IrProgram { code: [Ir], functions, memory, globals, data } + layout
                                                          │  Machine::invoke(export, args)
jolt-wasm-backend                                         ▼
                                  Execution { records: [Record], results, memory }
                                                          │  check_record (constraint-form spec)
                                                          ▼
                                                    Ok | RowViolation
jolt-wasm-tables      AluOp ──WasmTable::of──▶ prefix–suffix table; lookup_index(op, left, right)
                      (jolt-lookup-tables: + clz / ctz / popcnt tables, prefixes, suffixes)
jolt-wasm-program     IrProgram ──preprocess──▶ WasmBytecode { [BytecodeRow] padded, entries } + initial memory words
                      [Record] ──build_trace_rows──▶ [WasmTraceRow] (64 B, aliased slots, self-computed lookup output)
                      WasmTraceRow ──r1cs::cycle_witness──▶ [F; 34] ⊨ jolt_r1cs::constraints::wasm (24 eq + 2 product rows)
```

### Alternatives Considered

- **Memory-backed operand stack** (DelphinusLab zkWasm style): every
  push/pop is a RAM access, 2–3× more rows per WASM op, register file unused.
  Rejected: register allocation is free given static heights.
- **New stack-native proof row**: highest ceiling but blocks on rebuilding
  the memory-checking backend before the frontend can be tested. Rejected for
  the first slice; the register-row shape keeps the existing backend usable.
- **Register windows with a dynamic frame base**: breaks the static
  `rs1`/`rs2`/`rd` ids the bytecode lookup needs. Rejected.
- **Sign-extended i32 (RV64 `W` convention)** would let existing lookup
  tables be reused as-is. Rejected in favour of zero-extension because the
  lookup tables will be redefined for WASM anyway and zero-extension makes
  addressing and `extend_i32_u` trivial.

## Documentation

`CLAUDE.md` describes the WASM stack and its commands; the mdBook under
`book/` still documents the RISC-V era and is pending a rewrite.

## References

- WebAssembly Core Specification 2.0 (integer instructions, validation).
- `specs/source-jolt-instruction-split.md`, `specs/proof-trace-row-layout.md`
  (the RISC-V phase boundaries this frontend mirrors).

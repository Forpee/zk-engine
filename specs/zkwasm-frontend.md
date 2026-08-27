# Spec: zkWASM Frontend (IR and Execution Record)

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Claude, @Forpee                |
| Created     | 2026-08-27                     |
| Status      | implemented (frontend + table catalog) |
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
ALU op as a prefix–suffix-decomposable Jolt table). RISC-V crates remain in the tree only
until the prover is re-pointed at these records; they are not to be extended.

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
- **One address space.** Shadow stack (`0x1000_0000`), globals
  (`0x2000_0000`, 8 bytes each), system words (`0x3000_0000`; slot 0 is the
  linear-memory size in bytes), and linear memory (`0x8000_0000`) are disjoint
  regions of one 64-bit guest address space; a record's RAM address is always
  absolute. Shadow-stack exhaustion traps as `CallStackExhausted`.
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
  caller-agnostic.
- **Structured control flow lowers to jumps.** `block`/`loop`/`if` become
  forward/backward `Jump`/`BranchIfZero`/`BranchIfNonZero`; a branch that
  carries values `Move`s them down to the target label's base height first.
  `br_table` is a compare-and-branch chain; `select` is a branch around a
  `Move`.
- **Record stream segments.** `Machine::invoke` runs the `start` function
  (if any) and then the entry export as separate host-level calls. Each
  segment starts with host-initialized registers (parameters in frame slots,
  `SP` past a halt return address at `SHADOW_STACK_BASE`) and ends with the
  `Halt` record at `HALT_PC = 0`. The pc chains within a segment.
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
  pc + 1; `MEMORY_GROW` rewrites the size word). Differences from RV64:
  absolute branch/jump targets, no link-register write on jumps, no
  expanded/compressed pc, register-or-immediate right operands on ALU and
  assert rows.
- **Traps are typed and total.** Division by zero, signed overflow, OOB
  memory, `unreachable`, invalid jumps, and stack exhaustion surface as
  `ExecutionError::Trap { pc, trap }`; nothing panics on guest input.

### Non-Goals

- Floats (`f32`/`f64`), SIMD, tables/`call_indirect`, imports/host calls,
  multi-memory, memory64, bulk memory, exceptions, GC types. Each is a typed
  `DecodeError` today.
- Register spilling for frames larger than 120 slots.
- Proof-side integration: bytecode preprocessing over `IrProgram`, a
  compact proof row from `Record`, a WASM R1CS constraint set encoding
  `check_record`, re-pointing `jolt-prover`'s instruction lookup argument at
  `WasmTable`, and deleting the RISC-V crates.
- Tracing performance (chunked/parallel execution) — the interpreter is the
  reference oracle.

## Evaluation

### Acceptance Criteria

- [x] `WasmModule::decode` accepts the MVP integer core and rejects everything
      else with a typed error (`unsupported_operators_are_typed_errors`).
- [x] Recursive calls, loops, `br_table`, `select`, block results carried by
      branches, multi-value calls, globals + `start`, data segments, narrow
      loads/stores, `memory.grow`/`memory.size` execute to the spec-defined
      result (`crates/jolt-wasm-backend/tests/execute.rs`).
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
- [x] `cargo clippy -p jolt-wasm-ir -p jolt-wasm-frontend -p jolt-wasm-backend
      -p jolt-wasm-tables --all-targets -- -D warnings` clean; the frontend and
      backend do not depend on each other.

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

`CLAUDE.md` gains a `jolt-wasm` section; the book is updated when the prover
is re-pointed.

## References

- WebAssembly Core Specification 2.0 (integer instructions, validation).
- `specs/source-jolt-instruction-split.md`, `specs/proof-trace-row-layout.md`
  (the RISC-V phase boundaries this frontend mirrors).

//! End-to-end: WAT → decode → lower → execute, checking results, traps, and
//! the per-record witness shape.

#![expect(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use jolt_wasm_backend::{
    check_record, Execution, ExecutionError, Machine, RamAccess, RowViolation, Trap,
};
use jolt_wasm_frontend::{DecodeError, WasmModule};
use jolt_wasm_ir::layout::LINEAR_MEMORY_BASE;
use jolt_wasm_ir::{Ir, IrProgram, Reg};

fn program(wat: &str) -> IrProgram {
    let bytes = wat::parse_str(wat).expect("wat");
    let module = WasmModule::decode(&bytes).expect("decode");
    module.lower().expect("lower")
}

fn run(wat: &str, entry: &str, args: &[u64]) -> Execution {
    let program = program(wat);
    let execution = Machine::new(&program)
        .unwrap()
        .invoke(entry, args)
        .expect("execution");
    check_records(&program, &execution);
    execution
}

fn call(wat: &str, entry: &str, args: &[u64]) -> Result<Vec<u64>, ExecutionError> {
    let program = program(wat);
    Machine::new(&program)?
        .invoke(entry, args)
        .map(|e| e.results)
}

/// Every record is a valid proof-row witness: the pc chains within each
/// host-call segment, ZERO is never written, every recorded instruction is
/// the one at its pc, and every RAM access is an aligned 64-bit word.
fn check_records(program: &IrProgram, execution: &Execution) {
    let mut expected_pc = None;
    for record in &execution.records {
        assert_eq!(program.code[record.pc as usize], record.instruction);
        if let Some(pc) = expected_pc {
            assert_eq!(record.pc, pc, "pc chain broken");
        }
        // A `Halt` ends a host-call segment; the next segment starts wherever
        // the host placed the pc.
        expected_pc = (!matches!(record.instruction, Ir::Halt)).then_some(record.next_pc);
        if let Some(w) = record.rd {
            assert_ne!(w.register, Reg::ZERO);
        }
        let address = match record.ram {
            RamAccess::Read(r) => Some(r.address),
            RamAccess::Write(w) => Some(w.address),
            RamAccess::NoOp => None,
        };
        if let Some(address) = address {
            assert_eq!(address % 8, 0, "RAM access must be an aligned word");
        }
        check_record(record).unwrap();
    }
    assert!(matches!(
        execution.records.last().map(|r| r.instruction),
        Some(Ir::Halt)
    ));
}

const FIB: &str = r#"
(module
  (func $fib (export "fib") (param $n i64) (result i64)
    (if (result i64) (i64.lt_u (local.get $n) (i64.const 2))
      (then (local.get $n))
      (else
        (i64.add
          (call $fib (i64.sub (local.get $n) (i64.const 1)))
          (call $fib (i64.sub (local.get $n) (i64.const 2)))))))
)"#;

#[test]
fn recursive_fibonacci() {
    let execution = run(FIB, "fib", &[20]);
    assert_eq!(execution.results, vec![6765]);
    assert!(execution.records.len() > 6765);
}

#[test]
fn loop_sum_with_locals() {
    let wat = r#"
(module
  (func (export "sum") (param $n i32) (result i32)
    (local $i i32) (local $acc i32)
    (block $done
      (loop $top
        (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
        (local.set $acc (i32.add (local.get $acc) (local.get $i)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $top)))
    (local.get $acc)))"#;
    assert_eq!(run(wat, "sum", &[100]).results, vec![4950]);
}

#[test]
fn memory_data_segment_and_narrow_access() {
    let wat = r#"
(module
  (memory 1)
  (data (i32.const 16) "\01\02\03\04\ff\ff\ff\ff")
  (func (export "read") (result i64)
    (i64.add
      (i64.load32_u (i32.const 16))
      (i64.load8_s (i32.const 20))))
  (func (export "store") (result i32)
    (i32.store16 offset=100 (i32.const 0) (i32.const 0xbeef))
    (i32.store8 offset=102 (i32.const 0) (i32.const 0x7f))
    (i32.load offset=100 (i32.const 0))))"#;
    assert_eq!(
        run(wat, "read", &[]).results,
        vec![(0x0403_0201u64).wrapping_add(-1i64 as u64)]
    );
    let execution = run(wat, "store", &[]);
    assert_eq!(execution.results, vec![0x007f_beef]);
    let store = execution
        .records
        .iter()
        .find_map(|r| match r.ram {
            RamAccess::Write(w) => Some(w),
            _ => None,
        })
        .unwrap();
    assert_eq!(store.address, LINEAR_MEMORY_BASE + 96);
    assert_eq!(store.post_value, 0xbeef << 32);
    assert_eq!(store.pre_value, 0);
}

#[test]
fn globals_and_start() {
    let wat = r#"
(module
  (global $g (mut i64) (i64.const 40))
  (func $init (global.set $g (i64.add (global.get $g) (i64.const 2))))
  (start $init)
  (func (export "get") (result i64) (global.get $g)))"#;
    assert_eq!(run(wat, "get", &[]).results, vec![42]);
}

#[test]
fn br_table_and_select() {
    let wat = r#"
(module
  (func (export "sw") (param $x i32) (result i32)
    (block $c (block $b (block $a
      (br_table $a $b $c (local.get $x)))
      (return (i32.const 10)))
      (return (i32.const 20)))
    (i32.const 30))
  (func (export "sel") (param $c i32) (result i32)
    (select (i32.const 7) (i32.const 9) (local.get $c))))"#;
    for (x, want) in [(0, 10), (1, 20), (2, 30), (99, 30)] {
        assert_eq!(run(wat, "sw", &[x]).results, vec![want]);
    }
    assert_eq!(run(wat, "sel", &[1]).results, vec![7]);
    assert_eq!(run(wat, "sel", &[0]).results, vec![9]);
}

#[test]
fn block_results_and_branch_moves() {
    let wat = r#"
(module
  (func (export "f") (param $x i32) (result i32)
    (i32.const 100)
    (block $b (result i32)
      (i32.const 1)
      (i32.const 2)
      (br_if $b (i32.const 3) (local.get $x))
      (drop)
      (i32.add))
    (i32.add)))"#;
    assert_eq!(run(wat, "f", &[1]).results, vec![103]);
    assert_eq!(run(wat, "f", &[0]).results, vec![103]);
}

#[test]
fn i32_semantics_are_canonical_zero_extended() {
    let wat = r#"
(module
  (func (export "neg") (result i32) (i32.sub (i32.const 0) (i32.const 1)))
  (func (export "shr") (result i32) (i32.shr_s (i32.const -16) (i32.const 2)))
  (func (export "ext") (result i64) (i64.extend_i32_s (i32.const -1)))
  (func (export "wrap") (result i32) (i32.wrap_i64 (i64.const 0x1_0000_0005)))
  (func (export "rot") (result i32) (i32.rotl (i32.const 0x8000_0001) (i32.const 33)))
  (func (export "clz") (result i32) (i32.clz (i32.const 1))))"#;
    assert_eq!(run(wat, "neg", &[]).results, vec![0xFFFF_FFFF]);
    assert_eq!(run(wat, "shr", &[]).results, vec![0xFFFF_FFFC]);
    assert_eq!(run(wat, "ext", &[]).results, vec![u64::MAX]);
    assert_eq!(run(wat, "wrap", &[]).results, vec![5]);
    assert_eq!(run(wat, "rot", &[]).results, vec![3]);
    assert_eq!(run(wat, "clz", &[]).results, vec![31]);
}

const DIVMOD_SHIFTS: &str = r#"
(module
  (func (export "divs") (param i32 i32) (result i32) (i32.div_s (local.get 0) (local.get 1)))
  (func (export "divu") (param i32 i32) (result i32) (i32.div_u (local.get 0) (local.get 1)))
  (func (export "rems") (param i32 i32) (result i32) (i32.rem_s (local.get 0) (local.get 1)))
  (func (export "remu") (param i32 i32) (result i32) (i32.rem_u (local.get 0) (local.get 1)))
  (func (export "divs64") (param i64 i64) (result i64) (i64.div_s (local.get 0) (local.get 1)))
  (func (export "rems64") (param i64 i64) (result i64) (i64.rem_s (local.get 0) (local.get 1)))
  (func (export "remu64") (param i64 i64) (result i64) (i64.rem_u (local.get 0) (local.get 1)))
  (func (export "shl") (param i32 i32) (result i32) (i32.shl (local.get 0) (local.get 1)))
  (func (export "shl64") (param i64 i64) (result i64) (i64.shl (local.get 0) (local.get 1)))
  (func (export "rotl") (param i32 i32) (result i32) (i32.rotl (local.get 0) (local.get 1)))
  (func (export "rotl64") (param i64 i64) (result i64) (i64.rotl (local.get 0) (local.get 1)))
  (func (export "shrs") (param i32 i32) (result i32) (i32.shr_s (local.get 0) (local.get 1))))"#;

#[test]
fn division_shift_expansions() {
    let w = DIVMOD_SHIFTS;
    let neg = |x: i32| u64::from(x as u32);
    assert_eq!(run(w, "divs", &[neg(-7), 2]).results, vec![neg(-3)]);
    assert_eq!(run(w, "divs", &[7, neg(-2)]).results, vec![neg(-3)]);
    assert_eq!(run(w, "divs", &[neg(-7), neg(-2)]).results, vec![3]);
    assert_eq!(run(w, "rems", &[neg(-7), 2]).results, vec![neg(-1)]);
    assert_eq!(run(w, "rems", &[7, neg(-2)]).results, vec![1]);
    // i32.rem_s of MIN by -1 is 0 (no trap); div_s traps.
    assert_eq!(run(w, "rems", &[0x8000_0000, neg(-1)]).results, vec![0]);
    assert_eq!(
        run(w, "divu", &[0xFFFF_FFFF, 16]).results,
        vec![0x0FFF_FFFF]
    );
    assert_eq!(run(w, "remu", &[0xFFFF_FFFF, 16]).results, vec![15]);
    assert_eq!(run(w, "divu", &[5, 7]).results, vec![0]);
    assert_eq!(run(w, "remu", &[5, 7]).results, vec![5]);
    assert_eq!(
        run(w, "divs64", &[(-9i64) as u64, 4]).results,
        vec![(-2i64) as u64]
    );
    assert_eq!(
        run(w, "rems64", &[(-9i64) as u64, 4]).results,
        vec![(-1i64) as u64]
    );
    assert_eq!(run(w, "rems64", &[1 << 63, u64::MAX]).results, vec![0]);
    assert_eq!(run(w, "remu64", &[u64::MAX, 10]).results, vec![5]);
    assert_eq!(run(w, "shl", &[1, 31]).results, vec![0x8000_0000]);
    assert_eq!(run(w, "shl", &[1, 32]).results, vec![1]);
    assert_eq!(run(w, "shl", &[3, 33]).results, vec![6]);
    assert_eq!(run(w, "shl64", &[1, 63]).results, vec![1 << 63]);
    assert_eq!(run(w, "shl64", &[1, 64]).results, vec![1]);
    assert_eq!(run(w, "rotl", &[0x8000_0001, 1]).results, vec![3]);
    assert_eq!(run(w, "rotl", &[0x8000_0001, 0]).results, vec![0x8000_0001]);
    assert_eq!(run(w, "rotl", &[1, 32]).results, vec![1]);
    assert_eq!(run(w, "rotl64", &[1 << 63, 1]).results, vec![1]);
    assert_eq!(run(w, "rotl64", &[1, 0]).results, vec![1]);
    assert_eq!(run(w, "shrs", &[neg(-8), 1]).results, vec![neg(-4)]);
    let trap = |r: Result<Vec<u64>, ExecutionError>| match r {
        Err(ExecutionError::Trap { trap, .. }) => trap,
        other => panic!("expected trap, got {other:?}"),
    };
    assert_eq!(trap(call(w, "divs", &[1, 0])), Trap::DivideByZero);
    assert_eq!(trap(call(w, "rems", &[1, 0])), Trap::DivideByZero);
    assert_eq!(trap(call(w, "remu", &[1, 0])), Trap::DivideByZero);
    assert_eq!(
        trap(call(w, "divs", &[0x8000_0000, neg(-1)])),
        Trap::IntegerOverflow
    );
    assert_eq!(
        trap(call(w, "divs64", &[1 << 63, u64::MAX])),
        Trap::IntegerOverflow
    );
}

#[test]
fn traps() {
    let wat = r#"
(module
  (memory 1)
  (func (export "div") (param i32 i32) (result i32) (i32.div_s (local.get 0) (local.get 1)))
  (func (export "oob") (result i32) (i32.load (i32.const 65534)))
  (func (export "unr") unreachable)
  (func $rec (export "rec") (call $rec)))"#;
    let trap = |r: Result<Vec<u64>, ExecutionError>| match r {
        Err(ExecutionError::Trap { trap, .. }) => trap,
        other => panic!("expected trap, got {other:?}"),
    };
    assert_eq!(trap(call(wat, "div", &[1, 0])), Trap::DivideByZero);
    assert_eq!(
        trap(call(wat, "div", &[0x8000_0000, 0xFFFF_FFFF])),
        Trap::IntegerOverflow
    );
    assert_eq!(call(wat, "div", &[7, 2]).unwrap(), vec![3]);
    assert!(matches!(
        trap(call(wat, "oob", &[])),
        Trap::OutOfBoundsMemory { .. }
    ));
    assert_eq!(trap(call(wat, "unr", &[])), Trap::Unreachable);
    assert_eq!(trap(call(wat, "rec", &[])), Trap::CallStackExhausted);
}

#[test]
fn misaligned_and_word_crossing_accesses() {
    let wat = r#"
(module
  (memory 1)
  (data (i32.const 0) "\00\01\02\03\04\05\06\07\08\09\0a\0b\0c\0d\0e\0f")
  (func (export "ld64") (param $a i32) (result i64) (i64.load (local.get $a)))
  (func (export "ld32s") (param $a i32) (result i32) (i32.load16_s (local.get $a)))
  (func (export "st64") (param $a i32) (param $v i64) (result i64)
    (i64.store (local.get $a) (local.get $v))
    (i64.load (i32.const 0)))
  (func (export "st16") (param $a i32) (result i64)
    (i32.store16 (local.get $a) (i32.const 0xffff))
    (i64.load (i32.const 8)))
  (func (export "last8") (result i32) (i32.load8_u (i32.const 65535)))
  (func (export "last16") (result i32) (i32.load16_u (i32.const 65535)))
  (func (export "st_last") (result i32)
    (i32.store8 (i32.const 65535) (i32.const 0xab))
    (i32.load8_u (i32.const 65535))))"#;
    assert_eq!(run(wat, "ld64", &[0]).results, vec![0x0706_0504_0302_0100]);
    assert_eq!(run(wat, "ld64", &[5]).results, vec![0x0c0b_0a09_0807_0605]);
    assert_eq!(run(wat, "ld32s", &[7]).results, vec![0x0807]);
    // Crossing store: bytes 5..13 overwritten; the low word reflects 5..8.
    assert_eq!(
        run(wat, "st64", &[5, 0xffff_ffff_ffff_ffff]).results,
        vec![0xffff_ff04_0302_0100]
    );
    // 16-bit store at 7 crosses into the second word's low byte.
    assert_eq!(run(wat, "st16", &[7]).results, vec![0x0f0e_0d0c_0b0a_09ff]);
    assert_eq!(run(wat, "last8", &[]).results, vec![0]);
    assert_eq!(run(wat, "st_last", &[]).results, vec![0xab]);
    assert!(matches!(
        call(wat, "last16", &[]),
        Err(ExecutionError::Trap {
            trap: Trap::OutOfBoundsMemory { .. },
            ..
        })
    ));
}

#[test]
fn memory_grow_and_size() {
    let wat = r#"
(module
  (memory 1 3)
  (func (export "grow") (result i32 i32 i32)
    (memory.grow (i32.const 1))
    (memory.size)
    (memory.grow (i32.const 5))))"#;
    assert_eq!(run(wat, "grow", &[]).results, vec![1, 2, 0xFFFF_FFFF]);
}

#[test]
fn multi_value_calls_and_nested_frames() {
    let wat = r#"
(module
  (func $divmod (param i64 i64) (result i64 i64)
    (i64.div_u (local.get 0) (local.get 1))
    (i64.rem_u (local.get 0) (local.get 1)))
  (func (export "f") (param $a i64) (param $b i64) (result i64)
    (local $keep i64)
    (local.set $keep (i64.const 1000))
    (call $divmod (local.get $a) (local.get $b))
    (i64.mul)
    (local.get $keep)
    (i64.add)))"#;
    assert_eq!(run(wat, "f", &[47, 5]).results, vec![9 * 2 + 1000]);
}

#[test]
fn tampered_records_fail_the_row_constraints() {
    use RamAccess as Ram;
    let execution = run(FIB, "fib", &[5]);
    let mut seen = 0;
    for record in &execution.records {
        if let Some(mut w) = record.rd {
            let mut bad = *record;
            w.post_value = w.post_value.wrapping_add(1);
            bad.rd = Some(w);
            assert!(matches!(
                check_record(&bad),
                Err(RowViolation::RdWrite { .. } | RowViolation::Ram { .. })
            ));
            seen += 1;
        }
        if let Ram::Read(mut r) = record.ram {
            let mut bad = *record;
            r.address += 8;
            bad.ram = Ram::Read(r);
            assert!(matches!(check_record(&bad), Err(RowViolation::Ram { .. })));
        }
        let mut bad = *record;
        bad.next_pc = bad.next_pc.wrapping_add(1);
        assert!(matches!(
            check_record(&bad),
            Err(RowViolation::NextPc { .. })
        ));
    }
    assert!(seen > 0);
}

#[test]
fn unsupported_operators_are_typed_errors() {
    let bytes =
        wat::parse_str(r#"(module (func (export "f") (result f32) (f32.const 1.0)))"#).unwrap();
    assert!(matches!(
        WasmModule::decode(&bytes),
        Err(DecodeError::UnsupportedValType(_))
    ));
    let bytes = wat::parse_str(r#"(module (import "env" "x" (func)))"#).unwrap();
    assert!(matches!(
        WasmModule::decode(&bytes),
        Err(DecodeError::Import { .. })
    ));
}

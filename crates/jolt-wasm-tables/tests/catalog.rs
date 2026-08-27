//! The catalog is faithful: for every `AluOp`, the mapped table's
//! `materialize_entry` at the row's lookup index equals `AluOp::evaluate` —
//! on random inputs within each op's domain, and on every row of real traces.

#![expect(clippy::unwrap_used, clippy::expect_used)]

use jolt_wasm_backend::{Lookup, Machine, RowModel};
use jolt_wasm_frontend::WasmModule;
use jolt_wasm_ir::{shift_right_bitmask, AluOp, Width};
use jolt_wasm_tables::{table_output, WasmTable};
use rand::{rngs::StdRng, Rng, SeedableRng};

const ALL_OPS: &[AluOp] = &[
    AluOp::Add(Width::W64),
    AluOp::Add(Width::W32),
    AluOp::Sub(Width::W64),
    AluOp::Sub(Width::W32),
    AluOp::Mul(Width::W64),
    AluOp::Mul(Width::W32),
    AluOp::And,
    AluOp::Andn,
    AluOp::Or,
    AluOp::Xor,
    AluOp::Eq,
    AluOp::Ne,
    AluOp::LtU,
    AluOp::LtS,
    AluOp::GeU,
    AluOp::GeS,
    AluOp::LeU,
    AluOp::Srl,
    AluOp::Sra,
    AluOp::Rotr,
    AluOp::NegateIf,
    AluOp::MulUNoOverflow,
    AluOp::Pow2,
    AluOp::ShiftRightBitmask,
    AluOp::SignExtendWord,
    AluOp::LowerHalfWord,
    AluOp::Clz,
    AluOp::Ctz,
    AluOp::Popcnt,
];

/// Random inputs inside the op's valid domain: shift/rotate tables take a
/// right-shift bitmask, the 32-bit ops take canonical 32-bit values.
fn sample(rng: &mut StdRng, op: AluOp) -> (u64, u64) {
    let x: u64 = rng.gen();
    let y: u64 = rng.gen();
    match op {
        AluOp::Srl | AluOp::Sra | AluOp::Rotr => (x, shift_right_bitmask(rng.gen_range(0..64))),
        AluOp::Add(Width::W32) | AluOp::Sub(Width::W32) | AluOp::Mul(Width::W32) => {
            (u64::from(x as u32), u64::from(y as u32))
        }
        _ => (x, y),
    }
}

#[test]
fn every_op_matches_its_table_on_random_inputs() {
    let mut rng = StdRng::seed_from_u64(7);
    for &op in ALL_OPS {
        for _ in 0..2000 {
            let (x, y) = sample(&mut rng, op);
            assert_eq!(
                table_output(op, x, y),
                op.evaluate(x, y),
                "{op:?}({x:#x}, {y:#x})"
            );
        }
        for &(x, y) in &[
            (0, 0),
            (u64::MAX, u64::MAX),
            (0, u64::MAX),
            (1 << 63, u64::MAX),
        ] {
            let (x, y) = match op {
                AluOp::Srl | AluOp::Sra | AluOp::Rotr => (x, shift_right_bitmask(y & 63)),
                _ => (x, y),
            };
            assert_eq!(
                table_output(op, x, y),
                op.evaluate(x, y),
                "{op:?}({x:#x}, {y:#x})"
            );
        }
    }
}

#[test]
fn every_op_has_a_decomposable_table_with_a_valid_id() {
    for &op in ALL_OPS {
        let table = WasmTable::of(op);
        assert!(table.index() < WasmTable::COUNT);
        assert!(!table.prefixes().is_empty() && !table.suffixes().is_empty());
    }
}

const PROGRAM: &str = r#"
(module
  (memory 1)
  (data (i32.const 0) "\01\02\03\04\05\06\07\08\09\0a\0b\0c\0d\0e\0f\10")
  (func $fib (param $n i64) (result i64)
    (if (result i64) (i64.lt_u (local.get $n) (i64.const 2))
      (then (local.get $n))
      (else (i64.add (call $fib (i64.sub (local.get $n) (i64.const 1)))
                     (call $fib (i64.sub (local.get $n) (i64.const 2)))))))
  (func (export "mix") (param $a i32) (param $b i32) (result i64)
    (local $acc i64)
    (local.set $acc (call $fib (i64.const 12)))
    (local.set $acc (i64.add (local.get $acc) (i64.extend_i32_s (i32.div_s (local.get $a) (local.get $b)))))
    (local.set $acc (i64.add (local.get $acc) (i64.extend_i32_u (i32.rem_u (local.get $a) (local.get $b)))))
    (local.set $acc (i64.add (local.get $acc) (i64.extend_i32_u (i32.shl (local.get $a) (local.get $b)))))
    (local.set $acc (i64.add (local.get $acc) (i64.extend_i32_u (i32.shr_s (local.get $a) (local.get $b)))))
    (local.set $acc (i64.add (local.get $acc) (i64.extend_i32_u (i32.rotl (local.get $a) (local.get $b)))))
    (local.set $acc (i64.add (local.get $acc) (i64.extend_i32_u (i32.clz (local.get $a)))))
    (local.set $acc (i64.add (local.get $acc) (i64.extend_i32_u (i32.ctz (local.get $a)))))
    (local.set $acc (i64.add (local.get $acc) (i64.extend_i32_u (i32.popcnt (local.get $a)))))
    (local.set $acc (i64.add (local.get $acc) (i64.extend_i32_u (i32.lt_s (local.get $a) (local.get $b)))))
    (local.set $acc (i64.add (local.get $acc) (i64.extend_i32_u (i32.extend8_s (local.get $a)))))
    (local.set $acc (i64.add (local.get $acc) (i64.load (i32.const 3))))
    (local.set $acc (i64.add (local.get $acc) (i64.load16_s (i32.const 7))))
    (i64.store (i32.const 5) (local.get $acc))
    (i32.store8 (i32.const 15) (i32.const 0x7f))
    (i64.rotr (local.get $acc) (i64.const 13))))"#;

#[test]
fn every_row_of_a_real_trace_matches_its_table() {
    let bytes = wat::parse_str(PROGRAM).unwrap();
    let program = WasmModule::decode(&bytes).unwrap().lower().unwrap();
    let run = Machine::new(&program)
        .unwrap()
        .invoke("mix", &[0xdead_beef, 7])
        .expect("execution");
    let mut checked = 0;
    for record in &run.records {
        let spec = record.instruction.row_spec();
        let Some(Lookup::Table(op)) = spec.lookup else {
            continue;
        };
        let left = spec.left_input(record.rs1.map_or(0, |r| r.value));
        let right = spec.right_input(record.rs2.map_or(0, |r| r.value));
        assert_eq!(
            table_output(op, left, right),
            op.evaluate(left, right),
            "pc {}: {:?}",
            record.pc,
            record.instruction
        );
        checked += 1;
    }
    assert!(checked > 1000);
}

//! Every row of a real trace satisfies the WASM uniform R1CS, and tampering
//! any constrained column is caught by the constraint that owns it.

#![expect(clippy::unwrap_used, clippy::panic)]

use jolt_field::Fr;
use jolt_r1cs::constraints::wasm::{
    wasm_trace_constraints, V_NEXT_PC, V_RAM_ADDRESS, V_RD_WRITE_VALUE,
};
use jolt_wasm_backend::{Machine, RowFlag};
use jolt_wasm_frontend::WasmModule;
use jolt_wasm_program::r1cs::cycle_witness;
use jolt_wasm_program::{build_trace_rows, WasmTraceRow};

const PROGRAM: &str = r#"
(module
  (memory 1 4)
  (global $g (mut i64) (i64.const 7))
  (data (i32.const 0) "\01\02\03\04\05\06\07\08\09\0a\0b\0c\0d\0e\0f\10")
  (func $fib (param $n i64) (result i64)
    (if (result i64) (i64.lt_u (local.get $n) (i64.const 2))
      (then (local.get $n))
      (else (i64.add (call $fib (i64.sub (local.get $n) (i64.const 1)))
                     (call $fib (i64.sub (local.get $n) (i64.const 2)))))))
  (func (export "mix") (param $a i32) (param $b i32) (result i64)
    (local $acc i64)
    (local.set $acc (call $fib (i64.const 9)))
    (local.set $acc (i64.add (local.get $acc) (i64.extend_i32_s (i32.div_s (local.get $a) (local.get $b)))))
    (local.set $acc (i64.add (local.get $acc) (i64.extend_i32_u (i32.rotl (local.get $a) (local.get $b)))))
    (local.set $acc (i64.add (local.get $acc) (i64.extend_i32_u (i32.clz (local.get $a)))))
    (local.set $acc (i64.add (local.get $acc) (i64.load (i32.const 3))))
    (block $b (br_if $b (i32.eqz (local.get $b))) (i32.store16 (i32.const 7) (i32.const 0xbeef)))
    (global.set $g (i64.add (global.get $g) (local.get $acc)))
    (drop (memory.grow (i32.const 1)))
    (i64.add (global.get $g) (i64.extend_i32_u (memory.size)))))"#;

fn rows() -> Vec<WasmTraceRow> {
    let bytes = wat::parse_str(PROGRAM).unwrap();
    let program = WasmModule::decode(&bytes).unwrap().lower().unwrap();
    let run = Machine::new(&program)
        .unwrap()
        .invoke("mix", &[0xdead_beef, 7])
        .unwrap();
    build_trace_rows(&run.records).unwrap()
}

#[test]
fn every_row_satisfies_the_wasm_r1cs() {
    let matrices = wasm_trace_constraints::<Fr>();
    let rows = rows();
    assert!(rows.len() > 1000);
    let mut classes = [0usize; 6];
    for row in &rows {
        matrices
            .check_witness(&cycle_witness::<Fr>(row))
            .unwrap_or_else(|i| panic!("pc {}: constraint {i} violated", row.pc()));
        let f = row.flags();
        let class = if f.has(RowFlag::Load) {
            0
        } else if f.has(RowFlag::Store) {
            1
        } else if f.has(RowFlag::Branch) {
            2
        } else if f.has(RowFlag::Jump) {
            3
        } else if f.intersects(RowFlag::Advice | RowFlag::Assert) {
            4
        } else {
            5
        };
        classes[class] += 1;
    }
    assert!(classes.iter().all(|c| *c > 0), "{classes:?}");
    // The padding row satisfies the set too.
    matrices
        .check_witness(&cycle_witness::<Fr>(&WasmTraceRow::default()))
        .unwrap();
}

#[test]
fn tampered_columns_are_rejected() {
    let matrices = wasm_trace_constraints::<Fr>();
    let rows = rows();
    let alu = rows
        .iter()
        .find(|r| r.flags().has(RowFlag::WriteLookupToRd))
        .unwrap();
    let mut w = cycle_witness::<Fr>(alu);
    w[V_RD_WRITE_VALUE] += Fr::from(1u64);
    assert!(matches!(matrices.check_witness(&w), Err(17)));

    let load = rows.iter().find(|r| r.flags().has(RowFlag::Load)).unwrap();
    let mut w = cycle_witness::<Fr>(load);
    w[V_RAM_ADDRESS] += Fr::from(8u64);
    assert!(matches!(matrices.check_witness(&w), Err(5)));

    let branch = rows
        .iter()
        .find(|r| r.flags().has(RowFlag::Branch) && r.lookup_output() == 1)
        .unwrap();
    let mut w = cycle_witness::<Fr>(branch);
    w[V_NEXT_PC] += Fr::from(1u64);
    assert!(matches!(matrices.check_witness(&w), Err(19)));

    let plain = rows
        .iter()
        .find(|r| {
            !r.flags()
                .intersects(RowFlag::Branch | RowFlag::Jump | RowFlag::Halt)
        })
        .unwrap();
    let mut w = cycle_witness::<Fr>(plain);
    w[V_NEXT_PC] += Fr::from(1u64);
    assert!(matches!(matrices.check_witness(&w), Err(21)));
}

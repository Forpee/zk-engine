//! The compact row is a faithful, self-sufficient view of a record: every
//! logical accessor reproduces the record's values, the static half equals
//! the committed bytecode row, and the lookup output recomputed from the
//! row alone matches the rd write.

#![expect(clippy::unwrap_used)]

use jolt_wasm_backend::{Machine, RamAccess, Record, RowFlag};
use jolt_wasm_frontend::WasmModule;
use jolt_wasm_ir::{Ir, IrProgram, Reg};
use jolt_wasm_program::{
    build_trace_rows, TraceRowError, WasmBytecode, WasmProgramPreprocessing, WasmTraceRow,
};

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
    (local.set $acc (i64.add (local.get $acc) (i64.extend_i32_u (i32.popcnt (local.get $a)))))
    (local.set $acc (i64.add (local.get $acc) (i64.load (i32.const 3))))
    (i32.store16 (i32.const 7) (i32.const 0xbeef))
    (global.set $g (i64.add (global.get $g) (local.get $acc)))
    (drop (memory.grow (i32.const 1)))
    (drop (memory.grow (i32.const 9)))
    (i64.add (global.get $g) (i64.extend_i32_u (memory.size)))))"#;

fn run() -> (IrProgram, Vec<Record>) {
    let bytes = wat::parse_str(PROGRAM).unwrap();
    let program = WasmModule::decode(&bytes).unwrap().lower().unwrap();
    let execution = Machine::new(&program)
        .unwrap()
        .invoke("mix", &[0xdead_beef, 7])
        .unwrap();
    (program, execution.records)
}

#[test]
fn rows_reproduce_records_and_bytecode() {
    let (program, records) = run();
    let bytecode = WasmBytecode::preprocess(&program).unwrap();
    let rows = build_trace_rows(&records).unwrap();
    assert_eq!(rows.len(), records.len());
    let (mut loads, mut stores) = (0, 0);
    for (row, record) in rows.iter().zip(&records) {
        assert_eq!(row.pc(), record.pc);
        assert_eq!(row.next_pc(), record.next_pc);
        assert_eq!(row.bytecode_row(), bytecode.row(record.pc));
        assert_eq!(row.rs1_value(), record.rs1.map_or(0, |r| r.value));
        assert_eq!(row.rs2_value(), record.rs2.map_or(0, |r| r.value));
        assert_eq!(row.rd_pre_value(), record.rd.map_or(0, |w| w.pre_value));
        assert_eq!(row.rd_write_value(), record.rd.map_or(0, |w| w.post_value));
        assert_eq!(row.rs1_index(), record.rs1.map(|r| r.register));
        assert_eq!(row.rs2_index(), record.rs2.map(|r| r.register));
        assert_eq!(row.rd_index(), record.rd.map(|w| w.register));
        match record.ram {
            RamAccess::Read(read) => {
                loads += 1;
                assert!(row.is_load());
                assert_eq!(row.ram_address(), read.address);
                assert_eq!(row.ram_read_value(), read.value);
                assert_eq!(row.ram_write_value(), read.value);
            }
            RamAccess::Write(write) => {
                stores += 1;
                assert!(row.is_store());
                assert_eq!(row.ram_address(), write.address);
                assert_eq!(row.ram_read_value(), write.pre_value);
                assert_eq!(row.ram_write_value(), write.post_value);
            }
            RamAccess::NoOp => {
                assert_eq!(row.ram_address(), 0);
                assert_eq!(row.ram_read_value(), 0);
                assert_eq!(row.ram_write_value(), 0);
            }
        }
        if row.flags().has(RowFlag::WriteLookupToRd) {
            assert_eq!(row.lookup_output(), row.rd_write_value(), "pc {}", row.pc());
        }
        if row.flags().has(RowFlag::Assert) {
            assert_eq!(row.lookup_output(), 1);
        }
    }
    assert!(loads > 0 && stores > 0);
    assert!(rows.last().unwrap().is_noop());
    assert_eq!(
        rows[0].pc(),
        bytecode.entry("mix").unwrap(),
        "the trace starts at the entry stub"
    );
}

#[test]
fn no_op_row_is_the_halt_trampoline() {
    let row = WasmTraceRow::default();
    assert!(row.is_noop());
    assert_eq!(row.pc(), 0);
    assert_eq!(row.next_pc(), 0);
    assert_eq!(row.rd_index(), None);
    assert_eq!(row.ram_address(), 0);
    assert_eq!(row.lookup_output(), 0);
    let bytecode = WasmBytecode::preprocess(&run().0).unwrap();
    assert_eq!(row.bytecode_row(), bytecode.row(0));
    let pre = WasmProgramPreprocessing::new(&run().0, 1 << 10).unwrap();
    assert_eq!(pre.bytecode.row(0), row.bytecode_row());
}

#[test]
fn contract_violations_are_rejected() {
    let (_, records) = run();
    let load = records
        .iter()
        .find(|r| matches!(r.ram, RamAccess::Read(_)))
        .unwrap();
    let mut bad = *load;
    if let Some(w) = bad.rd.as_mut() {
        w.post_value ^= 1;
    }
    assert!(matches!(
        WasmTraceRow::from_record(&bad),
        Err(TraceRowError::MemoryContract { .. })
    ));
    let alu = records
        .iter()
        .find(|r| matches!(r.instruction, Ir::Alu { .. }) && r.rs2.is_some())
        .unwrap();
    let mut bad = *alu;
    if let Some(r) = bad.rs2.as_mut() {
        r.register = Reg::T4;
    }
    assert!(matches!(
        WasmTraceRow::from_record(&bad),
        Err(TraceRowError::RegisterOperands { .. })
    ));
}

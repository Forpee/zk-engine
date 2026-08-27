//! The bytecode table and memory images agree with the trace: every record's
//! pc addresses the row of its own instruction, exports resolve to their
//! entry stubs, and the public images are what the machine starts from and
//! ends with.

#![expect(clippy::unwrap_used)]

use jolt_wasm_backend::{Lookup, Machine, RowFlag, RowModel};
use jolt_wasm_frontend::WasmModule;
use jolt_wasm_ir::layout::{
    input_address, linear_address, output_address, GLOBALS_BASE, MEMORY_SIZE_ADDR, TERMINATION_ADDR,
};
use jolt_wasm_ir::{Ir, IrProgram};
use jolt_wasm_program::{
    BytecodeColumn, BytecodeRow, MemoryWord, PublicIo, WasmBytecode, WasmProgramPreprocessing,
};

const PROGRAM: &str = r#"
(module
  (memory 1)
  (global $g (mut i64) (i64.const 0x1122334455667788))
  (data (i32.const 3) "\aa\bb\cc\dd\ee")
  (func $fib (export "fib") (param $n i64) (result i64)
    (if (result i64) (i64.lt_u (local.get $n) (i64.const 2))
      (then (local.get $n))
      (else (i64.add (call $fib (i64.sub (local.get $n) (i64.const 1)))
                     (call $fib (i64.sub (local.get $n) (i64.const 2)))))))
  (func (export "touch") (param $a i32) (result i64)
    (global.set $g (i64.add (global.get $g) (i64.load8_u (local.get $a))))
    (global.get $g)))"#;

fn program() -> IrProgram {
    let bytes = wat::parse_str(PROGRAM).unwrap();
    WasmModule::decode(&bytes).unwrap().lower().unwrap()
}

#[test]
fn bytecode_rows_match_every_record() {
    let program = program();
    let pre = WasmProgramPreprocessing::new(&program, 1 << 16).unwrap();
    let bytecode = &pre.bytecode;
    assert!(bytecode.code_size().is_power_of_two());
    assert!(bytecode.code_size() >= bytecode.len());
    assert_eq!(bytecode.row(0), BytecodeRow::of(Ir::Halt));
    assert!(bytecode.row(0).flags.has(RowFlag::Halt));
    for pc in bytecode.len()..bytecode.code_size() {
        assert_eq!(
            bytecode.row(pc as u32),
            bytecode.row(0),
            "padding rows are the halt row"
        );
    }
    assert_eq!(
        bytecode.row(bytecode.code_size() as u32 + 5),
        bytecode.row(0)
    );
    assert_eq!(bytecode.entry("fib"), Some(program.entries["fib"]));
    assert_eq!(bytecode.entry("touch"), Some(program.entries["touch"]));
    assert_ne!(
        bytecode.entry("fib"),
        Some(program.functions[0].entry),
        "the entry is the stub, not the body"
    );

    let run = Machine::new(&program)
        .unwrap()
        .invoke("fib", &[10])
        .unwrap();
    assert_eq!(run.results, vec![55]);
    assert_eq!(run.records[0].pc, bytecode.entry("fib").unwrap());
    for record in &run.records {
        assert!((record.pc as usize) < bytecode.len());
        assert_eq!(bytecode.row(record.pc), BytecodeRow::of(record.instruction));
        let spec = record.instruction.row_spec();
        assert_eq!(
            bytecode.column(record.pc, BytecodeColumn::Imm),
            bytecode.row(record.pc).imm_signed()
        );
        assert_eq!(bytecode.row(record.pc).imm, spec.imm);
        assert_eq!(
            bytecode.column(record.pc, BytecodeColumn::Pc),
            i128::from(record.pc)
        );
        let raf = bytecode.column(record.pc, BytecodeColumn::RafFlag);
        assert_eq!(raf == 1, spec.raf_flag());
        if let Some(Lookup::Table(op)) = spec.lookup {
            assert_eq!(bytecode.column(record.pc, BytecodeColumn::TableFlag(op)), 1);
            assert_eq!(
                jolt_wasm_tables::bytecode_table_id(&bytecode.row(record.pc)),
                Some(jolt_wasm_tables::WasmTable::of(op).index())
            );
        }
    }
    let encoded = bytecode.encode();
    assert_eq!(encoded.len(), bytecode.code_size() * 16);
    assert_eq!(
        bytecode
            .column_values(BytecodeColumn::Flag(RowFlag::Halt))
            .len(),
        bytecode.code_size()
    );
}

#[test]
fn public_images_bracket_the_execution() {
    let program = program();
    let pre = WasmProgramPreprocessing::new(&program, 1 << 16).unwrap();
    // `touch` reads byte 3 (0xaa) and adds it to the global; the program
    // image holds the data segment, the global, and the size word.
    let run = Machine::new(&program)
        .unwrap()
        .invoke("touch", &[3])
        .unwrap();
    assert_eq!(run.results, vec![0x1122_3344_5566_7788 + 0xaa]);
    assert!(run.terminated);
    assert_eq!(
        pre.program_memory,
        vec![
            MemoryWord {
                address: MEMORY_SIZE_ADDR,
                value: 65_536
            },
            MemoryWord {
                address: GLOBALS_BASE,
                value: 0x1122_3344_5566_7788
            },
            // Bytes 3..8 of the segment: `aa` in the first wasm word's cell,
            // `bb cc dd ee` in the second's.
            MemoryWord {
                address: linear_address(0),
                value: 0xAA00_0000
            },
            MemoryWord {
                address: linear_address(4),
                value: 0xEEDD_CCBB
            },
        ]
    );
    let io = PublicIo {
        entry: "touch".into(),
        inputs: vec![3],
        outputs: run.results.clone(),
    };
    let initial = io.initial_memory(&program);
    assert!(initial.contains(&MemoryWord {
        address: input_address(0),
        value: 3
    }));
    assert!(pre.program_memory.iter().all(|w| initial.contains(w)));
    // A run that only reads its inputs leaves the program words intact...
    let fresh = Machine::new(&program).unwrap().invoke("fib", &[1]).unwrap();
    for word in &pre.program_memory {
        assert_eq!(
            fresh.memory.read_word(word.address).unwrap(),
            word.value,
            "{word:?}"
        );
    }
    // ...and the final memory holds the public outputs and termination word.
    assert_eq!(
        io.final_memory(),
        vec![
            MemoryWord {
                address: output_address(0),
                value: 0x1122_3344_5566_7788 + 0xaa
            },
            MemoryWord {
                address: TERMINATION_ADDR,
                value: 1
            },
        ]
    );
    for word in io.final_memory() {
        assert_eq!(
            run.memory.read_word(word.address).unwrap(),
            word.value,
            "{word:?}"
        );
    }
}

#[test]
fn rejects_programs_without_the_halt_trampoline() {
    let mut program = program();
    program.code[0] = Ir::Nop;
    assert!(matches!(
        WasmBytecode::preprocess(&program),
        Err(jolt_wasm_program::PreprocessingError::MissingHaltTrampoline(_))
    ));
}

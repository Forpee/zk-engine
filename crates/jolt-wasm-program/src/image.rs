//! The public memory images: the non-zero 64-bit words of guest memory
//! before execution — data segments in linear memory, globals, the
//! memory-size word, and the entry's arguments in the input words — as the
//! RAM argument's initial state; and the words the final state must hold —
//! the results in the output words and the termination word.

use jolt_wasm_ir::layout::linear_address;
use jolt_wasm_ir::layout::{
    input_address, output_address, table_slot_address, GLOBALS_BASE, LINEAR_CELL_BYTES,
    MEMORY_SIZE_ADDR, PAGE_SIZE, TERMINATION_ADDR, WASM_WORD_BYTES,
};
use jolt_wasm_ir::{IrProgram, TableSlot};

/// One initial memory word at an absolute, 8-byte-aligned guest address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct MemoryWord {
    pub address: u64,
    pub value: u64,
}

/// The non-zero initial words, in increasing address order, with `inputs`
/// (the entry's arguments) in the public input words. Later data segments
/// overwrite earlier ones byte-wise, as at instantiation.
pub fn initial_memory_words(program: &IrProgram, inputs: &[u64]) -> Vec<MemoryWord> {
    let mut words = std::collections::BTreeMap::<u64, u64>::new();
    let mut set_byte = |address: u64, byte: u8| {
        let word_address = address & !7;
        let shift = (address & 7) * 8;
        let word = words.entry(word_address).or_insert(0);
        *word = (*word & !(0xFFu64 << shift)) | (u64::from(byte) << shift);
    };
    for segment in &program.data {
        for (i, byte) in segment.bytes.iter().enumerate() {
            // Cell of the containing wasm word, then the byte within it.
            let a = segment.offset + i as u64;
            let cell = linear_address(a - a % WASM_WORD_BYTES);
            debug_assert!(cell.is_multiple_of(LINEAR_CELL_BYTES));
            set_byte(cell + a % WASM_WORD_BYTES, *byte);
        }
    }
    for (i, value) in program.globals.iter().enumerate() {
        for (b, byte) in value.to_le_bytes().into_iter().enumerate() {
            set_byte(GLOBALS_BASE + 8 * i as u64 + b as u64, byte);
        }
    }
    for (i, slot) in program.table.iter().enumerate() {
        for (word, value) in TableSlot::words(*slot).into_iter().enumerate() {
            for (b, byte) in value.to_le_bytes().into_iter().enumerate() {
                set_byte(
                    table_slot_address(i as u64) + 8 * word as u64 + b as u64,
                    byte,
                );
            }
        }
    }
    let size = program.memory.initial_pages * PAGE_SIZE;
    for (b, byte) in size.to_le_bytes().into_iter().enumerate() {
        set_byte(MEMORY_SIZE_ADDR + b as u64, byte);
    }
    for (i, input) in inputs.iter().enumerate() {
        for (b, byte) in input.to_le_bytes().into_iter().enumerate() {
            set_byte(input_address(i as u64) + b as u64, byte);
        }
    }
    words
        .into_iter()
        .filter(|(_, value)| *value != 0)
        .map(|(address, value)| MemoryWord { address, value })
        .collect()
}

/// The public words a completed execution's final memory must hold: the
/// results in the output words and the termination word set to `1`.
pub fn final_public_words(outputs: &[u64]) -> Vec<MemoryWord> {
    let mut words: Vec<MemoryWord> = outputs
        .iter()
        .enumerate()
        .map(|(k, value)| MemoryWord {
            address: output_address(k as u64),
            value: *value,
        })
        .collect();
    words.push(MemoryWord {
        address: TERMINATION_ADDR,
        value: 1,
    });
    words
}

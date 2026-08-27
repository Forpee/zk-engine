//! Guest address-space layout: disjoint, contiguous regions of one 64-bit
//! address space, so every record's RAM address is absolute and the RAM
//! argument's word index is dense (`(address − RAM_BASE) / 8`).
//!
//! All regions sit at or above `0x8000_0000`, the RAM window of Jolt's memory
//! argument, in this order: system words, public inputs, public outputs, the
//! shadow call stack (plus its guard page), globals, linear memory. Public
//! I/O lives in memory: the host writes the entry's arguments to
//! [`INPUTS_BASE`] before execution and reads its results from
//! [`OUTPUTS_BASE`] and the termination word after — the initial and final
//! memory states are the proof's public inputs.

pub const PAGE_SIZE: u64 = 65_536;
pub const WORD_BYTES: u64 = 8;

/// First guest address of the RAM argument's word index space.
pub const RAM_BASE: u64 = 0x9000_0000;

/// System words: slot 0 is the linear-memory size in bytes, slot 1 the
/// termination word (written `1` by an entry stub when it completes). The
/// system and public I/O words sit at the bottom of RAM so a program that
/// touches little memory proves over a small RAM domain.
pub const SYSTEM_BASE: u64 = RAM_BASE;
pub const SYSTEM_SIZE: u64 = 0x1000;
pub const MEMORY_SIZE_ADDR: u64 = SYSTEM_BASE;
pub const TERMINATION_ADDR: u64 = SYSTEM_BASE + 8;

/// Public inputs: the entry's arguments, one word each.
pub const INPUTS_BASE: u64 = SYSTEM_BASE + SYSTEM_SIZE;
pub const INPUTS_SIZE: u64 = 0x1000;
/// Public outputs: the entry's results, one word each.
pub const OUTPUTS_BASE: u64 = INPUTS_BASE + INPUTS_SIZE;
pub const OUTPUTS_SIZE: u64 = 0x1000;

/// Capacity of the public input / output regions in words.
pub const MAX_INPUT_WORDS: u64 = INPUTS_SIZE / WORD_BYTES;
pub const MAX_OUTPUT_WORDS: u64 = OUTPUTS_SIZE / WORD_BYTES;

/// Shadow call stack: spilled frames and return addresses, growing upward.
pub const SHADOW_STACK_BASE: u64 = OUTPUTS_BASE + OUTPUTS_SIZE;
pub const SHADOW_STACK_SIZE: u64 = 1 << 20;
/// Unmapped guard page above the shadow stack: a spill into it is the
/// `CallStackExhausted` trap, never a silent write into the globals.
pub const SHADOW_GUARD_SIZE: u64 = 0x1000;

/// Globals region: global `i` lives at `GLOBALS_BASE + 8 * i`.
pub const GLOBALS_BASE: u64 = SHADOW_STACK_BASE + SHADOW_STACK_SIZE + SHADOW_GUARD_SIZE;
pub const GLOBALS_SIZE: u64 = 0x1_0000;

/// Linear memory: wasm address `a` maps to guest address `LINEAR_MEMORY_BASE + a`.
pub const LINEAR_MEMORY_BASE: u64 = GLOBALS_BASE + GLOBALS_SIZE;

/// Default page cap when a module declares no maximum (256 pages = 16 MiB).
pub const DEFAULT_MAX_PAGES: u64 = 256;

/// Hard cap on linear memory (4 GiB): the full wasm32 address space.
pub const MAX_PAGES: u64 = 65_536;

/// Address of public input word `i`.
pub const fn input_address(i: u64) -> u64 {
    INPUTS_BASE + WORD_BYTES * i
}

/// Address of public output word `i`.
pub const fn output_address(i: u64) -> u64 {
    OUTPUTS_BASE + WORD_BYTES * i
}

/// The RAM argument's word index of an aligned guest address, or `None`
/// below the RAM window or misaligned.
pub const fn remap_word_address(address: u64) -> Option<u64> {
    if address < RAM_BASE || !address.is_multiple_of(WORD_BYTES) {
        None
    } else {
        Some((address - RAM_BASE) / WORD_BYTES)
    }
}

/// The guest address of a RAM word index.
pub const fn unmap_word_address(index: u64) -> u64 {
    RAM_BASE + index * WORD_BYTES
}

/// One past the last guest address a program with `max_pages` linear-memory
/// pages can touch (its slack word included).
pub const fn ram_end(max_pages: u64) -> u64 {
    LINEAR_MEMORY_BASE + max_pages * PAGE_SIZE + WORD_BYTES
}

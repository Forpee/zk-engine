//! Guest address-space layout: disjoint regions of one 64-bit address space,
//! so every record's RAM address is absolute. Shared by the lowering (which
//! bakes these into immediates) and the runtime.

pub const PAGE_SIZE: u64 = 65_536;
pub const WORD_BYTES: u64 = 8;

/// Shadow call stack: spilled frames and return addresses, growing upward.
pub const SHADOW_STACK_BASE: u64 = 0x1000_0000;
pub const SHADOW_STACK_SIZE: u64 = 1 << 20;

/// Globals region: global `i` lives at `GLOBALS_BASE + 8 * i`.
pub const GLOBALS_BASE: u64 = 0x2000_0000;

/// System words. Slot 0 holds the current linear-memory size in bytes.
pub const SYSTEM_BASE: u64 = 0x3000_0000;
pub const MEMORY_SIZE_ADDR: u64 = SYSTEM_BASE;
pub const SYSTEM_SIZE: u64 = 64;

/// Linear memory: wasm address `a` maps to guest address `LINEAR_MEMORY_BASE + a`.
pub const LINEAR_MEMORY_BASE: u64 = 0x8000_0000;

/// Default page cap when a module declares no maximum (256 pages = 16 MiB).
pub const DEFAULT_MAX_PAGES: u64 = 256;

/// Hard cap on linear memory (4 GiB): the full wasm32 address space.
pub const MAX_PAGES: u64 = 65_536;

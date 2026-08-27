//! Guest memory: the regions of `jolt_wasm_ir::layout` backed by byte
//! vectors. All accesses are aligned 64-bit words (the doubleword-addressable
//! RAM contract).

use jolt_wasm_ir::layout::{
    GLOBALS_BASE, LINEAR_MEMORY_BASE, PAGE_SIZE, SHADOW_STACK_BASE, SHADOW_STACK_SIZE, SYSTEM_BASE,
    SYSTEM_SIZE, WORD_BYTES,
};
use jolt_wasm_ir::MemoryLimits;

use crate::error::Trap;

/// Result of [`Memory::grow`]: the size word's old and new values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grown {
    pub old_pages: u64,
    pub old_bytes: u64,
    pub new_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct Memory {
    shadow: Vec<u8>,
    system: Vec<u8>,
    globals: Vec<u8>,
    /// Linear memory plus one zeroed slack word past the end, so a
    /// non-crossing access to the last bytes can still read its `Hi` word.
    linear: Vec<u8>,
    max_pages: u64,
}

impl Memory {
    pub fn new(limits: MemoryLimits, globals: &[u64]) -> Self {
        let mut global_bytes = Vec::with_capacity(globals.len() * 8);
        for g in globals {
            global_bytes.extend_from_slice(&g.to_le_bytes());
        }
        let bytes = limits.initial_pages * PAGE_SIZE;
        let mut system = vec![0; SYSTEM_SIZE as usize];
        system[..8].copy_from_slice(&bytes.to_le_bytes());
        Self {
            shadow: vec![0; SHADOW_STACK_SIZE as usize],
            system,
            globals: global_bytes,
            linear: vec![0; (bytes + WORD_BYTES) as usize],
            max_pages: limits.max_pages,
        }
    }

    /// Current linear-memory size in bytes (the system size word).
    pub fn size_bytes(&self) -> u64 {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&self.system[..8]);
        u64::from_le_bytes(buf)
    }

    pub fn pages(&self) -> u64 {
        self.size_bytes() / PAGE_SIZE
    }

    /// Grow linear memory by `delta` pages and update the size word; `None`
    /// if the cap would be exceeded.
    pub fn grow(&mut self, delta: u64) -> Option<Grown> {
        let old_pages = self.pages();
        let new_pages = old_pages.checked_add(delta)?;
        if new_pages > self.max_pages {
            return None;
        }
        let old_bytes = old_pages * PAGE_SIZE;
        let new_bytes = new_pages * PAGE_SIZE;
        self.linear.resize((new_bytes + WORD_BYTES) as usize, 0);
        self.system[..8].copy_from_slice(&new_bytes.to_le_bytes());
        Some(Grown {
            old_pages,
            old_bytes,
            new_bytes,
        })
    }

    /// Non-zero linear-memory bytes as `(wasm offset, byte)` — the final
    /// memory image without the zero padding.
    pub fn linear_nonzero_bytes(&self) -> impl Iterator<Item = (u64, u8)> + '_ {
        let size = self.size_bytes() as usize;
        self.linear[..size]
            .iter()
            .enumerate()
            .filter(|(_, b)| **b != 0)
            .map(|(i, b)| (i as u64, *b))
    }

    /// Initialize linear memory bytes at a wasm offset (data segments).
    pub fn init_linear(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Trap> {
        let end = offset.checked_add(bytes.len() as u64);
        if end.is_none_or(|end| end > self.size_bytes()) {
            return Err(Trap::OutOfBoundsMemory {
                address: LINEAR_MEMORY_BASE.wrapping_add(offset),
                width: bytes.len().min(255) as u8,
            });
        }
        let start = offset as usize;
        self.linear[start..start + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    pub fn read_word(&self, address: u64) -> Result<u64, Trap> {
        let (region, start) = self.locate(address)?;
        let buf = match region {
            Region::Shadow => &self.shadow,
            Region::System => &self.system,
            Region::Globals => &self.globals,
            Region::Linear => &self.linear,
        };
        let mut word = [0u8; 8];
        word.copy_from_slice(&buf[start..start + 8]);
        Ok(u64::from_le_bytes(word))
    }

    /// Write one word; returns the previous value.
    pub fn write_word(&mut self, address: u64, value: u64) -> Result<u64, Trap> {
        let (region, start) = self.locate(address)?;
        let buf = match region {
            Region::Shadow => &mut self.shadow,
            Region::System => &mut self.system,
            Region::Globals => &mut self.globals,
            Region::Linear => &mut self.linear,
        };
        let slice = &mut buf[start..start + 8];
        let mut word = [0u8; 8];
        word.copy_from_slice(slice);
        slice.copy_from_slice(&value.to_le_bytes());
        Ok(u64::from_le_bytes(word))
    }

    fn locate(&self, address: u64) -> Result<(Region, usize), Trap> {
        if !address.is_multiple_of(WORD_BYTES) {
            return Err(Trap::UnalignedWord(address));
        }
        let oob = || Trap::OutOfBoundsMemory { address, width: 8 };
        let (region, base, size) = if address >= LINEAR_MEMORY_BASE {
            (Region::Linear, LINEAR_MEMORY_BASE, self.linear.len())
        } else if address >= SYSTEM_BASE {
            (Region::System, SYSTEM_BASE, self.system.len())
        } else if address >= GLOBALS_BASE {
            (Region::Globals, GLOBALS_BASE, self.globals.len())
        } else if address >= SHADOW_STACK_BASE {
            (Region::Shadow, SHADOW_STACK_BASE, self.shadow.len())
        } else {
            return Err(oob());
        };
        let start = (address - base) as usize;
        if start.checked_add(8).is_none_or(|end| end > size) {
            return Err(if region == Region::Shadow {
                Trap::CallStackExhausted
            } else {
                oob()
            });
        }
        Ok((region, start))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Region {
    Shadow,
    System,
    Globals,
    Linear,
}

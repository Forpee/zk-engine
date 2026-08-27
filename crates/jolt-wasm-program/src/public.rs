//! The public memory a proof binds: the run-independent initial words
//! (memory-size word, program image) plus the per-run public inputs, and the
//! public I/O window the final memory is checked against. Both the prover
//! and the verifier derive these from the same [`WasmProgramPreprocessing`]
//! and [`PublicIo`], over the dense RAM word index of
//! [`jolt_wasm_ir::layout`].

use jolt_wasm_ir::layout::{
    ram_end, GLOBALS_BASE, INPUTS_BASE, MEMORY_SIZE_ADDR, OUTPUTS_BASE, PAGE_SIZE, RAM_BASE,
    TERMINATION_ADDR, WORD_BYTES,
};
use jolt_wasm_ir::MemoryLimits;

use crate::{MemoryWord, PublicIo, WasmProgramPreprocessing};

/// The RAM word index of a layout constant (all lie at or above [`RAM_BASE`]).
const fn word_index(address: u64) -> u64 {
    (address - RAM_BASE) / WORD_BYTES
}

/// The RAM word index the program image starts at ([`GLOBALS_BASE`]).
pub const PROGRAM_IMAGE_START_INDEX: u64 = word_index(GLOBALS_BASE);

/// A run of consecutive public words starting at a RAM word index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicMemorySegment {
    pub start_index: u128,
    pub words: Vec<u64>,
}

/// The dense program image: `words[i]` is the initial value of RAM word
/// `start_index + i`, covering [`GLOBALS_BASE`] through the last non-zero
/// program word.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProgramImage {
    pub start_index: u64,
    pub words: Vec<u64>,
}

impl ProgramImage {
    pub(crate) fn of_words(program_memory: &[MemoryWord]) -> Self {
        let start_index = PROGRAM_IMAGE_START_INDEX;
        let end = program_memory
            .iter()
            .filter(|word| word.address >= GLOBALS_BASE)
            .map(|word| word_index(word.address) + 1)
            .max()
            .unwrap_or(start_index);
        let mut words = vec![0; (end - start_index) as usize];
        for word in program_memory
            .iter()
            .filter(|word| word.address >= GLOBALS_BASE)
        {
            words[(word_index(word.address) - start_index) as usize] = word.value;
        }
        Self { start_index, words }
    }

    /// One past the last image word's index.
    pub fn end_index(&self) -> u64 {
        self.start_index + self.words.len() as u64
    }
}

/// The public words the final memory must hold, and the I/O window
/// `[io_mask_start, io_mask_end)` — termination word, inputs, outputs —
/// inside which every word not listed must be zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicIoMemory {
    pub segments: Vec<PublicMemorySegment>,
    pub io_mask_start: u128,
    pub io_mask_end: u128,
    io_num_vars: usize,
}

impl PublicIoMemory {
    pub fn new(io: &PublicIo) -> Self {
        let io_mask_start = u128::from(word_index(TERMINATION_ADDR));
        let io_mask_end = u128::from(word_index(GLOBALS_BASE));
        let io_num_vars = io_mask_end.next_power_of_two().max(1).ilog2() as usize;
        let mut segments = vec![PublicMemorySegment {
            start_index: io_mask_start,
            words: vec![1],
        }];
        if !io.inputs.is_empty() {
            segments.push(PublicMemorySegment {
                start_index: u128::from(word_index(INPUTS_BASE)),
                words: io.inputs.clone(),
            });
        }
        if !io.outputs.is_empty() {
            segments.push(PublicMemorySegment {
                start_index: u128::from(word_index(OUTPUTS_BASE)),
                words: io.outputs.clone(),
            });
        }
        Self {
            segments,
            io_mask_start,
            io_mask_end,
            io_num_vars,
        }
    }

    /// Number of address variables spanning the I/O window.
    pub fn io_num_vars(&self) -> usize {
        self.io_num_vars
    }
}

/// The public initial memory: the memory-size word, the run's inputs, and
/// (when the program image is public rather than committed) the image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicInitialRam {
    pub segments: Vec<PublicMemorySegment>,
}

impl PublicInitialRam {
    pub fn new(preprocessing: &WasmProgramPreprocessing, io: &PublicIo) -> Self {
        let mut this = Self::inputs_only(preprocessing.memory, io);
        let image = preprocessing.program_image();
        if !image.words.is_empty() {
            this.segments.push(PublicMemorySegment {
                start_index: u128::from(image.start_index),
                words: image.words,
            });
        }
        this
    }

    /// The initial words that are public even when the program image is
    /// committed: the memory-size word and the inputs.
    pub fn inputs_only(memory: MemoryLimits, io: &PublicIo) -> Self {
        let mut segments = vec![PublicMemorySegment {
            start_index: u128::from(word_index(MEMORY_SIZE_ADDR)),
            words: vec![memory.initial_pages * PAGE_SIZE],
        }];
        if !io.inputs.is_empty() {
            segments.push(PublicMemorySegment {
                start_index: u128::from(word_index(INPUTS_BASE)),
                words: io.inputs.clone(),
            });
        }
        Self { segments }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RamDomainError {
    #[error("RAM domain does not fit a power-of-two usize")]
    DomainTooLarge,
}

/// Smallest power-of-two RAM domain holding the public I/O window and the
/// program image (`program_image_end_index` words from index zero).
pub fn min_ram_k(program_image_end_index: u64) -> Result<usize, RamDomainError> {
    let words = program_image_end_index.max(word_index(GLOBALS_BASE));
    usize::try_from(words)
        .ok()
        .and_then(usize::checked_next_power_of_two)
        .ok_or(RamDomainError::DomainTooLarge)
}

/// Smallest power-of-two RAM domain holding every address a program with
/// these limits can touch.
pub fn max_ram_k(memory: MemoryLimits) -> Result<usize, RamDomainError> {
    let end = ram_end(memory.max_pages);
    usize::try_from(word_index(end) + 1)
        .ok()
        .and_then(usize::checked_next_power_of_two)
        .ok_or(RamDomainError::DomainTooLarge)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests fail loudly")]
mod tests {
    use super::*;

    fn io() -> PublicIo {
        PublicIo {
            entry: "main".to_owned(),
            inputs: vec![3, 4],
            outputs: vec![7],
        }
    }

    #[test]
    fn io_memory_lists_termination_inputs_outputs_inside_the_mask() {
        let memory = PublicIoMemory::new(&io());
        assert_eq!(memory.segments.len(), 3);
        assert_eq!(memory.segments[0].words, vec![1]);
        assert_eq!(memory.segments[1].words, vec![3, 4]);
        assert_eq!(memory.segments[2].words, vec![7]);
        for segment in &memory.segments {
            let end = segment.start_index + segment.words.len() as u128;
            assert!(segment.start_index >= memory.io_mask_start && end <= memory.io_mask_end);
        }
        assert!(memory.io_mask_end <= 1u128 << memory.io_num_vars());
    }

    #[test]
    fn initial_ram_holds_size_word_inputs_and_image() {
        let words = vec![
            MemoryWord {
                address: MEMORY_SIZE_ADDR,
                value: PAGE_SIZE,
            },
            MemoryWord {
                address: GLOBALS_BASE + 16,
                value: 9,
            },
        ];
        let image = ProgramImage::of_words(&words);
        assert_eq!(image.start_index, word_index(GLOBALS_BASE));
        assert_eq!(image.words, vec![0, 0, 9]);

        let memory = MemoryLimits {
            initial_pages: 1,
            max_pages: 1,
        };
        let initial = PublicInitialRam::inputs_only(memory, &io());
        assert_eq!(initial.segments[0].words, vec![PAGE_SIZE]);
        assert_eq!(initial.segments[1].words, vec![3, 4]);
        assert!(min_ram_k(image.end_index()).unwrap() >= image.end_index() as usize);
        assert!(max_ram_k(memory).unwrap() >= min_ram_k(image.end_index()).unwrap());
    }
}

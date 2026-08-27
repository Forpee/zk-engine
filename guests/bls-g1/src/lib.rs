//! BLS12-381 G1 scalar multiplication guest: `g1_mul(s0, s1, s2, s3)`
//! computes `[s]·G1` for the 256-bit little-endian scalar `s` and returns
//! (through memory, see `jolt.outputs.g1_mul`) the point's 48-byte
//! compressed encoding as six little-endian `u64` words. The
//! multiplication runs over the scalar's significant bits, so the trace
//! length scales with `log2(s)`.

#![no_std]

use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::panic::PanicInfo;

use blst::{blst_p1, blst_p1_compress, blst_p1_generator, blst_p1_mult};

/// Number of public output words `g1_mul` writes through its pointer.
#[link_section = "jolt.outputs.g1_mul"]
#[used]
static G1_MUL_OUTPUT_WORDS: [u8; 4] = 6u32.to_le_bytes();

static mut OUTPUT: [u64; 6] = [0; 6];

#[no_mangle]
pub extern "C" fn g1_mul(s0: u64, s1: u64, s2: u64, s3: u64) -> u32 {
    let mut scalar = [0u8; 32];
    for (chunk, limb) in scalar.chunks_exact_mut(8).zip([s0, s1, s2, s3]) {
        chunk.copy_from_slice(&limb.to_le_bytes());
    }
    let leading_zeros = [s3, s2, s1, s0]
        .iter()
        .try_fold(0usize, |acc, limb| match limb.leading_zeros() as usize {
            64 => Ok(acc + 64),
            lz => Err(acc + lz),
        })
        .unwrap_or_else(|lz| lz);
    let nbits = (256 - leading_zeros).max(1);
    let mut point = blst_p1::default();
    let mut compressed = [0u8; 48];
    // SAFETY: blst_p1_generator returns a valid static point; the buffers
    // have the sizes blst requires (32-byte scalar, 48-byte compressed).
    unsafe {
        blst_p1_mult(&mut point, blst_p1_generator(), scalar.as_ptr(), nbits);
        blst_p1_compress(compressed.as_mut_ptr(), &point);
        for (word, bytes) in OUTPUT.iter_mut().zip(compressed.chunks_exact(8)) {
            *word = u64::from_le_bytes(bytes.try_into().unwrap());
        }
        core::ptr::addr_of!(OUTPUT) as u32
    }
}

/// Bump allocator over a static arena: `blst`'s bindings need `alloc`,
/// and a guest run never frees.
struct Bump(UnsafeCell<(usize, [u8; ARENA_BYTES])>);
const ARENA_BYTES: usize = 1 << 16;

// SAFETY: the guest is single-threaded.
unsafe impl Sync for Bump {}

unsafe impl GlobalAlloc for Bump {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let (used, arena) = &mut *self.0.get();
        let start = (*used + layout.align() - 1) & !(layout.align() - 1);
        let end = start + layout.size();
        if end > arena.len() {
            return core::ptr::null_mut();
        }
        *used = end;
        arena.as_mut_ptr().add(start)
    }

    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
}

#[global_allocator]
static ALLOCATOR: Bump = Bump(UnsafeCell::new((0, [0; ARENA_BYTES])));

#[panic_handler]
fn panic(_: &PanicInfo<'_>) -> ! {
    core::arch::wasm32::unreachable()
}

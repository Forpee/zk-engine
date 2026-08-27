//! Spongefish-native [`ProverTranscript`] surface.
//!
//! Implemented directly on `spongefish::ProverState<H, R>` via the orphan
//! rule. Methods are positional, matching spongefish-native usage
//! (WhiR, sigma-rs).

use jolt_field::Fr;
use rand::{CryptoRng, RngCore};
use spongefish::{Decoding, DuplexSpongeInterface, Encoding, NargSerialize, ProverState};

/// Prover-side spongefish transcript.
///
/// `H::U` is the sponge alphabet (`u8` for every sponge in this crate).
pub trait ProverTranscript<H: DuplexSpongeInterface> {
    /// Absorbs `msg` symmetrically with the verifier; emits no NARG bytes.
    fn public_message<T: Encoding<[H::U]> + ?Sized>(&mut self, msg: &T);

    /// Absorbs `msg` and appends its NARG-serialized form for verifier replay.
    fn prover_message<T: Encoding<[H::U]> + NargSerialize + ?Sized>(&mut self, msg: &T);

    /// Squeezes a verifier challenge.
    fn verifier_message<T: Decoding<[H::U]>>(&mut self) -> T;

    /// Bytes accumulated in the NARG so far.
    fn narg_string(&self) -> &[u8];
}

impl<H, R> ProverTranscript<H> for ProverState<H, R>
where
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
{
    fn public_message<T: Encoding<[H::U]> + ?Sized>(&mut self, msg: &T) {
        ProverState::public_message(self, msg);
    }

    fn prover_message<T: Encoding<[H::U]> + NargSerialize + ?Sized>(&mut self, msg: &T) {
        ProverState::prover_message(self, msg);
    }

    fn verifier_message<T: Decoding<[H::U]>>(&mut self) -> T {
        ProverState::verifier_message::<T>(self)
    }

    fn narg_string(&self) -> &[u8] {
        ProverState::narg_string(self)
    }
}

/// 128-bit-truncating challenge decoder, sound for the Blake2b sponge
/// (uniform output bytes). Deliberately not implemented for algebraic
/// sponges whose squeezed field elements are not uniform bytes.
pub trait OptimizedChallenge {
    /// Squeezes a 128-bit-truncated challenge as an [`Fr`].
    fn challenge_128(&mut self) -> Fr;
}

impl<R> OptimizedChallenge for ProverState<spongefish::instantiations::Blake2b512, R>
where
    R: RngCore + CryptoRng,
{
    fn challenge_128(&mut self) -> Fr {
        Fr::from(ProverState::verifier_message::<u128>(self))
    }
}

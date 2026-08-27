//! ZK verifier stage construction and committed-proof boundary helpers.

/// BlindFold applies only to the homomorphic build: no zk protocol exists
/// over the packed axis (`akita` and `zk` are mutually exclusive features).
#[doc(hidden)]
pub mod blindfold;
pub(crate) mod committed;
#[doc(hidden)]
pub mod inputs;
#[doc(hidden)]
pub mod outputs;

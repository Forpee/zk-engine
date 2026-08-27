//! Stage 8: the final PCS opening. [`verify`] is the per-build entry point;
//! the feature-specific statement assembly lives beside it.

pub mod outputs;
pub mod precommitted;
mod verify;

pub use outputs::Stage8ClearOutput;
pub use outputs::{Stage8Output, Stage8ZkOutput};
pub use precommitted::precommitted_final_openings;
pub use verify::verify;
pub use verify::{batch_entries, Stage8BatchEntry};

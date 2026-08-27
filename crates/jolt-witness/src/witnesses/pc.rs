use jolt_field::JoltField;

use super::{Extract, ToField, WitnessEnv};
use crate::{TraceRow, WitnessError};

/// The cycle's bytecode slot, for both the read-RAF pushforward and the
/// committed one-hot. Total: IR pcs are dense and the padding row is the
/// `Halt` trampoline at slot 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BytecodePc(pub usize);

impl Extract for BytecodePc {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.pc() as usize))
    }
}

/// The row's program counter (its bytecode index).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pc(pub u64);

/// [`Pc`] of the successor row; 0 at the last cycle (the padding row's pc).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NextPc(pub u64);

impl ToField for Pc {
    fn to_field<F: JoltField>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for Pc {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(u64::from(row.pc())))
    }
}

impl ToField for NextPc {
    fn to_field<F: JoltField>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for NextPc {
    fn extract(
        _row: &TraceRow,
        next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(next.map_or(0, |row| u64::from(row.pc()))))
    }
}

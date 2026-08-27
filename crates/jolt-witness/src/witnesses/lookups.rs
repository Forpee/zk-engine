use jolt_field::JoltField;

use super::{Extract, ToField, WitnessEnv};
use crate::{TraceRow, WitnessError};

/// Output of the row's lookup.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LookupOutput(pub u64);

/// The row's 128-bit lookup index (its interleaved or raw lookup operands);
/// 0 for rows without a lookup.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LookupIndex(pub u128);

/// Which catalog table the row's lookup targets, if any.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TableIndex(pub Option<usize>);

impl ToField for LookupOutput {
    fn to_field<F: JoltField>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for LookupOutput {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.lookup_output()))
    }
}

impl ToField for LookupIndex {
    fn to_field<F: JoltField>(self) -> F {
        F::from_u128(self.0)
    }
}

impl Extract for LookupIndex {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.lookup_index().unwrap_or(0)))
    }
}

impl Extract for TableIndex {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.table()))
    }
}

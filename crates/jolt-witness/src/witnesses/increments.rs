use jolt_field::Field;

use super::{Extract, ToField, WitnessEnv};
use crate::{TraceRow, WitnessError};

/// Signed delta written to rd this cycle; 0 when the row has no rd operand.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RdInc(pub i128);

/// Signed delta written to RAM this cycle; 0 for reads and no-ops.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamInc(pub i128);

impl ToField for RdInc {
    fn to_field<F: Field>(self) -> F {
        F::from_i128(self.0)
    }
}

impl Extract for RdInc {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(if row.rd_index().is_some() {
            i128::from(row.rd_write_value()) - i128::from(row.rd_pre_value())
        } else {
            0
        }))
    }
}

impl ToField for RamInc {
    fn to_field<F: Field>(self) -> F {
        F::from_i128(self.0)
    }
}

impl Extract for RamInc {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(if row.is_store() {
            i128::from(row.ram_write_value()) - i128::from(row.ram_read_value())
        } else {
            0
        }))
    }
}

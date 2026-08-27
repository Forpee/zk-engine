use jolt_field::JoltField;

use super::{Extract, ToField, WitnessEnv};
use crate::{TraceRow, WitnessError};

/// Left lookup operand of the row's lookup.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LeftLookupOperand(pub u64);

/// Right lookup operand of the row's lookup (the raw index for
/// combined-operand and advice rows).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RightLookupOperand(pub u128);

/// Left instruction input (`rs1` or 0, per the row flags).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LeftInstructionInput(pub u64);

/// Right instruction input (`rs2`, the immediate, or 0, per the row flags).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RightInstructionInput(pub u64);

/// Product of the instruction inputs (exact: two 64-bit values).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Product(pub u128);

/// The row's immediate as the constraints see it (signed byte offset on
/// memory rows).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Imm(pub i128);

impl ToField for LeftLookupOperand {
    fn to_field<F: JoltField>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for LeftLookupOperand {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.lookup_operands().0))
    }
}

impl ToField for RightLookupOperand {
    fn to_field<F: JoltField>(self) -> F {
        F::from_u128(self.0)
    }
}

impl Extract for RightLookupOperand {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.lookup_operands().1))
    }
}

impl ToField for LeftInstructionInput {
    fn to_field<F: JoltField>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for LeftInstructionInput {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.left_input()))
    }
}

impl ToField for RightInstructionInput {
    fn to_field<F: JoltField>(self) -> F {
        F::from_u64(self.0)
    }
}

impl Extract for RightInstructionInput {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.right_input()))
    }
}

impl ToField for Product {
    fn to_field<F: JoltField>(self) -> F {
        F::from_u128(self.0)
    }
}

impl Extract for Product {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(
            u128::from(row.left_input()) * u128::from(row.right_input()),
        ))
    }
}

impl ToField for Imm {
    fn to_field<F: JoltField>(self) -> F {
        F::from_i128(self.0)
    }
}

impl Extract for Imm {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.imm_signed()))
    }
}

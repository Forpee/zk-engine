use jolt_field::JoltField;
use jolt_wasm_ir::RowFlag;

use super::{Extract, ExtractIndexed, ToField, WitnessEnv};
use crate::{TraceRow, WitnessError};

/// Branch row whose comparison output is 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShouldBranch(pub bool);

/// Set when the row's lookup index is its raw right lookup operand (the RAF
/// address decomposition applies): combined-operand tables and advice rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InstructionRafFlag(pub bool);

/// One row flag of the instruction; which flag is bound at the use site.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Flag(pub bool);

/// Whether the row's lookup targets the catalog table bound at the use site.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LookupTableFlag(pub bool);

macro_rules! bool_to_field {
    ($($name:ident),* $(,)?) => {
        $(impl ToField for $name {
            fn to_field<F: JoltField>(self) -> F {
                F::from_bool(self.0)
            }
        })*
    };
}
bool_to_field!(ShouldBranch, InstructionRafFlag, Flag, LookupTableFlag);

impl Extract for ShouldBranch {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(
            row.flags().has(RowFlag::Branch) && row.lookup_output() == 1,
        ))
    }
}

impl Extract for InstructionRafFlag {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.raf_flag()))
    }
}

impl ExtractIndexed<RowFlag> for Flag {
    fn extract_indexed(
        flag: RowFlag,
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.flags().has(flag)))
    }
}

impl ExtractIndexed<usize> for LookupTableFlag {
    fn extract_indexed(
        table: usize,
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.table() == Some(table)))
    }
}

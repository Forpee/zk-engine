//! The per-cycle R1CS witness of a [`WasmTraceRow`] under
//! `jolt_r1cs::constraints::wasm` — the variable layout the WASM uniform
//! constraints are written against, filled from the row's columns.

use jolt_field::JoltField;
use jolt_r1cs::constraints::wasm::{
    NUM_FLAGS, NUM_VARS_PER_CYCLE, V_CONST, V_FLAGS, V_IMM, V_LEFT_INPUT, V_LEFT_LOOKUP_OPERAND,
    V_LOOKUP_OUTPUT, V_NEXT_PC, V_PC, V_PRODUCT, V_RAM_ADDRESS, V_RAM_READ_VALUE,
    V_RAM_WRITE_VALUE, V_RD_WRITE_VALUE, V_RIGHT_INPUT, V_RIGHT_LOOKUP_OPERAND, V_RS1_VALUE,
    V_RS2_VALUE, V_SHOULD_BRANCH,
};
use jolt_wasm_ir::row::RowFlag;

use crate::trace_row::WasmTraceRow;

/// The row's witness vector, indexed by the `V_*` constants.
pub fn cycle_witness<F: JoltField>(row: &WasmTraceRow) -> Vec<F> {
    let mut w = vec![F::zero(); NUM_VARS_PER_CYCLE];
    let (left, right) = (row.left_input(), row.right_input());
    let output = row.lookup_output();
    let flags = row.flags();
    let (left_lookup, right_lookup) = row.lookup_operands();

    w[V_CONST] = F::one();
    w[V_LEFT_INPUT] = F::from_u64(left);
    w[V_RIGHT_INPUT] = F::from_u64(right);
    w[V_PRODUCT] = F::from_u128(u128::from(left) * u128::from(right));
    w[V_PC] = F::from_u64(u64::from(row.pc()));
    w[V_IMM] = F::from_i128(row.imm_signed());
    w[V_RAM_ADDRESS] = F::from_u64(row.ram_address());
    w[V_RS1_VALUE] = F::from_u64(row.rs1_value());
    w[V_RS2_VALUE] = F::from_u64(row.rs2_value());
    w[V_RD_WRITE_VALUE] = F::from_u64(row.rd_write_value());
    w[V_RAM_READ_VALUE] = F::from_u64(row.ram_read_value());
    w[V_RAM_WRITE_VALUE] = F::from_u64(row.ram_write_value());
    w[V_LEFT_LOOKUP_OPERAND] = F::from_u64(left_lookup);
    w[V_RIGHT_LOOKUP_OPERAND] = F::from_u128(right_lookup);
    w[V_NEXT_PC] = F::from_u64(u64::from(row.next_pc()));
    w[V_LOOKUP_OUTPUT] = F::from_u64(output);
    let branch = flags.has(RowFlag::Branch);
    w[V_SHOULD_BRANCH] = F::from_u64(u64::from(branch && output == 1));
    debug_assert_eq!(NUM_FLAGS, RowFlag::COUNT);
    for i in 0..NUM_FLAGS {
        if flags.bits() & (1 << i) != 0 {
            w[V_FLAGS + i] = F::one();
        }
    }
    w
}

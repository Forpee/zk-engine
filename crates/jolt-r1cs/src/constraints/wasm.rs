//! Jolt WebAssembly uniform R1CS.
//!
//! The per-cycle constraint set over the WebAssembly proof row
//! (`jolt-wasm-program`'s `WasmTraceRow` columns and `jolt-wasm-backend`'s
//! `RowFlags`): the constraint-form `check_record` of `jolt-wasm-backend`
//! transcribed into `guard · (left − right) = 0` and `left · right = output`
//! rows. Branch/jump targets are absolute (a branch's `imm` is its
//! target pc), jumps write no link register, a `Halt` row class holds the pc
//! (the padding row), every lookup row takes a register-or-immediate right
//! operand, and the instruction-input selection is constrained here rather
//! than by a separate sumcheck.
//!
//! # Variable layout
//!
//! | Range | Description |
//! |-------|-------------|
//! | `[0]` | Constant 1 |
//! | `[1..=16]` | Inputs (values, lookup operands, pc/next pc) |
//! | `[17..=31]` | Row flags (`RowFlag` bit order) |
//!
//! `ShouldBranch` (`[16]`) is the product factor `LookupOutput · Branch`.

use jolt_field::Field;

use super::{row, row_wide};
use crate::constraint::SparseRow;

pub const V_CONST: usize = 0;

pub const V_LEFT_INPUT: usize = 1;
pub const V_RIGHT_INPUT: usize = 2;
pub const V_PRODUCT: usize = 3;
pub const V_PC: usize = 4;
pub const V_IMM: usize = 5;
pub const V_RAM_ADDRESS: usize = 6;
pub const V_RS1_VALUE: usize = 7;
pub const V_RS2_VALUE: usize = 8;
pub const V_RD_WRITE_VALUE: usize = 9;
pub const V_RAM_READ_VALUE: usize = 10;
pub const V_RAM_WRITE_VALUE: usize = 11;
pub const V_LEFT_LOOKUP_OPERAND: usize = 12;
pub const V_RIGHT_LOOKUP_OPERAND: usize = 13;
pub const V_NEXT_PC: usize = 14;
pub const V_LOOKUP_OUTPUT: usize = 15;
pub const V_SHOULD_BRANCH: usize = 16;

/// First flag variable; flag `i` (in `RowFlags` bit order) is `V_FLAGS + i`.
pub const V_FLAGS: usize = 17;
pub const V_FLAG_LEFT_IS_RS1: usize = V_FLAGS;
pub const V_FLAG_RIGHT_IS_RS2: usize = V_FLAGS + 1;
pub const V_FLAG_RIGHT_IS_IMM: usize = V_FLAGS + 2;
pub const V_FLAG_ADD_OPERANDS: usize = V_FLAGS + 3;
pub const V_FLAG_SUB_OPERANDS: usize = V_FLAGS + 4;
pub const V_FLAG_MUL_OPERANDS: usize = V_FLAGS + 5;
pub const V_FLAG_WRITE_LOOKUP_TO_RD: usize = V_FLAGS + 6;
pub const V_FLAG_LOAD: usize = V_FLAGS + 7;
pub const V_FLAG_STORE: usize = V_FLAGS + 8;
pub const V_FLAG_JUMP: usize = V_FLAGS + 9;
pub const V_FLAG_BRANCH: usize = V_FLAGS + 10;
pub const V_FLAG_ASSERT: usize = V_FLAGS + 11;
pub const V_FLAG_HALT: usize = V_FLAGS + 12;
pub const V_FLAG_TRAP: usize = V_FLAGS + 13;
pub const V_FLAG_ADVICE: usize = V_FLAGS + 14;
pub const NUM_FLAGS: usize = 15;

/// The R1CS inputs: every variable but the constant.
pub const NUM_R1CS_INPUTS: usize = NUM_VARS_PER_CYCLE - 1;
pub const NUM_VARS_PER_CYCLE: usize = V_FLAGS + NUM_FLAGS; // 32
pub const NUM_EQ_CONSTRAINTS: usize = 22;
pub const NUM_PRODUCT_CONSTRAINTS: usize = 2;
pub const NUM_CONSTRAINTS_PER_CYCLE: usize = NUM_EQ_CONSTRAINTS + NUM_PRODUCT_CONSTRAINTS; // 24

pub const fn const_column() -> usize {
    V_CONST
}

pub const fn input_column(input_index: usize) -> Option<usize> {
    if input_index < NUM_R1CS_INPUTS {
        Some(1 + input_index)
    } else {
        None
    }
}

/// Two's complement bias for subtraction: 2^64.
const TWOS_COMPLEMENT_BIAS: i128 = 0x1_0000_0000_0000_0000;

type ConstraintRows<F> = (Vec<SparseRow<F>>, Vec<SparseRow<F>>, Vec<SparseRow<F>>);

fn wasm_eq_constraint_rows<F: Field>() -> ConstraintRows<F> {
    let mut a: Vec<SparseRow<F>> = Vec::with_capacity(NUM_EQ_CONSTRAINTS);
    let mut b: Vec<SparseRow<F>> = Vec::with_capacity(NUM_EQ_CONSTRAINTS);
    let mut c: Vec<SparseRow<F>> = Vec::with_capacity(NUM_EQ_CONSTRAINTS);
    let mut eq = |guard: SparseRow<F>, diff: SparseRow<F>| {
        a.push(guard);
        b.push(diff);
        c.push(Vec::new());
    };

    // 0: LeftInputEqRs1IfLeftIsRs1
    eq(
        row(&[(V_FLAG_LEFT_IS_RS1, 1)]),
        row(&[(V_LEFT_INPUT, 1), (V_RS1_VALUE, -1)]),
    );
    // 1: LeftInputZeroOtherwise
    eq(
        row(&[(V_CONST, 1), (V_FLAG_LEFT_IS_RS1, -1)]),
        row(&[(V_LEFT_INPUT, 1)]),
    );
    // 2: RightInputEqRs2IfRightIsRs2
    eq(
        row(&[(V_FLAG_RIGHT_IS_RS2, 1)]),
        row(&[(V_RIGHT_INPUT, 1), (V_RS2_VALUE, -1)]),
    );
    // 3: RightInputEqImmIfRightIsImm
    eq(
        row(&[(V_FLAG_RIGHT_IS_IMM, 1)]),
        row(&[(V_RIGHT_INPUT, 1), (V_IMM, -1)]),
    );
    // 4: RightInputZeroOtherwise
    eq(
        row(&[
            (V_CONST, 1),
            (V_FLAG_RIGHT_IS_RS2, -1),
            (V_FLAG_RIGHT_IS_IMM, -1),
        ]),
        row(&[(V_RIGHT_INPUT, 1)]),
    );
    // 5: RamAddrEqRs1PlusImmIfLoadStore
    eq(
        row(&[(V_FLAG_LOAD, 1), (V_FLAG_STORE, 1)]),
        row(&[(V_RAM_ADDRESS, 1), (V_RS1_VALUE, -1), (V_IMM, -1)]),
    );
    // 6: RamAddrZeroOtherwise
    eq(
        row(&[(V_CONST, 1), (V_FLAG_LOAD, -1), (V_FLAG_STORE, -1)]),
        row(&[(V_RAM_ADDRESS, 1)]),
    );
    // 7: RamReadEqRamWriteIfLoad
    eq(
        row(&[(V_FLAG_LOAD, 1)]),
        row(&[(V_RAM_READ_VALUE, 1), (V_RAM_WRITE_VALUE, -1)]),
    );
    // 8: RamReadEqRdWriteIfLoad
    eq(
        row(&[(V_FLAG_LOAD, 1)]),
        row(&[(V_RAM_READ_VALUE, 1), (V_RD_WRITE_VALUE, -1)]),
    );
    // 9: Rs2EqRamWriteIfStore
    eq(
        row(&[(V_FLAG_STORE, 1)]),
        row(&[(V_RS2_VALUE, 1), (V_RAM_WRITE_VALUE, -1)]),
    );
    // 10: LeftLookupZeroIfAddSubMul
    eq(
        row(&[
            (V_FLAG_ADD_OPERANDS, 1),
            (V_FLAG_SUB_OPERANDS, 1),
            (V_FLAG_MUL_OPERANDS, 1),
        ]),
        row(&[(V_LEFT_LOOKUP_OPERAND, 1)]),
    );
    // 11: LeftLookupEqLeftInputOtherwise
    eq(
        row(&[
            (V_CONST, 1),
            (V_FLAG_ADD_OPERANDS, -1),
            (V_FLAG_SUB_OPERANDS, -1),
            (V_FLAG_MUL_OPERANDS, -1),
        ]),
        row(&[(V_LEFT_LOOKUP_OPERAND, 1), (V_LEFT_INPUT, -1)]),
    );
    // 12: RightLookupAdd
    eq(
        row(&[(V_FLAG_ADD_OPERANDS, 1)]),
        row(&[
            (V_RIGHT_LOOKUP_OPERAND, 1),
            (V_LEFT_INPUT, -1),
            (V_RIGHT_INPUT, -1),
        ]),
    );
    // 13: RightLookupSub (left − right + 2^64)
    eq(
        row(&[(V_FLAG_SUB_OPERANDS, 1)]),
        row_wide(&[
            (V_RIGHT_LOOKUP_OPERAND, 1),
            (V_LEFT_INPUT, -1),
            (V_RIGHT_INPUT, 1),
            (V_CONST, -TWOS_COMPLEMENT_BIAS),
        ]),
    );
    // 14: RightLookupEqProductIfMul
    eq(
        row(&[(V_FLAG_MUL_OPERANDS, 1)]),
        row(&[(V_RIGHT_LOOKUP_OPERAND, 1), (V_PRODUCT, -1)]),
    );
    // 15: RightLookupEqRightInputOtherwise (advice rows look nothing up)
    eq(
        row(&[
            (V_CONST, 1),
            (V_FLAG_ADD_OPERANDS, -1),
            (V_FLAG_SUB_OPERANDS, -1),
            (V_FLAG_MUL_OPERANDS, -1),
            (V_FLAG_ADVICE, -1),
        ]),
        row(&[(V_RIGHT_LOOKUP_OPERAND, 1), (V_RIGHT_INPUT, -1)]),
    );
    // 16: AssertLookupOne
    eq(
        row(&[(V_FLAG_ASSERT, 1)]),
        row(&[(V_LOOKUP_OUTPUT, 1), (V_CONST, -1)]),
    );
    // 17: RdWriteEqLookupIfWriteLookupToRd
    eq(
        row(&[(V_FLAG_WRITE_LOOKUP_TO_RD, 1)]),
        row(&[(V_RD_WRITE_VALUE, 1), (V_LOOKUP_OUTPUT, -1)]),
    );
    // 18: NextPcEqLookupIfJump. The next-pc rule has four disjoint guards
    //     (Jump, ShouldBranch, Halt, otherwise), one row each: 18 here and
    //     19..=21 in `wasm_next_pc_rows`.
    eq(
        row(&[(V_FLAG_JUMP, 1)]),
        row(&[(V_NEXT_PC, 1), (V_LOOKUP_OUTPUT, -1)]),
    );
    let (mut a2, mut b2, mut c2) = wasm_next_pc_rows::<F>();
    a.append(&mut a2);
    b.append(&mut b2);
    c.append(&mut c2);
    (a, b, c)
}

/// Next-pc rows 19..=21: `ShouldBranch · (NextPc − Imm) = 0`, `Halt · (NextPc
/// − Pc) = 0`, `(1 − ShouldBranch − Jump − Halt) · (NextPc − Pc − 1) = 0`.
fn wasm_next_pc_rows<F: Field>() -> ConstraintRows<F> {
    (
        vec![
            row(&[(V_SHOULD_BRANCH, 1)]),
            row(&[(V_FLAG_HALT, 1)]),
            row(&[
                (V_CONST, 1),
                (V_SHOULD_BRANCH, -1),
                (V_FLAG_JUMP, -1),
                (V_FLAG_HALT, -1),
            ]),
        ],
        vec![
            row(&[(V_NEXT_PC, 1), (V_IMM, -1)]),
            row(&[(V_NEXT_PC, 1), (V_PC, -1)]),
            row(&[(V_NEXT_PC, 1), (V_PC, -1), (V_CONST, -1)]),
        ],
        vec![Vec::new(), Vec::new(), Vec::new()],
    )
}

fn append_product_constraints<F: Field>(
    a: &mut Vec<SparseRow<F>>,
    b: &mut Vec<SparseRow<F>>,
    c: &mut Vec<SparseRow<F>>,
) {
    // Product = LeftInput × RightInput
    a.push(row(&[(V_LEFT_INPUT, 1)]));
    b.push(row(&[(V_RIGHT_INPUT, 1)]));
    c.push(row(&[(V_PRODUCT, 1)]));
    // ShouldBranch = LookupOutput × Branch
    a.push(row(&[(V_LOOKUP_OUTPUT, 1)]));
    b.push(row(&[(V_FLAG_BRANCH, 1)]));
    c.push(row(&[(V_SHOULD_BRANCH, 1)]));
}

/// The equality-conditional rows only (`guard · (left − right) = 0`).
pub fn wasm_spartan_outer_constraints<F: Field>() -> crate::ConstraintMatrices<F> {
    let (a, b, c) = wasm_eq_constraint_rows();
    crate::ConstraintMatrices::new(NUM_EQ_CONSTRAINTS, NUM_VARS_PER_CYCLE, a, b, c)
}

/// The full per-cycle constraint set: the equality-conditional rows followed
/// by the two product rows.
pub fn wasm_trace_constraints<F: Field>() -> crate::ConstraintMatrices<F> {
    let (mut a, mut b, mut c) = wasm_eq_constraint_rows();
    append_product_constraints(&mut a, &mut b, &mut c);
    crate::ConstraintMatrices::new(NUM_CONSTRAINTS_PER_CYCLE, NUM_VARS_PER_CYCLE, a, b, c)
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may unwind via panic")]
#[expect(clippy::indexing_slicing, reason = "tests index fixture data")]
mod tests {
    use super::*;
    use jolt_field::{Fr, Ring};
    use num_traits::Zero;

    /// The `Halt` padding row at pc 0: const = 1, `Halt` set, everything
    /// else zero. Every guard vanishes except the "zero otherwise" rows
    /// (satisfied by zero inputs) and `Halt · (NextPc − Pc)`.
    fn halt_witness() -> Vec<Fr> {
        let mut w = vec![Fr::zero(); NUM_VARS_PER_CYCLE];
        w[V_CONST] = Fr::from_u64(1);
        w[V_FLAG_HALT] = Fr::from_u64(1);
        w
    }

    #[test]
    fn halt_satisfies_constraints() {
        let matrices = wasm_trace_constraints::<Fr>();
        assert_eq!(matrices.num_constraints, NUM_CONSTRAINTS_PER_CYCLE);
        assert_eq!(matrices.num_vars, NUM_VARS_PER_CYCLE);
        matrices
            .check_witness(&halt_witness())
            .expect("halt should satisfy all constraints");
    }

    #[test]
    fn halt_that_advances_the_pc_is_rejected() {
        let matrices = wasm_trace_constraints::<Fr>();
        let mut w = halt_witness();
        w[V_NEXT_PC] = Fr::from_u64(1);
        assert!(matrices.check_witness(&w).is_err());
    }

    #[test]
    fn constraint_count() {
        let matrices = wasm_trace_constraints::<Fr>();
        assert_eq!(matrices.a.len(), NUM_CONSTRAINTS_PER_CYCLE);
        assert_eq!(matrices.b.len(), NUM_CONSTRAINTS_PER_CYCLE);
        assert_eq!(matrices.c.len(), NUM_CONSTRAINTS_PER_CYCLE);
        let outer = wasm_spartan_outer_constraints::<Fr>();
        assert_eq!(outer.num_constraints, NUM_EQ_CONSTRAINTS);
    }
}

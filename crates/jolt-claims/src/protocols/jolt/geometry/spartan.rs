use jolt_field::Ring;
use jolt_wasm_ir::RowFlag;

use crate::derived;

use super::super::{
    JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
    SpartanProductVirtualizationPublic,
};

pub(crate) const OUTER_REMAINDER_DEGREE: usize = 3;
pub(crate) const PRODUCT_REMAINDER_DEGREE: usize = 3;
pub(crate) const SHIFT_DEGREE: usize = 2;

/// The Spartan outer R1CS inputs, in `jolt_r1cs::constraints::wasm` variable
/// order: the sixteen value/lookup/pc columns, then every [`RowFlag`] in bit
/// order.
pub const SPARTAN_OUTER_R1CS_INPUTS: [JoltVirtualPolynomial; 16 + RowFlag::COUNT] = {
    let mut inputs = [JoltVirtualPolynomial::PC; 16 + RowFlag::COUNT];
    let values = [
        JoltVirtualPolynomial::LeftInstructionInput,
        JoltVirtualPolynomial::RightInstructionInput,
        JoltVirtualPolynomial::Product,
        JoltVirtualPolynomial::PC,
        JoltVirtualPolynomial::Imm,
        JoltVirtualPolynomial::RamAddress,
        JoltVirtualPolynomial::Rs1Value,
        JoltVirtualPolynomial::Rs2Value,
        JoltVirtualPolynomial::RdWriteValue,
        JoltVirtualPolynomial::RamReadValue,
        JoltVirtualPolynomial::RamWriteValue,
        JoltVirtualPolynomial::LeftLookupOperand,
        JoltVirtualPolynomial::RightLookupOperand,
        JoltVirtualPolynomial::NextPC,
        JoltVirtualPolynomial::LookupOutput,
        JoltVirtualPolynomial::ShouldBranch,
    ];
    let mut i = 0;
    while i < 16 {
        inputs[i] = values[i];
        i += 1;
    }
    let mut f = 0;
    while f < RowFlag::COUNT {
        inputs[16 + f] = JoltVirtualPolynomial::RowFlag(RowFlag::ALL[f]);
        f += 1;
    }
    inputs
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpartanOuterDimensions {
    log_t: usize,
    variables: Vec<JoltVirtualPolynomial>,
    include_affine_terms: bool,
}

impl SpartanOuterDimensions {
    pub fn new(
        log_t: usize,
        variables: Vec<JoltVirtualPolynomial>,
        include_affine_terms: bool,
    ) -> Option<Self> {
        if variables.is_empty() {
            return None;
        }
        Some(Self {
            log_t,
            variables,
            include_affine_terms,
        })
    }

    pub fn variables(&self) -> &[JoltVirtualPolynomial] {
        &self.variables
    }

    pub fn log_t(&self) -> usize {
        self.log_t
    }

    /// Whether the `Az`/`Bz` linear forms carry their public-column constants
    /// (the affine parts — the source of the expanded form's linear and
    /// constant terms).
    pub fn include_affine_terms(&self) -> bool {
        self.include_affine_terms
    }

    pub const fn remainder_rounds(&self) -> usize {
        1 + self.log_t
    }

    pub fn wasm(log_t: usize) -> Self {
        Self {
            log_t,
            variables: SPARTAN_OUTER_R1CS_INPUTS.to_vec(),
            include_affine_terms: true,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SpartanProductDimensions {
    log_t: usize,
}

impl SpartanProductDimensions {
    pub const fn new(log_t: usize) -> Self {
        Self { log_t }
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }
}

pub fn outer_opening(polynomial: JoltVirtualPolynomial) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(polynomial, JoltRelationId::SpartanOuter)
}

pub fn outer_uniskip_opening() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::UnivariateSkip)
}

pub(crate) fn product_weight<F>(index: usize) -> JoltExpr<F>
where
    F: Ring,
{
    derived(JoltDerivedId::from(
        SpartanProductVirtualizationPublic::LagrangeWeight(index),
    ))
}

pub(crate) fn product_uniskip_weight<F>(index: usize) -> JoltExpr<F>
where
    F: Ring,
{
    derived(JoltDerivedId::from(
        SpartanProductVirtualizationPublic::UniskipLagrangeWeight(index),
    ))
}

pub(crate) fn product_tau_kernel<F>() -> JoltExpr<F>
where
    F: Ring,
{
    derived(JoltDerivedId::from(
        SpartanProductVirtualizationPublic::TauKernel,
    ))
}

pub fn product_uniskip_opening() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::UnivariateSkip,
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub fn product_outer_opening() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::Product)
}

pub fn product_should_branch_outer_opening() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::ShouldBranch)
}

pub fn left_instruction_input_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LeftInstructionInput,
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub fn right_instruction_input_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RightInstructionInput,
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub fn lookup_output_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LookupOutput,
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub fn branch_flag_product() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RowFlag(RowFlag::Branch),
        JoltRelationId::SpartanProductVirtualization,
    )
}
pub(crate) fn next_pc_outer() -> JoltOpeningId {
    outer_opening(JoltVirtualPolynomial::NextPC)
}
pub fn pc_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::PC, JoltRelationId::SpartanShift)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outer_dimensions_rejects_empty_variables() {
        assert_eq!(SpartanOuterDimensions::new(8, Vec::new(), false), None);
    }

    #[test]
    fn default_outer_dimensions_match_r1cs_input_catalog() {
        let dimensions = SpartanOuterDimensions::wasm(8);

        assert_eq!(dimensions.log_t(), 8);
        assert_eq!(dimensions.variables(), &SPARTAN_OUTER_R1CS_INPUTS);
        assert!(dimensions.include_affine_terms());
    }
}

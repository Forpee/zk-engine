//! Operand widths shared by the source and lowered universes.

/// Operand width of an integer operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Width {
    W32,
    W64,
}

impl Width {
    #[inline]
    pub fn bits(self) -> u32 {
        match self {
            Width::W32 => 32,
            Width::W64 => 64,
        }
    }
}

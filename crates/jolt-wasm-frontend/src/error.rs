/// Errors raised while decoding and validating a WebAssembly module into the
/// source representation.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("malformed or invalid wasm: {0}")]
    Wasm(#[from] wasmparser::BinaryReaderError),
    /// A WebAssembly operator outside the supported integer core.
    #[error("unsupported wasm operator `{0}`")]
    UnsupportedOperator(String),
    #[error("unsupported wasm feature: {0}")]
    Unsupported(&'static str),
    #[error("value type {0:?} is not supported (integer core only)")]
    UnsupportedValType(wasmparser::ValType),
    #[error("constant expression must be a single `i32.const`/`i64.const`")]
    UnsupportedConstExpr,
    #[error("module imports `{module}.{name}`; imports are not supported")]
    Import { module: String, name: String },
    #[error("data segment {0} is passive; only active segments are supported")]
    PassiveData(u32),
    #[error("data segment {index} targets memory {memory}; only memory 0 exists")]
    DataMemoryIndex { index: u32, memory: u32 },
    #[error("function {0} has no type")]
    MissingFunctionType(u32),
    #[error("type index {0} is out of range")]
    TypeIndex(u32),
    #[error("global index {0} is out of range")]
    GlobalIndex(u32),
    #[error("data segment {index} ({len} bytes at {offset}) exceeds the initial memory")]
    DataOutOfBounds { index: u32, offset: u64, len: usize },
    #[error("element segment {0} is passive or declarative; only active segments are supported")]
    PassiveElement(u32),
    #[error("element segment {index} targets table {table}; only table 0 exists")]
    ElementTableIndex { index: u32, table: u32 },
    #[error("element segment {index} ({len} slots at {offset}) exceeds the table")]
    ElementOutOfBounds { index: u32, offset: u64, len: usize },
}

/// Errors raised while lowering a decoded module to the register-machine IR.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum LowerError {
    /// Locals plus maximum operand-stack height exceed the register frame.
    #[error("function {function} needs {slots} frame slots; at most {max} are available")]
    FrameTooLarge {
        function: u32,
        slots: usize,
        max: usize,
    },
    #[error("function {function} returns {results} values; at most {max} are supported")]
    TooManyResults {
        function: u32,
        results: usize,
        max: usize,
    },
    #[error("lowered program exceeds the {0}-bit program counter")]
    ProgramTooLarge(u32),
    #[error("function {0} is out of range")]
    FunctionIndex(u32),
    #[error("label depth {0} is out of range")]
    LabelDepth(u32),
    #[error("control-flow structure of function {0} is malformed")]
    MalformedControl(u32),
    #[error("linear memory declares {pages} initial pages; at most {max} are supported")]
    MemoryTooLarge { pages: u64, max: u64 },
    #[error("the function table has {slots} slots; at most {max} are supported")]
    TableTooLarge { slots: u64, max: u64 },
}

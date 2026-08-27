//! Decoded and validated WebAssembly module in source form.
//!
//! Decoding runs `wasmparser`'s validator alongside the parser so that every
//! operator is annotated with the operand-stack height *before* it executes;
//! that static height is what the lowering uses to assign registers.

use std::collections::BTreeMap;

use wasmparser::{
    BlockType, CompositeInnerType, ConstExpr, DataKind, ElementItems, ElementKind, ExternalKind,
    FuncValidatorAllocations, Operator, Parser, Payload, ValidPayload, Validator,
};

use crate::error::DecodeError;
use jolt_wasm_ir::DataSegment;

use crate::source::{val_type, BlockSig, BlockTypes, ValType, WasmOp};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncType {
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
}

/// One source operator with the operand-stack height before it executes
/// (whole-function height, locals excluded).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceOp {
    pub op: WasmOp,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub type_index: u32,
    /// Declared locals, excluding parameters.
    pub locals: Vec<ValType>,
    pub body: Vec<SourceOp>,
    /// Maximum operand-stack height reached anywhere in the body.
    pub max_height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Global {
    pub ty: ValType,
    pub mutable: bool,
    pub init: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryDecl {
    pub initial_pages: u64,
    pub max_pages: Option<u64>,
}

/// The module's single `funcref` table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableDecl {
    pub initial: u64,
    pub maximum: Option<u64>,
}

/// An active element segment: function indices (`None` for `ref.null`)
/// written to table 0 from `offset`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementSegment {
    pub offset: u64,
    pub functions: Vec<Option<u32>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WasmModule {
    pub types: Vec<FuncType>,
    pub functions: Vec<Function>,
    pub globals: Vec<Global>,
    pub memory: Option<MemoryDecl>,
    pub data: Vec<DataSegment>,
    pub table: Option<TableDecl>,
    pub elements: Vec<ElementSegment>,
    /// Exported functions by name.
    pub exports: BTreeMap<String, u32>,
    pub start: Option<u32>,
}

struct TypeTable<'a>(&'a [FuncType]);

impl BlockTypes for TypeTable<'_> {
    fn block_sig(&self, ty: BlockType) -> Result<BlockSig, DecodeError> {
        Ok(match ty {
            BlockType::Empty => BlockSig {
                params: 0,
                results: 0,
            },
            BlockType::Type(_) => BlockSig {
                params: 0,
                results: 1,
            },
            BlockType::FuncType(index) => {
                let ft = self
                    .0
                    .get(index as usize)
                    .ok_or(DecodeError::TypeIndex(index))?;
                BlockSig {
                    params: ft.params.len() as u32,
                    results: ft.results.len() as u32,
                }
            }
        })
    }
}

impl WasmModule {
    /// Parse, validate, and decode a binary module into the supported
    /// integer core.
    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        let mut module = WasmModule::default();
        let mut validator = Validator::new();
        let mut allocs = FuncValidatorAllocations::default();
        let mut function_types: Vec<u32> = Vec::new();

        for payload in Parser::new(0).parse_all(bytes) {
            let payload = payload?;
            let valid = validator.payload(&payload)?;
            match payload {
                Payload::TypeSection(reader) => {
                    for group in reader {
                        for sub in group?.types() {
                            let CompositeInnerType::Func(ft) = &sub.composite_type.inner else {
                                return Err(DecodeError::Unsupported("gc types"));
                            };
                            module.types.push(FuncType {
                                params: convert_types(ft.params())?,
                                results: convert_types(ft.results())?,
                            });
                        }
                    }
                }
                Payload::ImportSection(reader) => {
                    if let Some(group) = reader.into_iter().next() {
                        let Some(item) = group?.into_iter().next() else {
                            continue;
                        };
                        let (_, import) = item?;
                        return Err(DecodeError::Import {
                            module: import.module.to_owned(),
                            name: import.name.to_owned(),
                        });
                    }
                }
                Payload::FunctionSection(reader) => {
                    for ty in reader {
                        function_types.push(ty?);
                    }
                }
                Payload::TableSection(reader) => {
                    for table in reader {
                        let table = table?;
                        if module.table.is_some() {
                            return Err(DecodeError::Unsupported("multiple tables"));
                        }
                        if !table.ty.element_type.is_func_ref() || table.ty.table64 {
                            return Err(DecodeError::Unsupported("non-funcref or 64-bit tables"));
                        }
                        if !matches!(table.init, wasmparser::TableInit::RefNull) {
                            return Err(DecodeError::Unsupported("table initializer expressions"));
                        }
                        module.table = Some(TableDecl {
                            initial: table.ty.initial,
                            maximum: table.ty.maximum,
                        });
                    }
                }
                Payload::ElementSection(reader) => {
                    for (index, element) in reader.into_iter().enumerate() {
                        let element = element?;
                        let index = index as u32;
                        let ElementKind::Active {
                            table_index,
                            offset_expr,
                        } = element.kind
                        else {
                            return Err(DecodeError::PassiveElement(index));
                        };
                        let table = table_index.unwrap_or(0);
                        if table != 0 {
                            return Err(DecodeError::ElementTableIndex { index, table });
                        }
                        let functions = match element.items {
                            ElementItems::Functions(items) => items
                                .into_iter()
                                .map(|f| f.map(Some))
                                .collect::<Result<Vec<_>, _>>()?,
                            ElementItems::Expressions(_, items) => items
                                .into_iter()
                                .map(|expr| func_ref_expr(&expr?))
                                .collect::<Result<Vec<_>, _>>()?,
                        };
                        module.elements.push(ElementSegment {
                            offset: const_expr(&offset_expr)?,
                            functions,
                        });
                    }
                }
                Payload::TagSection(_) => return Err(DecodeError::Unsupported("exceptions")),
                Payload::MemorySection(reader) => {
                    for mem in reader {
                        let mem = mem?;
                        if module.memory.is_some() {
                            return Err(DecodeError::Unsupported("multi-memory"));
                        }
                        if mem.memory64 || mem.shared {
                            return Err(DecodeError::Unsupported("memory64/shared memory"));
                        }
                        module.memory = Some(MemoryDecl {
                            initial_pages: mem.initial,
                            max_pages: mem.maximum,
                        });
                    }
                }
                Payload::GlobalSection(reader) => {
                    for global in reader {
                        let global = global?;
                        module.globals.push(Global {
                            ty: val_type(global.ty.content_type)?,
                            mutable: global.ty.mutable,
                            init: const_expr(&global.init_expr)?,
                        });
                    }
                }
                Payload::ExportSection(reader) => {
                    for export in reader {
                        let export = export?;
                        if matches!(export.kind, ExternalKind::Func | ExternalKind::FuncExact) {
                            let _ = module.exports.insert(export.name.to_owned(), export.index);
                        }
                    }
                }
                Payload::StartSection { func, .. } => module.start = Some(func),
                Payload::DataSection(reader) => {
                    for (index, data) in reader.into_iter().enumerate() {
                        let data = data?;
                        let index = index as u32;
                        let DataKind::Active {
                            memory_index,
                            offset_expr,
                        } = data.kind
                        else {
                            return Err(DecodeError::PassiveData(index));
                        };
                        if memory_index != 0 {
                            return Err(DecodeError::DataMemoryIndex {
                                index,
                                memory: memory_index,
                            });
                        }
                        module.data.push(DataSegment {
                            offset: const_expr(&offset_expr)?,
                            bytes: data.data.to_vec(),
                        });
                    }
                }
                Payload::CodeSectionEntry(_) => {
                    let ValidPayload::Func(func, body) = valid else {
                        return Err(DecodeError::Unsupported("code entry without validator"));
                    };
                    let index = module.functions.len() as u32;
                    let type_index = *function_types
                        .get(index as usize)
                        .ok_or(DecodeError::MissingFunctionType(index))?;
                    let mut fv = func.into_validator(std::mem::take(&mut allocs));

                    let mut ops_reader = body.get_operators_reader()?;
                    let locals_end = ops_reader.original_position();
                    let mut locals = Vec::new();
                    for local in body.get_locals_reader()? {
                        let (count, ty) = local?;
                        fv.define_locals(locals_end, count, ty)?;
                        let ty = val_type(ty)?;
                        locals.extend(std::iter::repeat_n(ty, count as usize));
                    }

                    let types = TypeTable(&module.types);
                    let mut ops = Vec::new();
                    let mut max_height = 0;
                    while !ops_reader.eof() {
                        let (op, offset) = ops_reader.read_with_offset()?;
                        let height = fv.operand_stack_height();
                        let decoded = WasmOp::decode(&op, &types)?;
                        fv.op(offset, &op)?;
                        max_height = max_height.max(fv.operand_stack_height());
                        ops.push(SourceOp {
                            op: decoded,
                            height,
                        });
                    }
                    ops_reader.finish()?;
                    allocs = fv.into_allocations();

                    module.functions.push(Function {
                        type_index,
                        locals,
                        body: ops,
                        max_height,
                    });
                }
                _ => {}
            }
        }

        for (index, segment) in module.elements.iter().enumerate() {
            let slots = module.table.map_or(0, |t| t.initial);
            let end = segment.offset + segment.functions.len() as u64;
            if end > slots {
                return Err(DecodeError::ElementOutOfBounds {
                    index: index as u32,
                    offset: segment.offset,
                    len: segment.functions.len(),
                });
            }
        }
        if let Some(memory) = module.memory {
            for (index, segment) in module.data.iter().enumerate() {
                let end = segment.offset + segment.bytes.len() as u64;
                if end > memory.initial_pages * jolt_wasm_ir::layout::PAGE_SIZE {
                    return Err(DecodeError::DataOutOfBounds {
                        index: index as u32,
                        offset: segment.offset,
                        len: segment.bytes.len(),
                    });
                }
            }
        }
        Ok(module)
    }

    pub fn func_type(&self, function: u32) -> Result<&FuncType, DecodeError> {
        let f = self
            .functions
            .get(function as usize)
            .ok_or(DecodeError::MissingFunctionType(function))?;
        self.types
            .get(f.type_index as usize)
            .ok_or(DecodeError::TypeIndex(f.type_index))
    }
}

fn convert_types(types: &[wasmparser::ValType]) -> Result<Vec<ValType>, DecodeError> {
    types.iter().map(|ty| val_type(*ty)).collect()
}

/// Evaluate an element expression of the form `ref.func f` / `ref.null` `end`.
fn func_ref_expr(expr: &ConstExpr<'_>) -> Result<Option<u32>, DecodeError> {
    let mut reader = expr.get_operators_reader();
    let function = match reader.read()? {
        Operator::RefFunc { function_index } => Some(function_index),
        Operator::RefNull { .. } => None,
        _ => return Err(DecodeError::UnsupportedConstExpr),
    };
    match reader.read()? {
        Operator::End => Ok(function),
        _ => Err(DecodeError::UnsupportedConstExpr),
    }
}

/// Evaluate a constant expression of the form `i32.const`/`i64.const` `end`.
fn const_expr(expr: &ConstExpr<'_>) -> Result<u64, DecodeError> {
    let mut reader = expr.get_operators_reader();
    let value = match reader.read()? {
        Operator::I32Const { value } => u64::from(value as u32),
        Operator::I64Const { value } => value as u64,
        _ => return Err(DecodeError::UnsupportedConstExpr),
    };
    match reader.read()? {
        Operator::End => Ok(value),
        _ => Err(DecodeError::UnsupportedConstExpr),
    }
}

//! Reference interpreter for the IR, emitting one [`Record`] per executed
//! instruction. Each step executes the instruction's [`RowSpec`] — reads,
//! lookup, flags — so the recorded witness is the row model by construction;
//! [`crate::row::check_record`] is the constraint-form restatement.

use jolt_wasm_ir::layout::{input_address, output_address, TERMINATION_ADDR};
use jolt_wasm_ir::{AssertFailure, Ir, IrProgram, Pc, Reg, REGISTER_COUNT};

use crate::error::{ExecutionError, Trap};
use crate::memory::Memory;
use jolt_wasm_ir::row::{Lookup, RowFlag, RowModel, RowSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterRead {
    pub register: Reg,
    pub value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterWrite {
    pub register: Reg,
    pub pre_value: u64,
    pub post_value: u64,
}

/// One aligned 64-bit word read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RamRead {
    pub address: u64,
    pub value: u64,
}

/// One aligned 64-bit word write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RamWrite {
    pub address: u64,
    pub pre_value: u64,
    pub post_value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RamAccess {
    Read(RamRead),
    Write(RamWrite),
    #[default]
    NoOp,
}

/// The execution record of one IR instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record {
    pub pc: Pc,
    pub next_pc: Pc,
    pub instruction: Ir,
    pub rs1: Option<RegisterRead>,
    pub rs2: Option<RegisterRead>,
    pub rd: Option<RegisterWrite>,
    pub ram: RamAccess,
}

/// A completed run of an exported function.
#[derive(Debug, Clone)]
pub struct Execution {
    pub records: Vec<Record>,
    /// The public output words.
    pub results: Vec<u64>,
    /// Whether the entry stub set the termination word (always true for a
    /// run that returned `Ok`; the field is the public-output view of it).
    pub terminated: bool,
    pub memory: Memory,
}

pub const DEFAULT_STEP_LIMIT: u64 = 1 << 32;

#[derive(Debug, Clone)]
pub struct Machine<'a> {
    program: &'a IrProgram,
    regs: [u64; REGISTER_COUNT],
    pc: Pc,
    memory: Memory,
    step_limit: u64,
}

impl<'a> Machine<'a> {
    pub fn new(program: &'a IrProgram) -> Result<Self, ExecutionError> {
        let mut memory = Memory::new(program.memory, &program.globals, &program.table);
        for segment in &program.data {
            memory
                .init_linear(segment.offset, &segment.bytes)
                .map_err(|trap| ExecutionError::Trap {
                    pc: IrProgram::HALT_PC,
                    trap,
                })?;
        }
        Ok(Self {
            program,
            regs: [0; REGISTER_COUNT],
            pc: IrProgram::HALT_PC,
            memory,
            step_limit: DEFAULT_STEP_LIMIT,
        })
    }

    pub fn with_step_limit(mut self, limit: u64) -> Self {
        self.step_limit = limit;
        self
    }

    /// Run the exported function `name` with `args` through its entry stub,
    /// recording every executed instruction as one contiguous trace.
    ///
    /// The host's only actions are outside the trace: it writes `args` to
    /// the public input words before execution and reads the results from
    /// the public output words after. Registers start at zero; the stub sets
    /// `SP`, runs `start`, calls the function, stores the results, sets the
    /// termination word, and jumps to the [`Ir::Halt`] trampoline at pc 0.
    pub fn invoke(mut self, name: &str, args: &[u64]) -> Result<Execution, ExecutionError> {
        let function = *self
            .program
            .exports
            .get(name)
            .ok_or_else(|| ExecutionError::UnknownExport(name.to_owned()))?;
        let entry = *self
            .program
            .entries
            .get(name)
            .ok_or_else(|| ExecutionError::UnknownExport(name.to_owned()))?;
        let f = self
            .program
            .functions
            .get(function as usize)
            .ok_or_else(|| ExecutionError::UnknownExport(name.to_owned()))?;
        let params = f.params as usize;
        let results = *self
            .program
            .output_words
            .get(name)
            .ok_or_else(|| ExecutionError::UnknownExport(name.to_owned()))?
            as usize;
        if args.len() != params {
            return Err(ExecutionError::ArgumentCount {
                name: name.to_owned(),
                expected: params,
                actual: args.len(),
            });
        }
        let host = |trap| ExecutionError::Trap {
            pc: IrProgram::HALT_PC,
            trap,
        };
        for (i, arg) in args.iter().enumerate() {
            let _ = self
                .memory
                .write_word(input_address(i as u64), *arg)
                .map_err(host)?;
        }
        self.regs = [0; REGISTER_COUNT];
        self.pc = entry;
        let mut records = Vec::new();
        self.run(&mut records)?;
        let results = (0..results as u64)
            .map(|k| self.memory.read_word(output_address(k)).map_err(host))
            .collect::<Result<Vec<u64>, _>>()?;
        let terminated = self.memory.read_word(TERMINATION_ADDR).map_err(host)? == 1;
        Ok(Execution {
            records,
            results,
            terminated,
            memory: self.memory,
        })
    }

    fn run(&mut self, records: &mut Vec<Record>) -> Result<(), ExecutionError> {
        let mut steps = 0u64;
        loop {
            if steps >= self.step_limit {
                return Err(ExecutionError::StepLimit(self.step_limit));
            }
            steps += 1;
            let pc = self.pc;
            let record = self
                .step()
                .map_err(|trap| ExecutionError::Trap { pc, trap })?;
            let halted = matches!(record.instruction, Ir::Halt);
            records.push(record);
            if halted {
                return Ok(());
            }
        }
    }

    #[inline]
    fn read(&self, reg: Reg) -> RegisterRead {
        RegisterRead {
            register: reg,
            value: self.regs[reg.index()],
        }
    }

    /// Writes `rd` (writes to `ZERO` are dropped, keeping it hardwired).
    #[inline]
    fn write(&mut self, rd: Reg, value: u64) -> RegisterWrite {
        let pre_value = self.regs[rd.index()];
        let post_value = if rd == Reg::ZERO { 0 } else { value };
        self.regs[rd.index()] = post_value;
        RegisterWrite {
            register: rd,
            pre_value,
            post_value,
        }
    }

    /// Execute one row: instruction inputs → lookup → flag-guarded effects.
    fn step(&mut self) -> Result<Record, Trap> {
        let pc = self.pc;
        let instruction = *self
            .program
            .code
            .get(pc as usize)
            .ok_or(Trap::InvalidJump(u64::from(pc)))?;
        let spec: RowSpec = instruction.row_spec();
        let flags = spec.flags;
        if flags.has(RowFlag::Trap) {
            return Err(Trap::Unreachable);
        }

        let rs1 = spec.rs1.map(|r| self.read(r));
        let rs2 = spec.rs2.map(|r| self.read(r));
        let rs1_value = rs1.map_or(0, |r| r.value);
        let rs2_value = rs2.map_or(0, |r| r.value);
        let left = spec.left_input(rs1_value);
        let right = spec.right_input(rs2_value);
        let output = match spec.lookup {
            Some(Lookup::Table(op)) => op.evaluate(left, right),
            // Advice is the honest prover's computation from the register
            // reads; the row's instruction inputs are zero by design.
            Some(Lookup::Advice(hint)) => hint.compute(rs1_value, rs2_value),
            None => 0,
        };
        if flags.has(RowFlag::Assert) && output != 1 {
            return Err(assert_trap(instruction, rs1_value));
        }

        let mut ram = RamAccess::NoOp;
        let mut rd_value = None;
        let address = rs1_value.wrapping_add(spec.imm);
        if flags.has(RowFlag::Load) {
            let value = self.memory.read_word(address)?;
            ram = RamAccess::Read(RamRead { address, value });
            rd_value = Some(value);
        } else if flags.has(RowFlag::Store) {
            let pre_value = self.memory.write_word(address, rs2_value)?;
            ram = RamAccess::Write(RamWrite {
                address,
                pre_value,
                post_value: rs2_value,
            });
        } else if flags.intersects(RowFlag::WriteLookupToRd | RowFlag::Advice) {
            rd_value = Some(output);
        }
        let rd = match (spec.rd, rd_value) {
            (Some(reg), Some(value)) => Some(self.write(reg, value)),
            _ => None,
        };

        let next_pc = if flags.has(RowFlag::Halt) {
            pc
        } else if flags.has(RowFlag::Jump) {
            Pc::try_from(output).map_err(|_| Trap::InvalidJump(output))?
        } else if flags.has(RowFlag::Branch) && output == 1 {
            spec.imm as Pc
        } else {
            pc + 1
        };
        self.pc = next_pc;
        Ok(Record {
            pc,
            next_pc,
            instruction,
            rs1,
            rs2,
            rd,
            ram,
        })
    }
}

/// The guest-visible trap for a failed assert row.
fn assert_trap(instruction: Ir, rs1_value: u64) -> Trap {
    match instruction {
        Ir::Assert { failure, .. } => match failure {
            AssertFailure::OutOfBounds(bytes) => Trap::OutOfBoundsMemory {
                address: rs1_value,
                width: bytes,
            },
            AssertFailure::DivideByZero => Trap::DivideByZero,
            AssertFailure::IntegerOverflow => Trap::IntegerOverflow,
            AssertFailure::TableOutOfBounds => Trap::TableOutOfBounds,
            AssertFailure::IndirectCallTypeMismatch => Trap::IndirectCallTypeMismatch,
        },
        _ => Trap::Unreachable,
    }
}

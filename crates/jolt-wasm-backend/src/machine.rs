//! Reference interpreter for the IR, emitting one [`Record`] per executed
//! instruction. Each step executes the instruction's [`RowSpec`] — reads,
//! lookup, flags — so the recorded witness is the row model by construction;
//! [`crate::row::check_record`] is the constraint-form restatement.

use jolt_wasm_ir::layout::{MEMORY_SIZE_ADDR, SHADOW_STACK_BASE};
use jolt_wasm_ir::{AssertFailure, Ir, IrProgram, Pc, Reg, REGISTER_COUNT};

use crate::error::{ExecutionError, Trap};
use crate::memory::Memory;
use crate::row::{Lookup, RowFlags, RowModel, RowSpec};

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
    pub results: Vec<u64>,
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
        let mut memory = Memory::new(program.memory, &program.globals);
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

    /// Run the module's start function (if any), then the exported function
    /// `name` with `args`, recording every executed instruction.
    ///
    /// Each host-level call is one *segment* of the record stream: it begins
    /// with the registers initialized by [`Machine::call`] and ends with the
    /// [`Ir::Halt`] record the callee returns to. The pc chains within a
    /// segment; the start-function segment (when present) precedes the entry
    /// segment.
    pub fn invoke(mut self, name: &str, args: &[u64]) -> Result<Execution, ExecutionError> {
        let function = *self
            .program
            .exports
            .get(name)
            .ok_or_else(|| ExecutionError::UnknownExport(name.to_owned()))?;
        let mut records = Vec::new();
        if let Some(start) = self.program.start {
            let _ = self.call(start, &[], &mut records)?;
        }
        let expected = self.function_params(function);
        if args.len() != expected {
            return Err(ExecutionError::ArgumentCount {
                name: name.to_owned(),
                expected,
                actual: args.len(),
            });
        }
        let results = self.call(function, args, &mut records)?;
        Ok(Execution {
            records,
            results,
            memory: self.memory,
        })
    }

    fn function_params(&self, function: u32) -> usize {
        self.program
            .functions
            .get(function as usize)
            .map_or(0, |f| f.params as usize)
    }

    /// Host-side call: mimics the lowered call sequence's post-state (return
    /// address on the shadow stack, `SP` past it, parameters in frame slots)
    /// so the callee's `return` lands on the halt trampoline.
    fn call(
        &mut self,
        function: u32,
        args: &[u64],
        records: &mut Vec<Record>,
    ) -> Result<Vec<u64>, ExecutionError> {
        let trap = |trap| ExecutionError::Trap {
            pc: IrProgram::HALT_PC,
            trap,
        };
        let f = self
            .program
            .functions
            .get(function as usize)
            .ok_or_else(|| trap(Trap::InvalidJump(u64::from(function))))?;
        self.regs = [0; REGISTER_COUNT];
        let _ = self
            .memory
            .write_word(SHADOW_STACK_BASE, u64::from(IrProgram::HALT_PC))
            .map_err(trap)?;
        self.regs[Reg::SP.index()] = SHADOW_STACK_BASE + 8;
        for (i, arg) in args.iter().enumerate() {
            let reg = Reg::frame_slot(i).ok_or_else(|| trap(Trap::CallStackExhausted))?;
            self.regs[reg.index()] = *arg;
        }
        self.pc = f.entry;
        let results = f.results as usize;
        self.run(records)?;
        Ok((0..results)
            .filter_map(Reg::temp)
            .map(|r| self.regs[r.index()])
            .collect())
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
        if flags.has(RowFlags::TRAP) {
            return Err(Trap::Unreachable);
        }

        let rs1 = spec.rs1.map(|r| self.read(r));
        let rs2 = spec.rs2.map(|r| self.read(r));
        let rs1_value = rs1.map_or(0, |r| r.value);
        let rs2_value = rs2.map_or(0, |r| r.value);
        let left = spec.left_input(rs1_value, pc);
        let right = spec.right_input(rs2_value);
        let output = match spec.lookup {
            Some(Lookup::Table(op)) => op.evaluate(left, right),
            Some(Lookup::Advice(hint)) => hint.compute(left, right),
            None => 0,
        };
        if flags.has(RowFlags::ASSERT) && output != 1 {
            return Err(assert_trap(instruction, rs1_value));
        }

        let mut ram = RamAccess::NoOp;
        let mut rd_value = None;
        let address = rs1_value.wrapping_add(spec.imm);
        if flags.has(RowFlags::LOAD) {
            let value = self.memory.read_word(address)?;
            ram = RamAccess::Read(RamRead { address, value });
            rd_value = Some(value);
        } else if flags.has(RowFlags::STORE) {
            let pre_value = self.memory.write_word(address, rs2_value)?;
            ram = RamAccess::Write(RamWrite {
                address,
                pre_value,
                post_value: rs2_value,
            });
        } else if flags.has(RowFlags::MEMORY_GROW) {
            let old_bytes = self.memory.size_bytes();
            let (result, new_bytes) = match self.memory.grow(rs1_value) {
                Some(grown) => (grown.old_pages, grown.new_bytes),
                None => (u64::from(u32::MAX), old_bytes),
            };
            ram = RamAccess::Write(RamWrite {
                address: MEMORY_SIZE_ADDR,
                pre_value: old_bytes,
                post_value: new_bytes,
            });
            rd_value = Some(result);
        } else if flags.has(RowFlags::WRITE_LOOKUP_TO_RD | RowFlags::ADVICE) {
            rd_value = Some(output);
        }
        let rd = match (spec.rd, rd_value) {
            (Some(reg), Some(value)) => Some(self.write(reg, value)),
            _ => None,
        };

        let next_pc = if flags.has(RowFlags::HALT) {
            pc
        } else if flags.has(RowFlags::JUMP) {
            Pc::try_from(output).map_err(|_| Trap::InvalidJump(output))?
        } else if flags.has(RowFlags::BRANCH) && output == 1 {
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
        },
        _ => Trap::Unreachable,
    }
}

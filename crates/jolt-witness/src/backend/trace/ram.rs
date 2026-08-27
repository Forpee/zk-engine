//! RAM virtual polynomials and memory-state reconstruction over the dense
//! word index (`jolt_wasm_ir::layout::remap_word_address`).

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use jolt_wasm_ir::layout::remap_word_address;

use super::*;

impl TraceBackend {
    pub(crate) fn materialize_ram_read_write_virtual<F: JoltField>(
        &self,
        id: JoltVirtualPolynomial,
    ) -> Result<Vec<F>, WitnessError> {
        match id {
            JoltVirtualPolynomial::RamVal => self.materialize_ram_val(),
            JoltVirtualPolynomial::RamRa => self.materialize_ram_ra(),
            _ => Err(WitnessError::UnknownOracle {
                label: JOLT_VM_LABEL,
            }),
        }
    }

    pub(crate) fn materialize_ram_val<F: JoltField>(&self) -> Result<Vec<F>, WitnessError> {
        let cycles = checked_pow2(self.config.log_t)?;
        let addresses = self.config.ram_k;
        let mut state = self.initial_ram_state()?;
        let mut values = jolt_utils::unsafe_allocate_zero_vec(addresses * cycles);
        for cycle in 0..cycles {
            for (address, value) in state.iter().copied().enumerate() {
                values[address * cycles + cycle] = F::from_u64(value);
            }
            let Some(row) = self.rows.get(cycle) else {
                continue;
            };
            if row.is_load() {
                let address = self.remapped_ram_address(row.ram_address())?;
                values[address * cycles + cycle] = F::from_u64(row.ram_read_value());
            } else if row.is_store() {
                let address = self.remapped_ram_address(row.ram_address())?;
                values[address * cycles + cycle] = F::from_u64(row.ram_read_value());
                state[address] = row.ram_write_value();
            }
        }
        Ok(values)
    }

    pub(crate) fn materialize_ram_ra<F: JoltField>(&self) -> Result<Vec<F>, WitnessError> {
        let cycles = checked_pow2(self.config.log_t)?;
        let addresses = self.config.ram_k;
        let mut values = jolt_utils::unsafe_allocate_zero_vec(addresses * cycles);
        for cycle in 0..cycles {
            let Some(row) = self.rows.get(cycle) else {
                continue;
            };
            if let Some(raw_address) = ram_access_address(row) {
                let address = self.remapped_ram_address(raw_address)?;
                values[address * cycles + cycle] = F::one();
            }
        }
        Ok(values)
    }

    pub(crate) fn materialize_ram_val_final<F: JoltField>(&self) -> Result<Vec<F>, WitnessError> {
        #[cfg(feature = "parallel")]
        return self
            .final_ram_state()
            .map(|state| state.into_par_iter().map(F::from_u64).collect());
        #[cfg(not(feature = "parallel"))]
        self.final_ram_state()
            .map(|state| state.into_iter().map(F::from_u64).collect())
    }

    /// The RAM word index of a guest address, bounded by `ram_k`.
    fn remapped_ram_address(&self, address: u64) -> Result<usize, WitnessError> {
        let index =
            remap_word_address(address).ok_or_else(|| WitnessError::InvalidWitnessData {
                label: JOLT_VM_LABEL,
                reason: format!("RAM address {address:#x} is outside the RAM window or misaligned"),
            })?;
        let index = usize::try_from(index)
            .ok()
            .filter(|index| *index < self.config.ram_k);
        index.ok_or_else(|| WitnessError::InvalidWitnessData {
            label: JOLT_VM_LABEL,
            reason: format!(
                "RAM address {address:#x} exceeds the {}-word RAM domain",
                self.config.ram_k
            ),
        })
    }

    /// The initial RAM state: the program image plus the run's inputs.
    pub(crate) fn initial_ram_state(&self) -> Result<Vec<u64>, WitnessError> {
        if self.config.ram_k == 0 {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: "ram_k must be nonzero".to_owned(),
            });
        }
        let mut state = vec![0; self.config.ram_k];
        for word in self.preprocessing.initial_memory(&self.io.inputs) {
            let index = self.remapped_ram_address(word.address)?;
            state[index] = word.value;
        }
        Ok(state)
    }

    /// The final RAM state: the initial state with every store of the trace
    /// replayed in order (the padding rows store nothing).
    pub(crate) fn final_ram_state(&self) -> Result<Vec<u64>, WitnessError> {
        let mut state = self.initial_ram_state()?;
        for row in self.rows.iter() {
            if row.is_store() {
                let address = self.remapped_ram_address(row.ram_address())?;
                state[address] = row.ram_write_value();
            }
        }
        Ok(state)
    }
}

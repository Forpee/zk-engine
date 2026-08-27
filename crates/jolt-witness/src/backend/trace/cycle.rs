//! The sequential cycle walk driving the atomic extractors, and the
//! trace-backed implementation of the streaming pass.

use std::ops::Range;

use super::*;
use crate::consumer::ChunkVisitor;
use crate::witnesses::{Extract, ExtractIndexed, RaChunkSelector, ToField, WitnessEnv};
use crate::{BundleSource, RandomAccessRows, RowSource, WitnessBundle};

impl TraceBackend {
    /// Materializes one cycle-domain witness column by walking the trace
    /// once; all per-witness logic lives on `W`.
    pub(crate) fn materialize_cycle<F: JoltField, W: Extract + ToField>(
        &self,
    ) -> Result<Vec<F>, WitnessError> {
        self.walk_cycles(|row, next, env| W::extract(row, next, env).map(ToField::to_field))
    }

    /// [`Self::materialize_cycle`] for indexed witness families; `index`
    /// selects the family member.
    pub(crate) fn materialize_cycle_indexed<
        F: JoltField,
        W: ExtractIndexed<I> + ToField,
        I: Copy + Send + Sync,
    >(
        &self,
        index: I,
    ) -> Result<Vec<F>, WitnessError> {
        self.walk_cycles(|row, next, env| {
            W::extract_indexed(index, row, next, env).map(ToField::to_field)
        })
    }

    /// Materializes one member of a one-hot RA decomposition as the flat
    /// address-major `(K x T)` grid, `K = 2^chunk_bits`: one cycle walk
    /// collecting the per-cycle hot addresses (`None` is a cold cycle),
    /// then a scatter of ones.
    ///
    /// The walk's padding (`Halt`) rows coincide with the one-hot conventions
    /// by construction: a padding row's lookup index is 0 and its pc is 0,
    /// so instruction/bytecode grids pad to the address-0 chunk and RAM
    /// grids to cold cycles.
    pub(crate) fn materialize_one_hot<F, W>(
        &self,
        index: usize,
        chunks: usize,
        chunk_bits: usize,
    ) -> Result<Vec<F>, WitnessError>
    where
        F: JoltField,
        W: ExtractIndexed<RaChunkSelector> + Into<Option<usize>>,
    {
        let selector = RaChunkSelector::new(index, chunks, chunk_bits)?;
        let cycles = checked_pow2(self.config.log_t)?;
        let log_rows = chunk_bits.checked_add(self.config.log_t).ok_or_else(|| {
            WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: "one-hot rows overflow".to_owned(),
            }
        })?;
        let hot_addresses: Vec<Option<usize>> = self.walk_cycles(|row, next, env| {
            W::extract_indexed(selector, row, next, env).map(W::into)
        })?;
        // The selector's mask bounds every hot address below `2^chunk_bits`.
        let mut values = jolt_utils::unsafe_allocate_zero_vec(checked_pow2(log_rows)?);
        for (cycle, address) in hot_addresses.into_iter().enumerate() {
            if let Some(address) = address {
                values[address * cycles + cycle] = F::one();
            }
        }
        Ok(values)
    }

    /// One pass over the padded cycle domain with one-row lookahead.
    fn walk_cycles<V, R>(&self, extract: V) -> Result<Vec<R>, WitnessError>
    where
        V: Fn(&TraceRow, Option<&TraceRow>, &WitnessEnv<'_>) -> Result<R, WitnessError>
            + Sync
            + Send,
        R: Send,
    {
        let access = self.random_access_rows()?;
        let cycles = access.cycles();
        let env = WitnessEnv::new(&self.preprocessing);
        let padding = TraceRow::default();
        let row_at = |index: usize| self.rows.get(index).unwrap_or(&padding);
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            (0..cycles)
                .into_par_iter()
                .map(|index| {
                    let next = (index + 1 < cycles).then(|| row_at(index + 1));
                    extract(row_at(index), next, &env)
                })
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..cycles)
                .map(|index| {
                    let next = (index + 1 < cycles).then(|| row_at(index + 1));
                    extract(row_at(index), next, &env)
                })
                .collect()
        }
    }

    fn random_access_rows(&self) -> Result<RandomAccessRows, WitnessError> {
        RandomAccessRows::new(
            Arc::clone(&self.rows),
            checked_pow2(self.config.log_t)?,
            Arc::clone(&self.preprocessing),
        )
    }
}

impl RowSource for TraceBackend {
    fn visit_chunks(
        &self,
        range: Range<usize>,
        chunk_size: usize,
        visitor: &mut ChunkVisitor<'_>,
    ) -> Result<(), WitnessError> {
        let cycles = checked_pow2(self.config.log_t)?;
        if range.end > cycles || range.start > range.end {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: format!("cycle range {range:?} exceeds the {cycles}-cycle domain"),
            });
        }
        let env = WitnessEnv::new(&self.preprocessing);
        let padding = TraceRow::default();
        let mut start = range.start;
        while start < range.end {
            let end = (start + chunk_size).min(range.end);
            let buffer: Vec<TraceRow> = (start..end)
                .map(|index| *self.rows.get(index).unwrap_or(&padding))
                .collect();
            let next_after = (end < cycles).then(|| self.rows.get(end).unwrap_or(&padding));
            visitor(&buffer, next_after, &env)?;
            start = end;
        }
        Ok(())
    }

    fn random_access(&self) -> Option<RandomAccessRows> {
        self.random_access_rows().ok()
    }
}

impl BundleSource for TraceBackend {
    fn bundles<B: WitnessBundle + Clone + Send + Sync>(&self) -> Result<Vec<B>, WitnessError> {
        self.walk_cycles(|row, next, env| B::from_row(row, next, env))
    }
}

//! Connected components ([`Partition`]) of frontier cells and cell priority ordering.

use crate::solver::{
    CellIdx, CellValue, IndexedOffsets, Metrics, cell::Cell, grid::Neighbor, indexed_neighbor_iter,
};
use std::{
    collections::VecDeque,
    fmt::Display,
    ops::{Index, IndexMut},
};

/// A connected component of bipartite frontier cells (`known` clue cells adjacent to `unknown` cells).
///
/// Partitions allow the solver to decompose the board into smaller, independent subproblems.
#[derive(Debug, Clone)]
pub struct Partition {
    /// Linear cell indices of frontier cells belonging to this partition.
    members: Vec<CellIdx>,
}

impl Partition {
    fn new(capacity: u16) -> Partition {
        Partition {
            members: Vec::with_capacity(capacity as usize),
        }
    }

    /// Returns the slice of linear cell indices belonging to this partition.
    #[inline]
    pub(super) fn members(&self) -> &[CellIdx] {
        &self.members
    }

    /// Returns the total number of cells (`known + unknown`) in this partition.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Returns `true` if the partition contains neither known nor unknown cells.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    fn reset(&mut self, cells: &mut [Cell]) {
        for &idx in &self.members {
            cells[idx].set_partition_id(PartitionId::new());
        }
        self.members.clear();
    }
}

#[derive(Debug)]
struct CachedPartition {
    partition: Partition,
    in_use: bool,
}

#[derive(Debug)]
pub(super) struct PartitionCache {
    partitions: Vec<CachedPartition>,
    recent_partitions: Vec<PartitionId>,
    new_partition_capacity: u16,
    work_queue: VecDeque<CellIdx>,
    breaking_work_queue: VecDeque<(CellIdx, Option<CellValue>)>,
    length: u16,
}

impl PartitionCache {
    pub(super) fn new(new_partition_capacity: u16) -> PartitionCache {
        PartitionCache {
            partitions: Vec::with_capacity(16),
            recent_partitions: Vec::with_capacity(16),
            new_partition_capacity,
            work_queue: VecDeque::with_capacity(new_partition_capacity as usize),
            breaking_work_queue: VecDeque::new(),
            length: 0,
        }
    }

    pub(super) fn create(
        &mut self,
        cells: &mut [Cell],
        offsets: IndexedOffsets,
        init_idx: CellIdx,
        metrics: &mut Metrics,
    ) -> Option<PartitionId> {
        let cell = &mut cells[init_idx];
        let value = cell.value();

        if value == CellValue::INFERRED_SAFE || cell.alt_neighbors().is_empty() {
            return None;
        }
        let id = self.reserve();
        let partition = &mut self.partitions[id].partition;

        let members = &mut partition.members;
        cell.set_partition_id(id);

        self.work_queue.push_back(init_idx);

        while let Some(cell_idx) = self.work_queue.pop_front() {
            let cell = cells[cell_idx];
            if cell.value().is_mine() {
                continue;
            }
            members.push(cell_idx);

            let alt_neighbors = cell.alt_neighbors();
            metrics.inc_neighbor_visits(alt_neighbors);
            for Neighbor { neighbor: idx, .. } in
                indexed_neighbor_iter(cell_idx, alt_neighbors, offsets)
            {
                let neighbor = &mut cells[idx];
                if neighbor.value() != CellValue::INFERRED_SAFE && neighbor.partition_id() != id {
                    neighbor.set_partition_id(id);
                    self.work_queue.push_back(idx);
                }
            }
        }
        metrics.inc_partition_visits(members.len());

        // clear out single partitions
        if members.len() <= 1 {
            self.release_no_recent_reset(id, cells);
            None
        } else {
            metrics.inc_partitions_created();
            metrics.set_max_partition_size(members.len());
            metrics.set_max_partitions(self.length);
            Some(id)
        }
    }

    pub(super) fn create_for_breaking(
        &mut self,
        cells: &mut [Cell],
        offsets: IndexedOffsets,
        init_idx: CellIdx,
        metrics: &mut Metrics,
    ) -> Option<PartitionId> {
        // Note: don't exit early on INFERRED_MINE - it will be reset
        if cells[init_idx].value() == CellValue::MINE {
            return None;
        }

        let id = self.reserve();
        let partition = &mut self.partitions[id].partition;

        let members = &mut partition.members;

        self.breaking_work_queue.push_back((init_idx, None));

        while let Some((cell_idx, ref_val)) = self.breaking_work_queue.pop_front() {
            // local copy of start_cell for resetting inferred state
            let cell = cells[cell_idx];
            if cell.value().is_inferred() {
                Cell::change_value(
                    cells,
                    cell_idx,
                    CellValue::UNKNOWN,
                    offsets,
                    metrics,
                    |_, _, _| {},
                );
            }
            if cell.partition_id() == id {
                continue;
            }
            let cell = &mut cells[cell_idx];
            if let Some(ref_val) = ref_val
                && !ref_val.is_alt(cell.value())
            {
                continue;
            }

            cell.set_partition_id(id);
            members.push(cell_idx);

            let ref_val = cell.value();

            // Use all neighbors rather than alt_neighbors, since their
            // inference may be reset
            metrics.inc_neighbor_visits(cell.neighbors());
            for Neighbor {
                neighbor: n_idx, ..
            } in indexed_neighbor_iter(cell_idx, cell.neighbors(), offsets)
            {
                let neighbor = cells[n_idx];
                if neighbor.partition_id() != id {
                    self.breaking_work_queue.push_back((n_idx, Some(ref_val)));
                }
            }
        }
        metrics.inc_partition_visits(members.len());

        // clear out single partitions
        if members.len() <= 1 {
            self.release_no_recent_reset(id, cells);
            None
        } else {
            metrics.inc_partitions_created();
            metrics.set_max_partition_size(members.len());
            metrics.set_max_partitions(self.length);
            Some(id)
        }
    }

    fn reserve(&mut self) -> PartitionId {
        for (id, partition) in self.partitions.iter_mut().enumerate() {
            if !partition.in_use {
                partition.in_use = true;
                self.length += 1;
                let pid = PartitionId::for_usize(id);
                self.recent_partitions.push(pid);
                return pid;
            }
        }
        let partition = CachedPartition {
            partition: Partition::new(self.new_partition_capacity),
            in_use: true,
        };
        if self.partitions.len() == u16::MAX as usize - 1 {
            panic!("Partition count exceeds maximum");
        }
        let pid = PartitionId::for_usize(self.partitions.len());
        self.partitions.push(partition);
        self.recent_partitions.push(pid);
        self.length += 1;
        pid
    }

    pub(super) fn get(&self, id: PartitionId) -> Option<&Partition> {
        if id == PartitionId::UNPARTITIONED {
            None
        } else {
            self.partitions.get(id.0 as usize).map(|p| &p.partition)
        }
    }

    pub(super) fn release(&mut self, id: PartitionId, cells: &mut [Cell]) {
        self.release_no_recent_reset(id, cells);
        self.recent_partitions.clear();
    }

    fn release_no_recent_reset(&mut self, id: PartitionId, cells: &mut [Cell]) {
        let p = &mut self.partitions[id];
        p.partition.reset(cells);
        p.in_use = false;
        self.length -= 1;
    }

    pub(super) fn recent_partitions(&self) -> impl Iterator<Item = PartitionId> {
        self.recent_partitions
            .iter()
            .copied()
            .filter(|&p| self.partitions[p].in_use)
    }

    pub(super) fn reset_recent_partitions(&mut self) {
        self.recent_partitions.clear();
    }

    pub(super) fn partitions(&self) -> impl Iterator<Item = PartitionId> {
        self.partitions
            .iter()
            .enumerate()
            .filter(|(_, p)| p.in_use)
            .map(|(i, _)| PartitionId::for_usize(i))
    }

    pub(super) fn len(&self) -> u16 {
        self.length
    }

    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub(super) fn capacity(&self) -> u16 {
        self.partitions.capacity() as u16
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) struct PartitionId(u16);

impl PartitionId {
    const UNPARTITIONED: PartitionId = PartitionId(u16::MAX);

    pub(super) const fn new() -> PartitionId {
        Self::UNPARTITIONED
    }

    fn for_usize(id: usize) -> PartitionId {
        PartitionId(id as u16)
    }

    pub(super) fn is_partitioned(&self) -> bool {
        *self != Self::UNPARTITIONED
    }
}

impl Index<PartitionId> for Vec<CachedPartition> {
    type Output = CachedPartition;

    fn index(&self, index: PartitionId) -> &Self::Output {
        &self[index.0 as usize]
    }
}

impl IndexMut<PartitionId> for Vec<CachedPartition> {
    fn index_mut(&mut self, index: PartitionId) -> &mut Self::Output {
        &mut self[index.0 as usize]
    }
}

impl Display for PartitionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Default for PartitionId {
    fn default() -> Self {
        Self::UNPARTITIONED
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{
        cell::CellValue,
        grid::{E, W},
    };

    fn create_partition(
        cells: &mut [Cell],
        offsets: IndexedOffsets,
        init_idx: CellIdx,
    ) -> Option<(PartitionId, Partition)> {
        let mut cache = PartitionCache::new(offsets.width() as u16 * 4);
        let id = cache.create(cells, offsets, init_idx, &mut Metrics::new());
        id.map(|i| (i, cache.partitions.swap_remove(i.0 as usize).partition))
    }

    #[test]
    fn test_partition_create_skips_mines_and_isolated_cells() {
        let mut cells = Cell::grid(2, 1);
        // Cell 0 is Unknown with no safe_neighbors (isolated)
        assert!(create_partition(&mut cells, IndexedOffsets::new(2), CellIdx(0),).is_none());

        // Cell 1 is a Mine even if safe_neighbors is non-empty
        cells[1].set_value(CellValue::MINE);
        cells[1].set_safe(W);
        assert!(create_partition(&mut cells, IndexedOffsets::new(2), CellIdx(1),).is_none());
    }

    #[test]
    fn test_partition_create_connected_bipartite_component() {
        // 3x1 board: [Count(1), Unknown, Count(2)]
        let mut cells = Cell::grid(3, 1);
        cells[0].set_value(CellValue::for_mine_count(1));

        cells[1].set_value(CellValue::UNKNOWN);
        cells[1].set_safe(W);
        cells[1].set_safe(E);

        cells[2].set_value(CellValue::for_mine_count(2));

        let (id, partition) = create_partition(&mut cells, IndexedOffsets::new(3), CellIdx(0))
            .expect("should create partition");
        assert!(!partition.is_empty());
        assert_eq!(partition.len(), 3);
        assert_eq!(partition.members(), &[CellIdx(0), CellIdx(1), CellIdx(2)]);
        for c in &cells {
            assert_eq!(c.partition_id(), id);
        }
    }
}

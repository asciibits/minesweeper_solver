//! Connected components ([`Partition`]) of frontier cells and cell priority ordering.

use crate::solver::{CellValue, IndexedOffsets, cell::Cell};
use std::collections::VecDeque;

/// A connected component of bipartite frontier cells (`known` clue cells adjacent to `unknown` cells).
///
/// Partitions allow the solver to decompose the board into smaller, independent subproblems.
#[derive(Debug, Clone)]
pub struct Partition {
    /// Linear cell indices of frontier cells belonging to this partition.
    members: Vec<usize>,
}

impl Partition {
    pub(super) fn new(members: Vec<usize>) -> Partition {
        Partition { members }
    }

    /// Returns the slice of linear cell indices belonging to this partition.
    #[inline]
    pub(super) fn members(&self) -> &[usize] {
        &self.members
    }

    /// Performs a breadth-first traversal over `alt_neighbors` starting from `start_idx`
    /// to collect an independent connected component of known and unknown boundary cells.
    ///
    /// Returns `None` if `start_idx` is a mine, inferred safe, or has no `alt_neighbors`
    /// (i.e., an interior known cell or an isolated unknown cell).
    ///
    /// Every visited cell is stamped with `partition_id`; cells already carrying that id are
    /// not revisited. Partitions containing only 1 cell are discarded.
    pub(super) fn create(
        cells: &mut [Cell],
        offsets: IndexedOffsets,
        start_idx: usize,
        partition_id: u32,
        work_queue: &mut VecDeque<usize>,
    ) -> Option<Partition> {
        let start_cell = &mut cells[start_idx];
        let value = start_cell.value();

        if value == CellValue::INFERRED_SAFE || start_cell.alt_neighbors().is_empty() {
            return None;
        }

        let perimeter = offsets.width() as usize * 4;
        let mut members = Vec::with_capacity(perimeter);
        start_cell.set_partition_id(partition_id);

        work_queue.push_back(start_idx);

        while let Some(cell_idx) = work_queue.pop_front() {
            let cell = cells[cell_idx];
            if cell.value().is_mine() {
                continue;
            }
            let cell = &mut cells[cell_idx];
            members.push(cell_idx);

            cell.mut_visit_neighbors(cell.alt_neighbors(), cells, offsets, |neighbor| {
                if neighbor.value() != CellValue::INFERRED_SAFE
                    && neighbor.set_partition_id(partition_id)
                {
                    work_queue.push_back(neighbor.idx());
                }
            });
        }

        // clear out single partitions
        if members.len() == 1 {
            cells[members[0]].set_partition_id(0);
            members.clear();
        }

        (!members.is_empty()).then_some(Partition::new(members))
    }

    /// Like [`Partition::create`], but resets any inferred cells encountered during traversal
    /// back to [`CellValue::UNKNOWN`], updating adjacent cells' neighbor masks accordingly.
    pub(super) fn create_reset_inferred(
        cells: &mut [Cell],
        offsets: IndexedOffsets,
        start_idx: usize,
        partition_id: u32,
    ) -> Option<Partition> {
        // Note: don't exit early on INFERRED_MINE - it will be reset
        if cells[start_idx].value() == CellValue::MINE {
            return None;
        }

        let perimeter = offsets.width() as usize * 4;
        let mut members = Vec::with_capacity(perimeter);
        let mut to_process: VecDeque<(usize, Option<CellValue>)> =
            VecDeque::with_capacity(perimeter);

        to_process.push_back((start_idx, None));

        while let Some((cell_idx, ref_val)) = to_process.pop_front() {
            // local copy of start_cell for resetting inferred state
            let cell = cells[cell_idx];
            if cell.value().is_inferred() {
                cell.change_value(cells, CellValue::UNKNOWN, offsets, |_| {});
            }
            if cell.partition_id() == partition_id {
                continue;
            }
            let cell = &mut cells[cell_idx];
            if let Some(ref_val) = ref_val
                && !ref_val.is_alt(cell.value())
            {
                continue;
            }

            cell.set_partition_id(partition_id);
            members.push(cell_idx);

            let ref_val = cell.value();

            // Use all neighbors rather than alt_neighbors, since their
            // inference may be reset
            cell.visit_neighbors(cell.neighbors(), cells, offsets, |neighbor| {
                if neighbor.partition_id() != partition_id {
                    to_process.push_back((neighbor.idx(), Some(ref_val)));
                }
            });
        }

        // clear out single partitions
        if members.len() == 1 {
            cells[members[0]].set_partition_id(0);
            members.clear();
        }

        (!members.is_empty()).then_some(Partition::new(members))
    }

    /// Returns the total number of cells (`known + unknown`) in this partition.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Returns `true` if the partition contains neither known nor unknown cells.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{
        cell::CellValue,
        grid::{E, W},
    };

    #[test]
    fn test_partition_create_skips_mines_and_isolated_cells() {
        let mut cells = Cell::grid(2, 1);
        // Cell 0 is Unknown with no safe_neighbors (isolated)
        cells[0].set_partition_id(99);
        assert!(
            Partition::create(
                &mut cells,
                IndexedOffsets::new(2),
                0,
                100,
                &mut VecDeque::new()
            )
            .is_none()
        );

        // Cell 1 is a Mine even if safe_neighbors is non-empty
        cells[1].set_value(CellValue::MINE);
        cells[1].set_safe(W);
        cells[1].set_partition_id(99);
        assert!(
            Partition::create(
                &mut cells,
                IndexedOffsets::new(2),
                1,
                100,
                &mut VecDeque::new()
            )
            .is_none()
        );
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

        let partition = Partition::create(
            &mut cells,
            IndexedOffsets::new(3),
            0,
            7,
            &mut VecDeque::new(),
        )
        .expect("should create partition");
        assert!(!partition.is_empty());
        assert_eq!(partition.len(), 3);
        assert_eq!(partition.members(), &[0, 1, 2]);
        for c in &cells {
            assert_eq!(c.partition_id(), 7);
        }
    }
}

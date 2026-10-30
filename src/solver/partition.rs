//! Connected components ([`Partition`]) of frontier cells and priority ordering ([`OrderedCell`]).

use std::{cmp::Ordering, collections::VecDeque};

use crate::solver::{IndexedOffsets, cell::Cell, grid::Neighbor, indexed_neighbor_iter, stats};

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
    /// Returns `None` if `start_idx` is a mine or has no `alt_neighbors` (i.e., an interior
    /// known cell or an isolated unknown cell).
    pub(super) fn create(
        cells: &mut [Cell],
        width: u32,
        height: u32,
        indexed_offsets: &IndexedOffsets,
        start_idx: usize,
        partition_id: u32,
        ignore_inferred: bool,
    ) -> Option<Partition> {
        let start_cell = &mut cells[start_idx];
        if start_cell.is_mine(ignore_inferred)
            || start_cell.alt_neighbors(ignore_inferred).is_empty()
        {
            start_cell.set_partition_id(0);
            return None;
        }

        let perimeter = (width as usize + height as usize) * 2;
        let mut members = Vec::with_capacity(perimeter);
        let mut to_process = VecDeque::with_capacity(perimeter);

        start_cell.set_partition_id(partition_id);
        to_process.push_back(start_idx);

        while let Some(cell_idx) = to_process.pop_front() {
            let cell = &cells[cell_idx];
            if cell.is_mine(ignore_inferred) {
                continue;
            }
            members.push(cell_idx);

            for Neighbor {
                neighbor: neighbor_idx,
                ..
            } in indexed_neighbor_iter(
                cell_idx,
                cell.alt_neighbors(ignore_inferred),
                indexed_offsets,
            ) {
                let neighbor_cell = &mut cells[neighbor_idx];
                if neighbor_cell.partition_id() != partition_id {
                    neighbor_cell.set_partition_id(partition_id);
                    to_process.push_back(neighbor_idx);
                }
            }
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

/// A known clue cell annotated with its local combinatorial complexity (`solution_count`),
/// ordered from most constrained (fewest valid mine placements) to least constrained.
#[derive(Debug, Copy, Hash, Clone, PartialEq, Eq)]
pub(super) struct CellOrdering {
    /// Linear index of the cell on the board.
    cell_idx: usize,
    /// Original clue count on the cell.
    mine_count: i8,
    /// Number of mines still unaccounted for among the cell's unknown neighbors.
    remaining_mines: i8,
    /// Number of adjacent unknown cells available to hold `remaining_mines`.
    available_cells: i8,
    /// Number of ways to choose `remaining_mines` from `available_cells` ($\binom{\text{available}}{\text{remaining}}$).
    solution_count: u8,
}

impl CellOrdering {
    /// Constructs an [`OrderedCell`] if `cell` holds a [`CellValue::Count`], or `None` otherwise.
    pub(super) fn from_cell(cell: &Cell) -> Option<CellOrdering> {
        if let Some(mine_count) = cell.value().mine_count() {
            let available_cells = cell.inferred_unknown_neighbors().len() as i8;
            let remaining_mines = mine_count - cell.inferred_mine_neighbors().len() as i8;
            let solution_count = stats::choose(available_cells, remaining_mines) as u8;
            Some(CellOrdering {
                cell_idx: cell.idx(),
                mine_count,
                remaining_mines,
                available_cells,
                solution_count,
            })
        } else {
            None
        }
    }

    /// Returns the linear board index of this cell.
    #[inline]
    pub(super) fn cell_idx(&self) -> usize {
        self.cell_idx
    }

    /// Returns the number of mines still unaccounted for among the cell's unknown neighbors.
    #[inline]
    pub(super) fn remaining_mines(&self) -> i8 {
        self.remaining_mines
    }

    /// Returns the number of ways to choose `remaining_mines` from `available_cells`.
    #[inline]
    pub(super) fn solution_count(&self) -> u8 {
        self.solution_count
    }
}

impl PartialOrd for CellOrdering {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CellOrdering {
    fn cmp(&self, other: &Self) -> Ordering {
        // Cells with 0 or 1 possible configurations are equally trivial (immediately solvable),
        // so clamp solution_count to at least 1 before tie-breaking.
        self.solution_count
            .max(1)
            .cmp(&other.solution_count.max(1))
            .then(self.mine_count.cmp(&other.mine_count))
            .then(self.remaining_mines.cmp(&other.remaining_mines))
            .then(self.cell_idx.cmp(&other.cell_idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{
        cell::CellValue,
        grid::{E, S, SE, W},
    };

    #[test]
    fn test_partition_create_skips_mines_and_isolated_cells() {
        let mut cells = Cell::grid(2, 1);
        // Cell 0 is Unknown with no safe_neighbors (isolated)
        cells[0].set_partition_id(99);
        assert!(
            Partition::create(&mut cells, 2, 1, &IndexedOffsets::new(2), 0, 1, false).is_none()
        );
        assert_eq!(cells[0].partition_id(), 0);

        // Cell 1 is a Mine even if safe_neighbors is non-empty
        cells[1].set_value(CellValue::MINE);
        cells[1].set_safe(W, true);
        cells[1].set_partition_id(99);
        assert!(
            Partition::create(&mut cells, 2, 1, &IndexedOffsets::new(2), 1, 1, false).is_none()
        );
        assert_eq!(cells[1].partition_id(), 0);
    }

    #[test]
    fn test_partition_create_connected_bipartite_component() {
        // 3x1 board: [Count(1), Unknown, Count(2)]
        let mut cells = Cell::grid(3, 1);
        cells[0].set_value(CellValue::for_mine_count(1));

        cells[1].set_value(CellValue::UNKNOWN);
        cells[1].set_safe(W, false);
        cells[1].set_safe(E, false);

        cells[2].set_value(CellValue::for_mine_count(2));

        let partition = Partition::create(&mut cells, 3, 1, &IndexedOffsets::new(3), 0, 7, false)
            .expect("should create partition");
        assert!(!partition.is_empty());
        assert_eq!(partition.len(), 3);
        assert_eq!(partition.members(), &[0, 1, 2]);
        for c in &cells {
            assert_eq!(c.partition_id(), 7);
        }
    }

    #[test]
    fn test_ordered_cell_from_cell_and_ordering() {
        let grid = Cell::grid(3, 3);
        let mut unknown_cell = grid[0];
        assert!(CellOrdering::from_cell(&unknown_cell).is_none());

        unknown_cell.set_value(CellValue::MINE);
        assert!(CellOrdering::from_cell(&unknown_cell).is_none());

        // Trivial cell A: 2 available, 2 remaining -> choose(2, 2) = 1 solution
        let mut cell_a = grid[0];
        cell_a.set_value(CellValue::for_mine_count(2));
        cell_a.set_safe(SE, false);

        // Trivial cell B: 2 available, 0 remaining (1 mine already found) -> choose(2, 0) = 1 solution, lower mine_count
        let mut cell_b = grid[1];
        cell_b.set_value(CellValue::for_mine_count(1));
        cell_b.set_mine(S, true);

        // Non-trivial cell C: 3 available, 1 remaining -> choose(3, 1) = 3 solutions
        let mut cell_c = grid[2];
        cell_c.set_value(CellValue::for_mine_count(1));

        let ord_a = CellOrdering::from_cell(&cell_a).unwrap();
        let ord_b = CellOrdering::from_cell(&cell_b).unwrap();
        let ord_c = CellOrdering::from_cell(&cell_c).unwrap();

        assert_eq!(ord_a.remaining_mines(), 2);
        assert_eq!(ord_b.remaining_mines(), 0);
        assert_eq!(ord_c.remaining_mines(), 1);

        assert_eq!(ord_a.solution_count(), 1);
        assert_eq!(ord_b.solution_count(), 1);
        assert_eq!(ord_c.solution_count(), 3);

        // ord_b < ord_a (both have solution_count <= 1, but ord_b has smaller mine_count)
        assert!(ord_b < ord_a);
        // ord_a < ord_c (1 solution < 3 solutions)
        assert!(ord_a < ord_c);
    }
}

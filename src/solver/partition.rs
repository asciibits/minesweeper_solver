//! Connected components ([`Partition`]) of frontier cells and priority ordering ([`OrderedCell`]).

use std::{cmp::Ordering, collections::VecDeque};

use crate::solver::{
    cell::{CellValue, ProcessedCell},
    grid::Neighbor,
    stats,
};

/// A connected component of bipartite frontier cells (`known` clue cells adjacent to `unknown` cells).
///
/// Partitions allow the solver to decompose the board into smaller, independent subproblems.
#[derive(Debug, Clone)]
pub struct Partition {
    /// Linear cell indices of revealed clue cells on this partition's boundary.
    pub(super) known: Vec<usize>,
    /// Linear cell indices of unrevealed cells adjacent to `known` cells in this partition.
    pub(super) unknown: Vec<usize>,
}

impl Partition {
    /// Performs a breadth-first traversal over `alt_neighbors` starting from `start_idx`
    /// to collect an independent connected component of known and unknown boundary cells.
    ///
    /// Returns `None` if `start_idx` is a mine or has no `alt_neighbors` (i.e., an interior
    /// known cell or an isolated unknown cell).
    pub(super) fn create(
        cells: &mut [ProcessedCell],
        width: u32,
        height: u32,
        start_idx: usize,
        partition_id: usize,
    ) -> Option<Partition> {
        let start_cell = &mut cells[start_idx];
        if start_cell.value() == CellValue::Mine || start_cell.alt_neighbors.is_empty() {
            start_cell.partition_id = 0;
            return None;
        }

        let perimiter = (width as usize + height as usize) * 2;
        let mut known = Vec::with_capacity(perimiter);
        let mut unknown = Vec::with_capacity(perimiter);
        let mut to_process = VecDeque::with_capacity(perimiter);

        cells[start_idx].partition_id = partition_id;
        to_process.push_back(start_idx);

        while let Some(cell_idx) = to_process.pop_front() {
            let cell = &cells[cell_idx];
            if cell.value() == CellValue::Mine {
                continue;
            }
            if cell.value().is_known() {
                known.push(cell_idx);
            } else {
                unknown.push(cell_idx);
            }

            for Neighbor { neighbor, .. } in cell.cell.point.neighbors(cell.alt_neighbors) {
                let neighbor_idx = neighbor.to_index(width);
                let neighbor_cell = &mut cells[neighbor_idx];
                if neighbor_cell.partition_id != partition_id {
                    neighbor_cell.partition_id = partition_id;
                    to_process.push_back(neighbor_idx);
                }
            }
        }

        (!known.is_empty() && !unknown.is_empty()).then_some(Partition { known, unknown })
    }

    /// Returns the total number of cells (`known + unknown`) in this partition.
    pub fn len(&self) -> usize {
        self.known.len() + self.unknown.len()
    }

    /// Returns `true` if the partition contains neither known nor unknown cells.
    pub fn is_empty(&self) -> bool {
        self.known.is_empty() && self.unknown.is_empty()
    }
}

/// A known clue cell annotated with its local combinatorial complexity (`solution_count`),
/// ordered from most constrained (fewest valid mine placements) to least constrained.
#[derive(Debug, Copy, Hash, Clone, PartialEq, Eq)]
pub(super) struct OrderedCell {
    /// Linear index of the cell on the board.
    pub(super) idx: usize,
    /// Original clue count on the cell.
    pub(super) mine_count: i8,
    /// Number of mines still unaccounted for among the cell's unknown neighbors.
    pub(super) remaining_mines: i8,
    /// Number of adjacent unknown cells available to hold `remaining_mines`.
    pub(super) available_cells: i8,
    /// Number of ways to choose `remaining_mines` from `available_cells` ($\binom{\text{available}}{\text{remaining}}$).
    pub(super) solution_count: u64,
}

impl OrderedCell {
    /// Constructs an [`OrderedCell`] if `cell` holds a [`CellValue::Count`], or `None` otherwise.
    pub(super) fn from_cell(cell: &ProcessedCell) -> Option<OrderedCell> {
        if let CellValue::Count(mine_count) = cell.value() {
            let available_cells = cell.alt_neighbors.len() as i8;
            Some(OrderedCell {
                idx: cell.idx,
                mine_count,
                remaining_mines: cell.remaining_mines,
                available_cells,
                solution_count: stats::choose(available_cells, cell.remaining_mines),
            })
        } else {
            None
        }
    }
}

impl PartialOrd for OrderedCell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedCell {
    fn cmp(&self, other: &Self) -> Ordering {
        // Cells with 0 or 1 possible configurations are equally trivial (immediately solvable),
        // so clamp solution_count to at least 1 before tie-breaking.
        self.solution_count
            .max(1)
            .cmp(&other.solution_count.max(1))
            .then(self.mine_count.cmp(&other.mine_count))
            .then(self.remaining_mines.cmp(&other.remaining_mines))
            .then(self.idx.cmp(&other.idx))
    }
}

//! Connected components ([`Partition`]) of frontier cells and priority ordering ([`CellOrdering`]).

use std::{cmp::Ordering, collections::VecDeque};

use crate::solver::{CellValue, IndexedOffsets, cell::Cell, stats};

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
    ///
    /// Every visited cell is stamped with `partition_id`; cells already carrying that id are
    /// not revisited. When `reset_inferred` is `true`, every inferred cell reached by the
    /// traversal is first reset to [`CellValue::UNKNOWN`] (updating its neighbors' masks) so the
    /// partition is rebuilt from user-supplied information only.
    pub(super) fn create(
        cells: &mut [Cell],
        offsets: IndexedOffsets,
        start_idx: usize,
        partition_id: u32,
    ) -> Option<Partition> {
        let start_cell = &mut cells[start_idx];
        let value = start_cell.value();

        if value == CellValue::INFERRED_SAFE || start_cell.alt_neighbors().is_empty() {
            return None;
        }

        let perimeter = offsets.width() as usize * 4;
        let mut members = Vec::with_capacity(perimeter);
        let mut to_process = VecDeque::with_capacity(perimeter);
        start_cell.set_partition_id(partition_id);

        to_process.push_back(start_idx);

        while let Some(cell_idx) = to_process.pop_front() {
            let cell = cells[cell_idx];
            if cell.value().is_mine() {
                continue;
            }
            let cell = &mut cells[cell_idx];
            members.push(cell_idx);

            cell.mut_visit_neighbors(cell.alt_neighbors(), cells, offsets, |neighbor, _| {
                if neighbor.value() != CellValue::INFERRED_SAFE {
                    if neighbor.set_partition_id(partition_id) {
                        to_process.push_back(neighbor.idx());
                    }
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
                cell.change_value(cells, CellValue::UNKNOWN, offsets, |_, _| {});
            }
            if cell.partition_id() == partition_id {
                continue;
            }
            let cell = &mut cells[cell_idx];
            if let Some(ref_val) = ref_val {
                if !ref_val.is_alt(cell.value()) {
                    dbg!("alt skipping:", cell_idx, cell.value().to_string(), ref_val.to_string());
                    continue;
                }
            }

            cell.set_partition_id(partition_id);
            members.push(cell_idx);

            let ref_val = cell.value();

            // Use all neighbors rather than alt_neighbors, since their
            // inference may be reset
            cell.visit_neighbors(cell.neighbors(), cells, offsets, |neighbor, _| {
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

/// A known clue cell annotated with its local combinatorial complexity (`solution_count`),
/// ordered from most constrained (fewest valid mine placements) to least constrained.
#[derive(Debug, Copy, Hash, Clone, PartialEq, Eq)]
pub(super) struct CellOrdering {
    /// Linear index of the cell on the board.
    cell_idx: usize,
    /// Original clue count on the cell.
    mine_count: u8,
    /// Number of mines still unaccounted for among the cell's unknown neighbors.
    remaining_mines: i8,
    /// Number of adjacent unknown cells available to hold `remaining_mines`.
    available_cells: u8,
    /// Number of ways to choose `remaining_mines` from `available_cells` ($\binom{\text{available}}{\text{remaining}}$).
    solution_count: u8,
}

impl CellOrdering {
    /// Constructs an [`OrderedCell`] if `cell` holds a [`CellValue::Count`], or `None` otherwise.
    pub(super) fn from_cell(cell: &Cell) -> Option<CellOrdering> {
        if let Some(mine_count) = cell.value().safe_mine_count() {
            let available_cells = cell.unknown_neighbors().len();
            let remaining_mines = cell.remaining_mines();
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
            .then(self.available_cells.cmp(&other.available_cells))
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
        assert!(Partition::create(&mut cells, IndexedOffsets::new(2), 0, 100).is_none());

        // Cell 1 is a Mine even if safe_neighbors is non-empty
        cells[1].set_value(CellValue::MINE);
        cells[1].set_safe(W);
        cells[1].set_partition_id(99);
        assert!(Partition::create(&mut cells, IndexedOffsets::new(2), 1, 100).is_none());
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

        let partition = Partition::create(&mut cells, IndexedOffsets::new(3), 0, 7)
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
        cell_a.set_safe(SE);

        // Trivial cell B: 2 available, 0 remaining (1 mine already found) -> choose(2, 0) = 1 solution, lower mine_count
        let mut cell_b = grid[1];
        cell_b.set_value(CellValue::for_mine_count(1));
        cell_b.set_mine(S);

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

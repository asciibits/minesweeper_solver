//! Core Minesweeper solver state and partition orchestration.

use cell::{Cell, CellValue, ProcessedCell};
use grid::{DirectionSet, Neighbor, Point};
use partition::{OrderedCell, Partition};
use std::{
    collections::{BTreeSet, HashMap},
    error::Error,
    fmt,
};

use crate::solver::cell::InferredState;

pub mod cell;
pub mod display;
pub mod grid;
pub mod partition;
mod stats;

/// Represents the state of a Minesweeper board and its active solving partitions.
#[derive(Debug, Clone)]
pub struct MineField {
    /// Width of the board in cells.
    pub width: u32,
    /// Height of the board in cells.
    pub height: u32,
    /// Total number of mines on the board.
    pub mine_count: u32,
    /// Row-major contiguous array of all board cells and their precomputed solver metadata.
    cells: Box<[ProcessedCell]>,
    /// Map of cell indices pending modification.
    pending: Vec<usize>,
    /// Set of non-dirty neighbor cell indices whose adjacency state changed.
    /// Note: Users of this vector must be prepared for duplicates, as the same
    /// neighbor may be adjacent to multiple dirty cells.
    dirty_neighbors: Vec<usize>,
    /// Active connected components of frontier cells, keyed by partition ID.
    partitions: HashMap<u32, Partition>,
    /// Monotonically increasing counter used to assign unique partition IDs.
    next_partition_id: u32,
}

impl MineField {
    /// Creates a new [`MineField`] with the given dimensions and total `mine_count`.
    ///
    /// All cells are initialized to [`CellValue::Unknown`].
    ///
    /// # Errors
    ///
    /// * Returns [`SolverError::AreaTooLarge`] if `width * height` exceeds `usize::MAX >> 9`.
    /// * Returns [`SolverError::MineCountTooLarge`] if `mine_count >= width * height`.
    pub fn new(width: u32, height: u32, mine_count: u32) -> Result<MineField, SolverError> {
        if (width as usize).saturating_mul(height as usize) > (usize::MAX >> 9) {
            return Err(SolverError::AreaTooLarge { width, height });
        }
        if mine_count as usize >= width as usize * height as usize {
            return Err(SolverError::MineCountTooLarge {
                mine_count,
                width,
                height,
            });
        }
        let mut cells = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                cells.push(ProcessedCell::new(Point { x, y }, width, height));
            }
        }

        let field = MineField {
            width,
            height,
            mine_count,
            cells: cells.into_boxed_slice(),
            pending: Vec::with_capacity(width as usize * height as usize / 4),
            partitions: HashMap::with_capacity(20),
            next_partition_id: 1,
            dirty_neighbors: Vec::with_capacity((width as usize + height as usize) * 4),
        };

        Ok(field)
    }

    /// Returns a reference to the [`Cell`] at the given coordinate `p`.
    ///
    /// # Panics
    ///
    /// Panics if `p` is outside the bounds of the field.
    pub fn cell(&self, p: Point) -> Cell {
        self.cells[self.cell_idx(p)].cell()
    }

    /// Updates a single cell's value and records it in `self.dirty` without running the solver.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CoordinatesOutOfBounds`] if `p.x >= self.width` or `p.y >= self.height`.
    fn set_cell_helper(&mut self, p: Point, value: CellValue) -> Result<(), SolverError> {
        if p.x >= self.width || p.y >= self.height {
            return Err(SolverError::CoordinatesOutOfBounds {
                p,
                width: self.width,
                height: self.height,
            });
        }
        let cell = &mut self.cells[self.cell_idx(p)];
        if cell.value != value {
            // Prepare the value to be set in the next solver pass.
            if cell.staged_value.is_none() {
                self.pending.push(cell.idx);
            }
            cell.staged_value = Some(value);
        } else {
            cell.staged_value = None;
        }
        Ok(())
    }

    /// Sets the [`CellValue`] at coordinate `p` and triggers a solver pass.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CoordinatesOutOfBounds`] if `p` is outside the board dimensions.
    pub fn set_cell_value(&mut self, p: Point, value: CellValue) -> Result<(), SolverError> {
        self.set_cell_helper(p, value)?;
        self.solve();
        Ok(())
    }

    /// Updates the board with the given [`Cell`] and triggers a solver pass.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CoordinatesOutOfBounds`] if `cell.point` is outside the board dimensions.
    pub fn set_cell(&mut self, cell: &Cell) -> Result<(), SolverError> {
        self.set_cell_value(cell.point, cell.value)
    }

    /// Updates a batch of [`Cell`]s and runs a single solver pass after all updates are applied.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CoordinatesOutOfBounds`] if any cell in `cells` lies outside the board.
    pub fn set_cells<'a>(
        &mut self,
        cells: impl IntoIterator<Item = &'a Cell>,
    ) -> Result<(), SolverError> {
        for cell in cells.into_iter() {
            self.set_cell_helper(cell.point, cell.value)?;
        }
        self.solve();
        Ok(())
    }

    /// Re-indexes modified cells, rebuilds affected frontier partitions, and runs deduction passes.
    fn solve(&mut self) {
        // TODO: Investigate any optimizations for single-cell updates
        self.index_pending();

        // Clear existing dirty partitions and repartition
        let start_id = self.next_partition_id;
        self.partition_dirty();

        self.basic_solve(start_id);

        self.pending.clear();
        self.dirty_neighbors.clear();
    }

    /// Updates neighbor masks (`unknown_neighbors`, `remaining_neighbors`) and `remaining_mines`
    /// for all cells in `dirty` and their immediate neighbors.
    ///
    /// Returns `true` if any inference was broken, `false` otherwise.
    fn index_pending(&mut self) -> bool {
        let mut inference_breaking = false;
        for &pending_idx in &self.pending {
            let dirty_cell = self.cells[pending_idx];
            let Some(new_value) = dirty_cell.staged_value else {
                continue;
            };

            let prev_value = dirty_cell.value;

            match prev_value {
                CellValue::Count(_) => {
                    // Transitioning from Count to Count. No changes to
                    // neighbor sets
                    if matches!(new_value, CellValue::Count(_)) {
                        continue;
                    }
                }
                val => {
                    if val == new_value {
                        continue;
                    }
                }
            }

            for Neighbor {
                neighbor: neighbor_pt,
                from,
            } in dirty_cell.point.neighbors(dirty_cell.neighbors)
            {
                let neighbor_cell = &mut self.cells[self.cell_idx(neighbor_pt)];
                // update any neighbors
                match prev_value {
                    CellValue::Unknown => neighbor_cell.remove_unknown_neighbor(from),
                    CellValue::Count(_) => neighbor_cell.remove_open_neighbor(from),
                    CellValue::Mine => neighbor_cell.remove_mine_neighbor(from),
                };
                match new_value {
                    CellValue::Unknown => neighbor_cell.add_unknown_neighbor(from),
                    CellValue::Count(_) => neighbor_cell.add_open_neighbor(from),
                    CellValue::Mine => neighbor_cell.add_mine_neighbor(from),
                };
                if neighbor_cell.staged_value.is_none() {
                    // Note: The same neighbor may be adjacent to multiple dirty
                    // cells, so we may index it multiple times. Future users of
                    // the dirty_neighbors must be prepared for duplicates.
                    self.dirty_neighbors.push(neighbor_cell.idx);
                }
            }

            // Transitioning from Open or Mine breaks previous inference passes,
            // as does a direct violation of existing inferences.
            inference_breaking = inference_breaking
                || !matches!(prev_value, CellValue::Unknown)
                || match dirty_cell.inferred_state {
                    InferredState::Mine => new_value != CellValue::Mine,
                    InferredState::Open => !matches!(new_value, CellValue::Count(_)),
                    InferredState::None => false,
                };

            let dirty_cell = &mut self.cells[pending_idx];
            dirty_cell.value = new_value;
            dirty_cell.inferred_state = InferredState::None;
        }

        // clear all staged values now that they have been applied
        for &pending_idx in &self.pending {
            self.cells[pending_idx].staged_value = None;
        }

        inference_breaking
    }

    /// Invalidates partitions touched by `dirty` or `dirty_neighbors` and constructs
    /// new connected [`Partition`]s from the updated frontier.
    ///
    /// Returns the IDs of all newly created partitions.
    fn partition_dirty(&mut self) {
        for &cell_idx in self.pending.iter().chain(self.dirty_neighbors.iter()) {
            let cell = &mut self.cells[cell_idx];
            if cell.partition_id != 0 {
                self.partitions.remove(&cell.partition_id);
                cell.partition_id = 0;
            }
        }

        // Re-partition the new cells
        // clone the keys to allow for mutating self.cells[xx].partition_id
        for cell_idx in self
            .pending
            .iter()
            .chain(self.dirty_neighbors.iter())
            .copied()
        {
            let cell = &self.cells[cell_idx];
            if cell.partition_id == 0 {
                let partition_id = self.next_partition_id;
                if let Some(partition) = Partition::create(
                    &mut self.cells,
                    self.width,
                    self.height,
                    cell_idx,
                    partition_id,
                ) {
                    self.partitions.insert(partition_id, partition);
                    self.next_partition_id += 1;
                }
            }
        }
    }

    /// Performs local constraint deductions on the newly created partitions.
    fn basic_solve(&mut self, start_partition_id: u32) {
        // Find all basic logic solutions
        for partition_id in start_partition_id..self.next_partition_id {
            let partition = self.partitions.get_mut(&partition_id).unwrap();

            let mut sorted_knowns = BTreeSet::new();
            for &cell_idx in &partition.members {
                let cell = &self.cells[cell_idx];
                if let Some(ordered_cell) = OrderedCell::from_cell(cell) {
                    sorted_knowns.insert(ordered_cell);
                }
            }

            loop {
                if let Some(easy) = sorted_knowns.pop_first() {
                    if easy.solution_count > 1 {
                        // We have exhausted the easy deductions for this partition
                        sorted_knowns.insert(easy);
                        break;
                    }

                    let cell = self.cells[easy.cell_idx];

                    if easy.solution_count == 0 {
                        // This is an impossible state, but we can still mark the
                        // alt_neighbors as open and update their neighbors.
                        partition.invalid.push(easy.cell_idx);
                    }

                    // two options here: either the remaining_mines are all in
                    // the remaining_neighbors, or none of them are
                    if easy.remaining_mines >= easy.available_cells {
                        // all remaining uninferred open cells are mines
                        for neighbor_idx in cell.point.neighbors(cell.inferred_unknown_neighbors) {
                            // let cell_idx = self.cell_idx(neighbor_idx.neighbor);
                            // let neighbor_cell = &mut self.cells[cell_idx];
                            // neighbor_cell.inferred_state = InferredState::Mine;
                        }
                    } else {
                        // mark alt_neighbors as open, and update their neighbors
                        // cell.inferred_state = InferredState::Open;
                    }
                } else {
                    break;
                }
            }

            eprintln!("For Partition {partition_id}");
            dbg!(sorted_knowns);
        }
    }

    /// Get the cell index for the given x/y coordinates.
    fn cell_idx(&self, p: Point) -> usize {
        p.to_index(self.width)
    }
}

/// Errors that can occur when constructing or updating a [`MineField`].
#[derive(Debug)]
#[non_exhaustive]
pub enum SolverError {
    /// The requested board dimensions exceed the maximum supported cell count.
    AreaTooLarge { width: u32, height: u32 },
    /// The requested mine count is greater than or equal to the total number of cells.
    MineCountTooLarge {
        mine_count: u32,
        width: u32,
        height: u32,
    },
    /// A cell coordinate lies outside the board's `width` and `height`.
    CoordinatesOutOfBounds { p: Point, width: u32, height: u32 },
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverError::AreaTooLarge { width, height } => write!(
                f,
                "The mine field area ({width} x {height}) must be below 2^{}.",
                usize::BITS - 10
            ),
            SolverError::MineCountTooLarge {
                mine_count,
                width,
                height,
            } => {
                write!(
                    f,
                    "The mine count ({mine_count}) must be strictly less than the area of the mine field ({width} x {height}."
                )
            }
            SolverError::CoordinatesOutOfBounds { p, width, height } => write!(
                f,
                "The provided coordinates {p} are outside the mine field ({width} x {height})."
            ),
        }
    }
}

impl Error for SolverError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::grid::{E, N, NE, NW, S, SE, SW, W};

    /// Helper asserting that a cell's `unknown_neighbors`, `open_neighbors`, and `mine_neighbors`
    /// form a pairwise-disjoint partition of `cell.neighbors`, and that its `inferred_*` sets
    /// are synced with its base neighbor sets after partition construction/clearing.
    fn assert_neighbor_invariants(cell: &ProcessedCell) {
        assert_eq!(
            cell.unknown_neighbors | cell.open_neighbors | cell.mine_neighbors,
            cell.neighbors
        );

        // standard set are pairwise disjoint
        assert!((cell.unknown_neighbors & cell.open_neighbors).is_empty());
        assert!((cell.unknown_neighbors & cell.mine_neighbors).is_empty());
        assert!((cell.open_neighbors & cell.mine_neighbors).is_empty());

        // standard set covers all neighbors
        assert_eq!(
            (cell.open_neighbors | cell.mine_neighbors | cell.unknown_neighbors),
            cell.neighbors
        );

        // inferred set are pairwise disjoint
        assert!((cell.inferred_unknown_neighbors & cell.inferred_open_neighbors).is_empty());
        assert!((cell.inferred_unknown_neighbors & cell.inferred_mine_neighbors).is_empty());
        assert!((cell.inferred_open_neighbors & cell.inferred_mine_neighbors).is_empty());

        // inferred set covers all neighbors
        assert_eq!(
            (cell.inferred_open_neighbors
                | cell.inferred_mine_neighbors
                | cell.inferred_unknown_neighbors),
            cell.neighbors
        );

        // inferred open and mines are a super-set of standard versions
        assert_eq!(
            cell.open_neighbors | cell.inferred_open_neighbors,
            cell.inferred_open_neighbors
        );
        assert_eq!(
            cell.mine_neighbors | cell.inferred_mine_neighbors,
            cell.inferred_mine_neighbors
        );
    }

    #[test]
    fn test_new_minefield_validation() {
        let field = MineField::new(10, 10, 15).expect("valid field");
        assert_eq!(field.width, 10);
        assert_eq!(field.height, 10);
        assert_eq!(field.mine_count, 15);
        assert_eq!(field.cell(Point { x: 0, y: 0 }).value, CellValue::Unknown);
        for cell in &field.cells {
            assert_eq!(cell.unknown_neighbors, cell.neighbors);
            assert!(cell.open_neighbors.is_empty());
            assert!(cell.mine_neighbors.is_empty());
            assert_neighbor_invariants(cell);
        }

        // AreaTooLarge
        let err = MineField::new(u32::MAX, u32::MAX, 10).unwrap_err();
        assert!(matches!(err, SolverError::AreaTooLarge { .. }));
        assert!(err.to_string().contains("must be below"));

        // MineCountTooLarge
        let err = MineField::new(4, 4, 16).unwrap_err();
        assert!(matches!(
            err,
            SolverError::MineCountTooLarge {
                mine_count: 16,
                width: 4,
                height: 4
            }
        ));
        assert!(err.to_string().contains("must be strictly less"));

        // 0x0 field rejected as MineCountTooLarge
        assert!(matches!(
            MineField::new(0, 0, 0),
            Err(SolverError::MineCountTooLarge { .. })
        ));
    }

    #[test]
    fn test_set_cell_out_of_bounds() {
        let mut field = MineField::new(3, 3, 1).unwrap();
        let err = field
            .set_cell_value(Point { x: 3, y: 0 }, CellValue::Count(1))
            .unwrap_err();
        assert!(matches!(
            err,
            SolverError::CoordinatesOutOfBounds {
                p: Point { x: 3, y: 0 },
                width: 3,
                height: 3
            }
        ));
        assert!(err.to_string().contains("outside the mine field"));

        let err_y = field
            .set_cell(&Cell {
                point: Point { x: 0, y: 3 },
                value: CellValue::Count(1),
            })
            .unwrap_err();
        assert!(matches!(err_y, SolverError::CoordinatesOutOfBounds { .. }));
    }

    #[test]
    fn test_neighbor_sets_and_remaining_mines_tracking() {
        let mut field = MineField::new(3, 3, 2).unwrap();

        // Reveal (0, 0) as Count(2)
        field
            .set_cell_value(Point { x: 0, y: 0 }, CellValue::Count(2))
            .unwrap();
        let c00 = field.cells[field.cell_idx(Point { x: 0, y: 0 })];
        assert_eq!(OrderedCell::from_cell(&c00).unwrap().remaining_mines, 2);
        assert_eq!(c00.unknown_neighbors, [E, S, SE].into_iter().collect());
        assert!(c00.open_neighbors.is_empty());
        assert!(c00.mine_neighbors.is_empty());
        assert_neighbor_invariants(&c00);

        // Neighbor (1, 0) is Unknown and should move W from unknown_neighbors to open_neighbors
        let c10 = field.cells[field.cell_idx(Point { x: 1, y: 0 })];
        assert_eq!(c10.open_neighbors, [W].into_iter().collect());
        assert_eq!(c10.unknown_neighbors, [E, SW, S, SE].into_iter().collect());
        assert!(c10.mine_neighbors.is_empty());
        assert_neighbor_invariants(&c10);

        // Mark (1, 0) as a Mine: (0, 0) moves E from unknown_neighbors to mine_neighbors (remaining_mines == 1)
        field
            .set_cell_value(Point { x: 1, y: 0 }, CellValue::Mine)
            .unwrap();
        let c00 = field.cells[field.cell_idx(Point { x: 0, y: 0 })];
        assert_eq!(c00.mine_neighbors, [E].into_iter().collect());
        assert!(c00.open_neighbors.is_empty());
        assert_eq!(c00.unknown_neighbors, [S, SE].into_iter().collect());
        assert_eq!(OrderedCell::from_cell(&c00).unwrap().remaining_mines, 1);
        assert_neighbor_invariants(&c00);

        let c10 = field.cells[field.cell_idx(Point { x: 1, y: 0 })];
        assert_eq!(c10.open_neighbors, [W].into_iter().collect());
        assert_eq!(c10.unknown_neighbors, [E, SW, S, SE].into_iter().collect());
        assert!(c10.mine_neighbors.is_empty());
        assert_neighbor_invariants(&c10);

        // Change (1, 0) from Mine to Count(1): (0, 0) moves E from mine_neighbors to open_neighbors
        field
            .set_cell_value(Point { x: 1, y: 0 }, CellValue::Count(1))
            .unwrap();
        let c00 = field.cells[field.cell_idx(Point { x: 0, y: 0 })];
        assert!(c00.mine_neighbors.is_empty());
        assert_eq!(c00.open_neighbors, [E].into_iter().collect());
        assert_eq!(c00.unknown_neighbors, [S, SE].into_iter().collect());
        assert_eq!(OrderedCell::from_cell(&c00).unwrap().remaining_mines, 2);
        assert_neighbor_invariants(&c00);

        // Revert (1, 0) back to Unknown: (0, 0) moves E from open_neighbors back to unknown_neighbors
        field
            .set_cell_value(Point { x: 1, y: 0 }, CellValue::Unknown)
            .unwrap();
        let c00 = field.cells[field.cell_idx(Point { x: 0, y: 0 })];
        assert!(c00.mine_neighbors.is_empty());
        assert!(c00.open_neighbors.is_empty());
        assert_eq!(c00.unknown_neighbors, [E, S, SE].into_iter().collect());
        assert_neighbor_invariants(&c00);
    }

    #[test]
    fn test_partitioning_and_repartitioning_across_updates() {
        // 5x1 strip: [Count(1), Unknown, Unknown, Unknown, Count(1)]
        let mut field = MineField::new(5, 1, 2).unwrap();
        field
            .set_cells(&[
                Cell {
                    point: Point { x: 0, y: 0 },
                    value: CellValue::Count(1),
                },
                Cell {
                    point: Point { x: 4, y: 0 },
                    value: CellValue::Count(1),
                },
            ])
            .unwrap();

        // Two disconnected frontier components: {(0,0), (1,0)} and {(3,0), (4,0)}
        assert_eq!(field.partitions.len(), 2);
        let p_left = field.cells[0].partition_id;
        let p_right = field.cells[4].partition_id;
        assert_ne!(p_left, p_right);
        assert_eq!(field.partitions[&p_left].len(), 2);
        assert_eq!(field.partitions[&p_right].len(), 2);
        assert_eq!(field.cells[2].partition_id, 0);
        assert_eq!(
            field.cells[2].unknown_neighbors,
            [W, E].into_iter().collect()
        );
        assert!(field.cells[2].open_neighbors.is_empty());

        // Reveal the middle cell (2, 0) as Count(1): now (1, 0) and (3, 0) both have
        // open_neighbors == [W, E], merging both components into a single 5-cell partition!
        field
            .set_cell_value(Point { x: 2, y: 0 }, CellValue::Count(1))
            .unwrap();
        assert_eq!(field.partitions.len(), 1);
        let merged_partition = field.partitions.values().next().unwrap();
        assert_eq!(merged_partition.len(), 5);
        assert_eq!(field.cells[1].open_neighbors, [W, E].into_iter().collect());
        assert!(field.cells[1].unknown_neighbors.is_empty());
        assert_eq!(field.cells[3].open_neighbors, [W, E].into_iter().collect());
        assert!(field.cells[3].unknown_neighbors.is_empty());
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_partition_split_with_distant_non_dirty_cells() {
        // 5x1 strip: [Unknown(0), Count(1), Unknown(2), Count(3), Unknown(4)]
        // Initially all 5 cells are connected via Unknown(2) into a single partition.
        let mut field = MineField::new(5, 1, 2).unwrap();
        field
            .set_cells(&[
                Cell {
                    point: Point { x: 1, y: 0 },
                    value: CellValue::Count(1),
                },
                Cell {
                    point: Point { x: 3, y: 0 },
                    value: CellValue::Count(1),
                },
            ])
            .unwrap();
        assert_eq!(field.partitions.len(), 1);
        let initial_pid = field.cells[0].partition_id;
        assert!(field.cells.iter().all(|c| c.partition_id == initial_pid));

        // Simulate dirty inferred_unknown_neighbors / inferred_open_neighbors on distant cell (0, 0)
        // field.cells[0].inferred_open_neighbors = DirectionSet::new();

        // Now reveal the bridge cell (2, 0) as Count(0).
        // Note that (0, 0) and (4, 0) are 2 hops away, so they are in neither `pending`
        // nor `dirty_neighbors` and still hold `initial_pid` entering Loop 2.
        field
            .set_cell_value(Point { x: 2, y: 0 }, CellValue::Count(0))
            .unwrap();

        // Should split into two 2-cell partitions: {(0,0), (1,0)} and {(3,0), (4,0)},
        // while interior cell (2,0) has no unknown_neighbors and gets partition_id == 0.
        assert_eq!(field.partitions.len(), 2);
        assert_eq!(field.cells[2].partition_id, 0);
        assert!(field.cells[2].unknown_neighbors.is_empty());
        assert_eq!(field.cells[2].open_neighbors, [W, E].into_iter().collect());

        let left_pid = field.cells[0].partition_id;
        let right_pid = field.cells[4].partition_id;
        assert_ne!(left_pid, 0);
        assert_ne!(right_pid, 0);
        assert_ne!(left_pid, right_pid);
        assert_eq!(field.cells[1].partition_id, left_pid);
        assert_eq!(field.cells[3].partition_id, right_pid);
        assert_eq!(field.partitions[&left_pid].len(), 2);
        assert_eq!(field.partitions[&right_pid].len(), 2);

        // Verify all cells (including distant cell 0) have synced inferred neighbor sets
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_partition_complete_dissolution() {
        // 2x1 board: [(0,0) = Count(1), (1,0) = Unknown]
        let mut field = MineField::new(2, 1, 1).unwrap();
        field
            .set_cell_value(Point { x: 0, y: 0 }, CellValue::Count(1))
            .unwrap();
        assert_eq!(field.partitions.len(), 1);
        assert_eq!(field.cells[0].unknown_neighbors, [E].into_iter().collect());
        assert_eq!(field.cells[1].open_neighbors, [W].into_iter().collect());

        // Flag the only unknown cell (1, 0) as a Mine: no frontier remains.
        field
            .set_cell_value(Point { x: 1, y: 0 }, CellValue::Mine)
            .unwrap();
        assert!(field.partitions.is_empty());
        assert_eq!(field.cells[0].partition_id, 0);
        assert_eq!(field.cells[1].partition_id, 0);
        assert_eq!(
            OrderedCell::from_cell(&field.cells[0])
                .unwrap()
                .remaining_mines,
            0
        );
        assert!(field.cells[0].unknown_neighbors.is_empty());
        assert!(field.cells[0].open_neighbors.is_empty());
        assert_eq!(field.cells[0].mine_neighbors, [E].into_iter().collect());

        assert!(field.cells[1].unknown_neighbors.is_empty());
        assert_eq!(field.cells[1].open_neighbors, [W].into_iter().collect());
        assert!(field.cells[1].mine_neighbors.is_empty());

        assert_neighbor_invariants(&field.cells[0]);
        assert_neighbor_invariants(&field.cells[1]);
    }

    #[test]
    fn test_batch_adjacent_cells_and_duplicate_dirty_neighbors() {
        // 3x2 board: reveal (1, 0) as Count(2) first
        let mut field = MineField::new(3, 2, 2).unwrap();
        field
            .set_cell_value(Point { x: 1, y: 0 }, CellValue::Count(2))
            .unwrap();
        assert_eq!(
            OrderedCell::from_cell(&field.cells[1])
                .unwrap()
                .remaining_mines,
            2
        );

        // In one batch, set both (0, 0) and (2, 0) to Mine, and (1, 1) to Count(2).
        // (1, 0) is a non-dirty neighbor of all three pending cells, so it appears 3 times
        // in `dirty_neighbors`.
        field
            .set_cells(&[
                Cell {
                    point: Point { x: 0, y: 0 },
                    value: CellValue::Mine,
                },
                Cell {
                    point: Point { x: 2, y: 0 },
                    value: CellValue::Mine,
                },
                Cell {
                    point: Point { x: 1, y: 1 },
                    value: CellValue::Count(2),
                },
            ])
            .unwrap();

        // Both (1, 0) and (1, 1) are adjacent to the two mines at (0, 0) and (2, 0),
        // so both should have remaining_mines == 0 and list each other in open_neighbors.
        let c10 = field.cells[field.cell_idx(Point { x: 1, y: 0 })];
        let c11 = field.cells[field.cell_idx(Point { x: 1, y: 1 })];
        assert_eq!(OrderedCell::from_cell(&c10).unwrap().remaining_mines, 0);
        assert_eq!(OrderedCell::from_cell(&c11).unwrap().remaining_mines, 0);

        assert_eq!(c10.mine_neighbors, [W, E].into_iter().collect());
        assert_eq!(c10.open_neighbors, [S].into_iter().collect());
        assert_eq!(c10.unknown_neighbors, [SW, SE].into_iter().collect());

        assert_eq!(c11.mine_neighbors, [NW, NE].into_iter().collect());
        assert_eq!(c11.open_neighbors, [N].into_iter().collect());
        assert_eq!(c11.unknown_neighbors, [W, E].into_iter().collect());

        assert_eq!(field.partitions.len(), 1);
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_repeated_and_reverted_staging_on_same_cell() {
        let mut field = MineField::new(3, 1, 1).unwrap();
        field
            .set_cell_value(Point { x: 1, y: 0 }, CellValue::Count(1))
            .unwrap();
        assert_eq!(
            OrderedCell::from_cell(&field.cells[1])
                .unwrap()
                .remaining_mines,
            1
        );

        // Stage (0, 0) as Mine -> revert to Unknown -> stage again as Mine in one batch.
        // Neighbor (1, 0)'s mine_neighbors and unknown_neighbors must update cleanly.
        field
            .set_cells(&[
                Cell {
                    point: Point { x: 0, y: 0 },
                    value: CellValue::Mine,
                },
                Cell {
                    point: Point { x: 0, y: 0 },
                    value: CellValue::Unknown,
                },
                Cell {
                    point: Point { x: 0, y: 0 },
                    value: CellValue::Mine,
                },
            ])
            .unwrap();

        assert_eq!(field.cell(Point { x: 0, y: 0 }).value, CellValue::Mine);
        assert_eq!(field.cells[1].mine_neighbors, [W].into_iter().collect());
        assert_eq!(field.cells[1].unknown_neighbors, [E].into_iter().collect());
        assert!(field.cells[1].open_neighbors.is_empty());
        assert_eq!(
            OrderedCell::from_cell(&field.cells[1])
                .unwrap()
                .remaining_mines,
            0
        );
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }
}

//! Core Minesweeper solver state and partition orchestration.
use cell::{Cell, UserCell, UserCellValue};
use grid::Point;
use partition::{CellOrdering, Partition};
use std::{
    collections::{BinaryHeap, HashMap, VecDeque},
    error::Error,
    fmt::{self, Display},
};

use crate::solver::{
    cell::{CellValue, CellValueChangeType, UserCellWithInference, cell_idx, coordinates},
    grid::{IndexedOffsets, Neighbor, indexed_neighbor_iter},
};

pub mod cell;
#[cfg(test)]
mod display;
pub mod grid;
pub mod partition;
mod stats;
#[cfg(test)]
mod test_utils;

/// Represents the state of a Minesweeper board and its active solving partitions.
#[derive(Debug, Clone)]
pub struct MineField {
    /// Width of the board in cells.
    width: u32,
    /// Height of the board in cells.
    height: u32,
    /// Total number of mines on the board.
    mine_count: u32,
    /// Row-major contiguous array of all board cells and their precomputed solver metadata.
    cells: Box<[Cell]>,
    /// User-supplied cell updates waiting to be applied by [`MineField::index_pending`].
    pending: Vec<UserCell>,
    /// Indices of cells whose neighborhood changed in a way that leaves existing inferences
    /// valid, queued as seeds for the next [`MineField::partition`] pass.
    /// Note: Users of this vector must be prepared for duplicates, as the same
    /// neighbor may be adjacent to multiple changed cells.
    pending_partition: Vec<usize>,
    /// Like `pending_partition`, but for changes that may invalidate existing inferences. Seeds
    /// from this list cause the partitions they touch to be rebuilt with inferences reset.
    breaking_pending_partition: Vec<usize>,
    partition_work_queue: VecDeque<usize>,
    global_solve_queue: BinaryHeap<CellOrdering>,

    /// Active connected components of frontier cells, keyed by partition ID.
    partitions: HashMap<u32, Partition>,
    /// Monotonically increasing counter used to assign unique partition IDs.
    next_partition_id: u32,
    /// Precomputed index offsets for walking a cell's neighbors in the flat `cells` array.
    indexed_offsets: IndexedOffsets,
}

impl MineField {
    /// Creates a new [`MineField`] with the given dimensions and total `mine_count`.
    ///
    /// All cells are initialized to [`UserCellValue::Unknown`].
    ///
    /// # Errors
    ///
    /// * Returns [`SolverError::AreaTooLarge`] if `width * height` exceeds `usize::MAX >> 9`.
    /// * Returns [`SolverError::MineCountTooLarge`] if `mine_count >= width * height`.
    pub fn new(width: u32, height: u32, mine_count: u32) -> Result<MineField, SolverError> {
        let area = (width as usize).saturating_mul(height as usize);
        let perimeter = (width as usize + height as usize) * 2;
        if area > (usize::MAX >> 9) {
            return Err(SolverError::AreaTooLarge { width, height });
        }
        if mine_count as usize >= area {
            return Err(SolverError::MineCountTooLarge {
                mine_count,
                width,
                height,
            });
        }

        let field = MineField {
            width,
            height,
            mine_count,
            cells: Cell::grid(width, height),
            pending: Vec::with_capacity(area / 4),
            partitions: HashMap::with_capacity(20),
            pending_partition: Vec::with_capacity(perimeter * 2),
            breaking_pending_partition: Vec::new(),
            partition_work_queue: VecDeque::with_capacity(perimeter * 2),
            global_solve_queue: BinaryHeap::with_capacity(perimeter),
            next_partition_id: 1,
            indexed_offsets: IndexedOffsets::new(width),
        };

        Ok(field)
    }

    /// Returns the width of the board in cells.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the height of the board in cells.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the total number of mines on the board.
    #[inline]
    pub fn mine_count(&self) -> u32 {
        self.mine_count
    }

    /// Returns the [`UserCellWithInference`] at the given coordinate `p`.
    pub fn cell(&self, p: Point) -> UserCellWithInference {
        self.cells[self.cell_idx(p)].value().into()
    }

    #[cfg(test)]
    fn cells(&self) -> &[Cell] {
        &self.cells
    }

    /// Queues a single cell update in `self.pending` without running the solver.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CoordinatesOutOfBounds`] if `p.x >= self.width` or `p.y >= self.height`.
    fn set_cell_helper(&mut self, cell: UserCell) -> Result<(), SolverError> {
        if cell.point.x >= self.width || cell.point.y >= self.height {
            return Err(SolverError::CoordinatesOutOfBounds {
                p: cell.point,
                width: self.width,
                height: self.height,
            });
        }
        self.pending.push(cell);
        Ok(())
    }

    /// Updates the board with the given [`UserCell`] and triggers a solver pass.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CoordinatesOutOfBounds`] if `cell.point` is outside the board dimensions.
    pub fn set_cell(&mut self, cell: UserCell) -> Result<(), SolverError> {
        self.set_cell_helper(cell)?;
        self.solve();
        Ok(())
    }

    /// Updates a batch of [`UserCell`]s and runs a single solver pass after all updates are applied.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CoordinatesOutOfBounds`] if any cell in `cells` lies outside the board.
    pub fn set_cells(
        &mut self,
        cells: impl IntoIterator<Item = UserCell>,
    ) -> Result<(), SolverError> {
        for cell in cells.into_iter() {
            self.set_cell_helper(cell)?;
        }
        self.solve();
        Ok(())
    }

    pub fn inferred_safe(&self) -> Vec<Point> {
        self.cells
            .iter()
            .filter(|&c| c.value().is_safe() && c.value().is_inferred())
            .map(|c| coordinates(c.idx(), self.width))
            .collect()
    }

    pub fn inferred_mines(&self) -> Vec<Point> {
        self.cells
            .iter()
            .filter(|&c| c.value().is_mine() && c.value().is_inferred())
            .map(|c| coordinates(c.idx(), self.width))
            .collect()
    }

    pub fn invalid(&self) -> Vec<Point> {
        self.cells
            .iter()
            .filter(|&c| c.is_invalid())
            .map(|c| coordinates(c.idx(), self.width))
            .collect()
    }

    /// Re-indexes modified cells, rebuilds affected frontier partitions, and runs deduction passes.
    fn solve(&mut self) {
        self.index_pending();

        // Clear existing dirty partitions and repartition
        let start_id = self.next_partition_id;
        self.partition();

        // Do another partitioning round if any new inferences were found in the
        // basic_solve phase
        self.basic_solve(start_id);

        // repartition anything that landed in the pending_partition list
        self.partition();
    }

    /// Applies every queued [`UserCell`] in `self.pending` to `self.cells`, updating the
    /// `unknown`/`mine` neighbor masks of each affected cell's neighbors, then drains `pending`.
    ///
    /// The indices of all neighbors of each changed cell are appended to
    /// `self.pending_partition`, or to `self.breaking_pending_partition` when the change may
    /// invalidate existing inferences (see [`CellValueChangeType::Breaking`]).
    fn index_pending(&mut self) {
        for &UserCell {
            point,
            value: new_value,
        } in &self.pending
        {
            let cells = &mut self.cells;
            let pending_cell = &mut cells[cell_idx(point, self.width)];
            let pending_idx = pending_cell.idx();
            let log_idx = self.pending_partition.len();

            let change_type = pending_cell.change_value(
                cells,
                new_value.into(),
                self.indexed_offsets,
                |neighbor| {
                    self.pending_partition.push(neighbor.idx());
                },
            );
            if !matches!(change_type, CellValueChangeType::NoChange) {
                self.pending_partition.push(pending_idx);
            }

            // move the entries from change_log to breaking_change_log if this
            // was a breaking change
            if matches!(change_type, CellValueChangeType::Breaking) {
                // this is uncommon, so keeping this off the hot path
                self.breaking_pending_partition
                    .extend(self.pending_partition.drain(log_idx..));
            }
        }

        self.pending.clear();
    }

    /// Dissolves the partitions touched by `breaking_pending_partition` / `pending_partition`
    /// and rebuilds connected [`Partition`]s from those seeds, then drains both lists.
    ///
    /// Seeds from `breaking_pending_partition` rebuild with `reset_inferred = true`, discarding
    /// inferences made inside the rebuilt partitions.
    fn partition(&mut self) {
        // clear out the old partitions
        for &cell_idx in self
            .breaking_pending_partition
            .iter()
            .chain(self.pending_partition.iter())
        {
            let cell = &mut self.cells[cell_idx];
            let pid = cell.partition_id();
            if pid != 0 {
                self.partitions.remove(&pid);
                cell.set_partition_id(0);
                // Note: no need to reset all partition_id values here. Just
                // resetting those in pending_partition speeds up this loop by
                // avoiding unnecessary calls to BTreeSet.remove
            }
        }

        // let partition_start
        let start_partition_id = self.next_partition_id;
        self.partition_reset_inferred(start_partition_id);
        self.partition_pending(start_partition_id);
    }

    fn partition_pending(&mut self, start_partition_id: u32) {
        // Re-partition the new cells.
        for &cell_idx in self.pending_partition.iter() {
            let cell = &self.cells[cell_idx];
            if cell.partition_id() < start_partition_id {
                let partition_id = self.next_partition_id;
                if let Some(partition) = Partition::create(
                    &mut self.cells,
                    self.indexed_offsets,
                    cell_idx,
                    partition_id,
                    &mut self.partition_work_queue,
                ) {
                    self.partitions.insert(partition_id, partition);
                    self.next_partition_id += 1;
                }
            }
        }

        self.pending_partition.clear();
    }

    fn partition_reset_inferred(&mut self, start_partition_id: u32) {
        // Re-partition the new cells.
        for &cell_idx in self.breaking_pending_partition.iter() {
            let cell = &self.cells[cell_idx];
            if cell.partition_id() < start_partition_id {
                let partition_id = self.next_partition_id;
                if let Some(partition) = Partition::create_reset_inferred(
                    &mut self.cells,
                    self.indexed_offsets,
                    cell_idx,
                    partition_id,
                ) {
                    self.partitions.insert(partition_id, partition);
                    self.next_partition_id += 1;
                }
            }
        }

        self.breaking_pending_partition.clear();
    }

    /// Performs local constraint deductions on the newly created partitions.
    ///
    /// Populates `self.pending` with newly inferred cells and `self.dirty_neighbors`
    /// with their neighbors, returning `true` if at least one cell was inferred.
    fn basic_solve(&mut self, start_partition_id: u32) {
        // Find all basic logic solutions
        let unresolved = &mut self.global_solve_queue;

        // work on all partitions simultaneously - load all knowns into the
        // ordered set. As long as the minimum has 1 or 0 solutions, keep
        // popping them off the set and processing their neighbors. As neighbors
        // are processed, first remove them from the set, make the changes, then
        // re-add them.
        for partition_id in start_partition_id..self.next_partition_id {
            let partition = self.partitions.get(&partition_id).unwrap();

            for &cell_idx in partition.members() {
                let cell = &self.cells[cell_idx];
                if let Some(ordered_cell) = CellOrdering::from_cell(cell)
                    && ordered_cell.solution_count() <= 1
                {
                    unresolved.push(ordered_cell);
                }
            }
        }

        while let Some(solve_cell_ord) = unresolved.pop() {
            let solve_cell = self.cells[solve_cell_ord.cell_idx()];

            if solve_cell.unknown_neighbors().is_empty() {
                // short-circuit cases where the there are no unknown neighbors.
                // The `unresolved` queue can get cluttered with lots of
                // already-resolved cells, and this exists before creating the
                // direction-set iterator below.
                continue;
            }

            let partition_id = solve_cell.partition_id();

            // two options here: either the remaining_mines are all in
            // the remaining_neighbors, or none of them are
            let inferred_value = if solve_cell_ord.remaining_mines() > 0 {
                CellValue::INFERRED_MINE
            } else {
                CellValue::INFERRED_SAFE
            };

            for Neighbor {
                neighbor: inferred_idx,
                ..
            } in indexed_neighbor_iter(
                solve_cell_ord.cell_idx(),
                solve_cell.unknown_neighbors(),
                self.indexed_offsets,
            ) {
                let inferred_cell = self.cells[inferred_idx];
                inferred_cell.change_value(
                    &mut self.cells,
                    inferred_value,
                    self.indexed_offsets,
                    |neighbor| {
                        if neighbor.partition_id() == partition_id
                            && !neighbor.value().is_inferred()
                            && let Some(cell_ord) = CellOrdering::from_cell(neighbor)
                        {
                            if cell_ord.solution_count() <= 1 {
                                unresolved.push(cell_ord);
                            }
                            self.pending_partition.push(neighbor.idx());
                        };
                    },
                );

                self.pending_partition.push(inferred_idx);
            }
        }
    }

    /// Get the cell index for the given x/y coordinates.
    fn cell_idx(&self, p: Point) -> usize {
        p.to_index(self.width)
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn to_details<'a>(&'a self) -> DetailMineField<'a> {
        DetailMineField(self)
    }
}

#[cfg(test)]
pub(crate) struct DetailMineField<'a>(&'a MineField);

#[cfg(test)]
impl<'a> Display for DetailMineField<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let spacer = if f.alternate() { "\n " } else { " " };
        write!(
            f,
            "MineField ({}, {}, {}) {{{spacer}",
            self.0.width, self.0.height, self.0.mine_count
        )?;
        for cell in &self.0.cells {
            cell.fmt(f)?;
            write!(f, ",{spacer}")?;
        }
        write!(f, "pending: {:?},{spacer}", self.0.pending)?;
        write!(
            f,
            "pending_partition: {:?},{spacer}",
            self.0.pending_partition
        )?;
        write!(
            f,
            "breaking_pending_partition: {:?},{spacer}",
            self.0.breaking_pending_partition
        )?;
        write!(
            f,
            "next_partition_id: {}{spacer}}}",
            self.0.next_partition_id
        )
    }
}

impl Display for MineField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let spacer = if f.alternate() { "\n  " } else { " " };
        write!(f, "MineField: [")?;
        for y in 0..self.height {
            write!(f, "{spacer}\"")?;
            for x in 0..self.width {
                let cell = self.cell(Point { x, y });
                let char = match cell.value {
                    UserCellValue::Mine => '*',
                    UserCellValue::Unknown => '.',
                    UserCellValue::Safe(n) => n.to_string().chars().next().unwrap(),
                };
                write!(f, "{char}")?;
            }
            write!(f, "\",")?;
        }
        if f.alternate() {
            writeln!(f)?;
        }
        write!(f, "]")
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
    use crate::solver::{
        cell::InferredValue,
        grid::{E, N, NE, NW, S, SE, SW, W},
        test_utils::{create_minefield, update_minefield},
    };

    /// Helper asserting that a cell's `unknown_neighbors`, `safe_neighbors`, and `mine_neighbors`
    /// form a pairwise-disjoint partition of `cell.neighbors`, and that its `inferred_*` sets
    /// are synced with its base neighbor sets after partition construction/clearing.
    fn assert_neighbor_invariants(cell: &Cell) {
        assert_eq!(
            cell.unknown_neighbors() | cell.safe_neighbors() | cell.mine_neighbors(),
            cell.neighbors()
        );

        // standard set are pairwise disjoint
        assert!((cell.unknown_neighbors() & cell.safe_neighbors()).is_empty());
        assert!((cell.unknown_neighbors() & cell.mine_neighbors()).is_empty());
        assert!((cell.safe_neighbors() & cell.mine_neighbors()).is_empty());

        // standard set covers all neighbors
        assert_eq!(
            (cell.safe_neighbors() | cell.mine_neighbors() | cell.unknown_neighbors()),
            cell.neighbors()
        );

        // inferred set are pairwise disjoint
        assert!((cell.unknown_neighbors() & cell.safe_neighbors()).is_empty());
        assert!((cell.unknown_neighbors() & cell.mine_neighbors()).is_empty());
        assert!((cell.safe_neighbors() & cell.mine_neighbors()).is_empty());

        // inferred set covers all neighbors
        assert_eq!(
            (cell.safe_neighbors() | cell.mine_neighbors() | cell.unknown_neighbors()),
            cell.neighbors()
        );

        // inferred safe and mines are a super-set of standard versions
        assert_eq!(
            cell.safe_neighbors() | cell.safe_neighbors(),
            cell.safe_neighbors()
        );
        assert_eq!(
            cell.mine_neighbors() | cell.mine_neighbors(),
            cell.mine_neighbors()
        );
    }

    #[test]
    fn test_new_minefield_validation() {
        let field = MineField::new(10, 10, 15).expect("valid field");
        assert_eq!(field.width(), 10);
        assert_eq!(field.height(), 10);
        assert_eq!(field.mine_count(), 15);
        assert_eq!(
            field.cell(Point { x: 0, y: 0 }).value,
            UserCellValue::Unknown
        );
        for cell in &field.cells {
            assert_eq!(cell.unknown_neighbors(), cell.neighbors());
            assert!(cell.safe_neighbors().is_empty());
            assert!(cell.mine_neighbors().is_empty());
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
            .set_cell(UserCell {
                point: Point { x: 3, y: 0 },
                value: UserCellValue::Safe(1),
            })
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
            .set_cell(UserCell {
                point: Point { x: 0, y: 3 },
                value: UserCellValue::Safe(1),
            })
            .unwrap_err();
        assert!(matches!(err_y, SolverError::CoordinatesOutOfBounds { .. }));
    }

    #[test]
    fn test_neighbor_sets_and_remaining_mines_tracking() {
        // Reveal (0, 0) as Count(2) on a 3x3 board
        let mut field = create_minefield(["2..", "...", "..."], 2);
        let c00 = field.cells[field.cell_idx(Point { x: 0, y: 0 })];
        assert_eq!(c00.remaining_mines(), 2);
        assert_eq!(c00.unknown_neighbors(), E | S | SE);
        assert!(c00.safe_neighbors().is_empty());
        assert!(c00.mine_neighbors().is_empty());
        assert_neighbor_invariants(&c00);

        // Neighbor (1, 0) is Unknown and should move W from unknown_neighbors to safe_neighbors
        let c10 = field.cells[field.cell_idx(Point { x: 1, y: 0 })];
        assert_eq!(c10.safe_neighbors(), !!W);
        assert_eq!(c10.unknown_neighbors(), E | SW | S | SE);
        assert!(c10.mine_neighbors().is_empty());
        assert_neighbor_invariants(&c10);

        // Mark (1, 0) as a Mine: (0, 0) moves E from unknown_neighbors to
        // mine_neighbors (remaining_mines == 1)
        update_minefield(&mut field, [".*"]);
        let c00 = field.cells[field.cell_idx(Point { x: 0, y: 0 })];
        assert_eq!(c00.mine_neighbors(), !!E);
        assert!(c00.safe_neighbors().is_empty());
        assert_eq!(c00.unknown_neighbors(), S | SE);
        assert_eq!(c00.remaining_mines(), 1);
        assert_neighbor_invariants(&c00);

        let c10 = field.cells[field.cell_idx(Point { x: 1, y: 0 })];
        assert_eq!(c10.safe_neighbors(), !!W);
        assert_eq!(c10.unknown_neighbors(), E | SW | S | SE);
        assert!(c10.mine_neighbors().is_empty());
        assert_neighbor_invariants(&c10);

        // Change (1, 0) from Mine to Count(1): (0, 0) moves E from mine_neighbors to safe_neighbors.
        // Because (0, 0) is Count(2) and now has only 2 unknown neighbors left (S and SE),
        // basic_solve infers both S and SE are mines, reducing remaining_mines() to 0!
        update_minefield(&mut field, [".1"]);
        let c00 = field.cells[field.cell_idx(Point { x: 0, y: 0 })];
        assert_eq!(c00.safe_neighbors(), !!E);
        assert!(c00.unknown_neighbors().is_empty());
        assert_eq!(c00.mine_neighbors(), S | SE);
        assert_eq!(c00.remaining_mines(), 0);
        assert_neighbor_invariants(&c00);

        // Revert (1, 0) back to Unknown (inference-breaking): (0, 0) moves E from safe_neighbors
        // back to unknown_neighbors, and the inferences on S and SE are cleared!
        update_minefield(&mut field, ["._"]);
        let c00 = field.cells[field.cell_idx(Point { x: 0, y: 0 })];
        assert!(c00.mine_neighbors().is_empty());
        assert!(c00.safe_neighbors().is_empty());
        assert_eq!(c00.unknown_neighbors(), E | S | SE);
        assert!(c00.mine_neighbors().is_empty());
        assert_eq!(c00.remaining_mines(), 2);
        assert_neighbor_invariants(&c00);
    }

    #[test]
    fn test_partitioning_and_repartitioning_across_updates() {
        // 5x2 board with Count(1) at top-left (0, 0) and top-right (4, 0)
        // Each corner has 3 unknown neighbors (choose(3, 1) = 3 > 1), so neither is trivially solved.
        let mut field = create_minefield(["1...1", "....."], 2);

        // Two disconnected frontier components:
        // Left: {(0,0), (1,0), (0,1), (1,1)} (4 cells)
        // Right: {(4,0), (3,0), (4,1), (3,1)} (4 cells)
        assert_eq!(field.partitions.len(), 2);
        let p_left = field.cells[0].partition_id();
        let p_right = field.cells[4].partition_id();
        assert_ne!(p_left, p_right);
        assert_eq!(field.partitions[&p_left].len(), 4);
        assert_eq!(field.partitions[&p_right].len(), 4);
        assert_eq!(field.cells[2].partition_id(), 0);
        assert_eq!(field.cells[2].unknown_neighbors(), W | E | SW | S | SE);
        assert!(field.cells[2].safe_neighbors().is_empty());

        // Reveal the middle top cell (2, 0) as Count(2): now both components are bridged
        // into a single 10-cell partition!
        update_minefield(&mut field, ["..2"]);
        assert_eq!(field.partitions.len(), 1);
        let merged_partition = field.partitions.values().next().unwrap();
        assert_eq!(merged_partition.len(), 10);
        assert_eq!(field.cells[1].safe_neighbors(), W | E);
        assert_eq!(field.cells[3].safe_neighbors(), W | E);
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_partition_split_with_distant_non_dirty_cells() {
        // 5x2 board: reveal (1, 0) = Count(1) and (3, 0) = Count(1).
        // Initially all 10 cells are connected via (2, 0) and (2, 1) into a single partition.
        let mut field = create_minefield([".1.1.", "....."], 2);
        assert_eq!(field.partitions.len(), 1);
        let initial_pid = field.cells[0].partition_id();
        assert!(field.cells.iter().all(|c| c.partition_id() == initial_pid));

        // Now reveal the bridge cell (2, 0) as Count(0).
        // `basic_solve` infers (1, 1), (2, 1), and (3, 1) as InferredValue::Safe,
        // and the post-`basic_solve` repartition pass splits the board into two 3-cell
        // partitions: {(0,0), (0,1), (1,0)} and {(3,0), (4,0), (4,1)}, while (2,0)
        // and its inferred-safe neighbors get partition_id == 0.
        update_minefield(&mut field, ["..0"]);

        assert_eq!(field.partitions.len(), 2);
        assert_eq!(field.cells[2].partition_id(), 0);
        assert_eq!(
            field.cells[field.cell_idx(Point { x: 1, y: 1 })].partition_id(),
            0
        );
        assert_eq!(
            field.cells[field.cell_idx(Point { x: 2, y: 1 })].partition_id(),
            0
        );
        assert_eq!(
            field.cells[field.cell_idx(Point { x: 3, y: 1 })].partition_id(),
            0
        );
        assert_eq!(field.cells[2].safe_neighbors(), W | E | SW | S | SE);

        let left_pid = field.cells[0].partition_id();
        let right_pid = field.cells[4].partition_id();
        assert_ne!(left_pid, 0);
        assert_ne!(right_pid, 0);
        assert_ne!(left_pid, right_pid);
        assert_eq!(field.cells[1].partition_id(), left_pid);
        assert_eq!(
            field.cells[field.cell_idx(Point { x: 0, y: 1 })].partition_id(),
            left_pid
        );
        assert_eq!(field.cells[3].partition_id(), right_pid);
        assert_eq!(
            field.cells[field.cell_idx(Point { x: 4, y: 1 })].partition_id(),
            right_pid
        );
        assert_eq!(field.partitions[&left_pid].len(), 3);
        assert_eq!(field.partitions[&right_pid].len(), 3);

        // Verify all cells (including distant cell 0) have synced inferred neighbor sets
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_partition_complete_dissolution() {
        // 3x1 board: [(0,0) = Unknown, (1,0) = Count(1), (2,0) = Unknown]
        // (1,0) has 2 unknown neighbors (choose(2, 1) = 2 > 1), so 1 partition of 3 cells remains.
        let mut field = create_minefield([".1."], 1);
        assert_eq!(field.partitions.len(), 1);
        assert_eq!(field.cells[1].unknown_neighbors(), W | E);
        assert_eq!(field.cells[0].safe_neighbors(), !!E);
        assert_eq!(field.cells[2].safe_neighbors(), !!W);

        // Flag (0, 0) as a Mine: basic_solve infers (2, 0) is Safe, and the post-basic_solve
        // repartition pass dissolves the partition completely.
        update_minefield(&mut field, ["*"]);

        assert!(field.partitions.is_empty());
        assert_eq!(field.cells[0].partition_id(), 0);
        assert_eq!(field.cells[1].partition_id(), 0);
        assert_eq!(field.cells[2].partition_id(), 0);
        assert!(field.cells[2].value().is_safe());
        assert_eq!(field.cells[2].remaining_mines(), 0);
        assert!(field.cells[1].unknown_neighbors().is_empty());
        assert_eq!(field.cells[1].safe_neighbors(), !!E);
        assert_eq!(field.cells[1].mine_neighbors(), !!W);

        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_batch_adjacent_cells_and_duplicate_dirty_neighbors() {
        // 3x2 board: reveal (1, 0) as Count(2) first
        let mut field = create_minefield([".2.", "..."], 2);
        assert_eq!(field.cells[1].remaining_mines(), 2);

        // In one batch, set both (0, 0) and (2, 0) to Mine, and (1, 1) to Count(2).
        // (1, 0) is a non-dirty neighbor of all three pending cells, so it appears 3 times
        // in `dirty_neighbors`.
        update_minefield(&mut field, ["*.*", ".2."]);

        // Both (1, 0) and (1, 1) are adjacent to the two mines at (0, 0) and (2, 0),
        // so both have remaining_mines == 0, basic_solve infers (0, 1) and (2, 1) as Safe,
        // and the post-basic_solve repartition pass dissolves the solved partition.
        let c10 = field.cells[field.cell_idx(Point { x: 1, y: 0 })];
        let c11 = field.cells[field.cell_idx(Point { x: 1, y: 1 })];
        assert_eq!(c10.remaining_mines(), 0);
        assert_eq!(c11.remaining_mines(), 0);

        assert_eq!(c10.mine_neighbors(), W | E);
        assert_eq!(c10.safe_neighbors(), SW | S | SE);
        assert!(c10.unknown_neighbors().is_empty());

        assert_eq!(c11.mine_neighbors(), NW | NE);
        assert_eq!(c11.safe_neighbors(), N | W | E);
        assert!(c11.unknown_neighbors().is_empty());

        assert!(field.partitions.is_empty());
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_repeated_and_reverted_staging_on_same_cell() {
        let mut field = create_minefield([".1."], 1);
        assert_eq!(field.cells[1].remaining_mines(), 1);

        // Stage (0, 0) as Mine -> revert to Unknown -> stage again as Mine in one batch.
        // Neighbor (1, 0)'s mine_neighbors and unknown_neighbors must update cleanly.
        field
            .set_cells([
                UserCell {
                    point: Point { x: 0, y: 0 },
                    value: UserCellValue::Mine,
                },
                UserCell {
                    point: Point { x: 0, y: 0 },
                    value: UserCellValue::Unknown,
                },
                UserCell {
                    point: Point { x: 0, y: 0 },
                    value: UserCellValue::Mine,
                },
            ])
            .unwrap();

        assert_eq!(field.cell(Point { x: 0, y: 0 }).value, UserCellValue::Mine);
        assert_eq!(field.cells[1].mine_neighbors(), !!W);
        // basic_solve now infers (2,0) is safe. Inferred cells are folded into the neighbor
        // masks, so it moves from `unknown_neighbors` to `safe_neighbors`.
        assert_eq!(
            field.cell(Point { x: 2, y: 0 }).inferred,
            InferredValue::Safe
        );
        assert!(field.cells[1].unknown_neighbors().is_empty());
        assert_eq!(field.cells[1].safe_neighbors(), !!E);
        assert_eq!(field.cells[1].remaining_mines(), 0);
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_basic_solve_mine_and_safe_inferences_and_cascading() {
        // 4x1 strip: [(0,0)=Count(1), (1,0)=Unknown, (2,0)=Count(1), (3,0)=Unknown]
        // 1. (0,0)=Count(1) has only 1 unknown neighbor (1,0) -> infers (1,0) is Mine.
        // 2. That satisfies (2,0)=Count(1), which then infers (3,0) is Safe!
        let field = create_minefield(["1.1."], 1);

        assert!(field.cells[1].value().is_mine());
        assert!(field.cells[3].value().is_safe());

        // Verify inferred neighbor sets reflect the deductions
        assert_eq!(field.cells[0].mine_neighbors(), !!E);
        assert!(field.cells[0].unknown_neighbors().is_empty());
        assert_eq!(field.cells[2].mine_neighbors(), !!W);
        assert_eq!(field.cells[2].safe_neighbors(), !!E);
        assert!(field.cells[2].unknown_neighbors().is_empty());

        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_basic_solve_cascade_multi_step_chain() {
        // 6x1 strip: [(0,0)=1, (1,0)=., (2,0)=2, (3,0)=., (4,0)=1, (5,0)=.]
        // Hop 1: (0,0)=1 has 1 unknown neighbor (1,0) -> infers (1,0) is Mine.
        // Hop 2: (2,0)=2 now has 1 mine at (1,0), so 1 remaining mine for (3,0) -> infers (3,0) is Mine.
        // Hop 3: (4,0)=1 now has 1 mine at (3,0), so 0 remaining mines for (5,0) -> infers (5,0) is Safe.
        let field = create_minefield(["1.2.1."], 2);

        assert!(field.cells[1].value().is_mine() && field.cells[1].value().is_inferred());
        assert!(field.cells[3].value().is_mine() && field.cells[3].value().is_inferred());
        assert!(field.cells[5].value().is_safe() && field.cells[5].value().is_inferred());

        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_basic_solve_detects_impossible_contradictory_clues() {
        // 3x1 strip: [(0,0)=Count(0), (1,0)=Unknown, (2,0)=Count(1)]
        // (0,0)=Count(0) infers (1,0) is Safe, leaving (2,0)=Count(1) with 0 available cells
        // and 1 remaining mine (solution_count == 0).
        let field = create_minefield(["0.1"], 1);

        assert!(field.cells[1].value().is_safe());
        assert!(field.cells[2].is_invalid());
        assert_eq!(field.invalid().len(), 1);
        assert!(field.invalid().contains(&Point { x: 2, y: 0 }));
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_inference_compatible_update_preserves_inferences() {
        // 4x1 strip: [(0,0)=Count(1), (1,0)=Unknown, (2,0)=Unknown, (3,0)=Unknown]
        // Round 1: Reveal (0,0)=Count(1) -> infers (1,0) is Mine, and updates (2,0)'s
        // inferred_mine_neighbors to include W.
        let mut field = create_minefield(["1..."], 1);
        assert!(field.cells[1].value().is_mine());
        assert_eq!(field.cells[2].mine_neighbors(), !!W);

        // Round 2 (inference-compatible): Reveal (2,0) as Count(1).
        // Because (1,0)'s Mine inference is preserved, (2,0) immediately knows its 1 mine is at W
        // and infers (3,0) is Safe!
        update_minefield(&mut field, ["..1"]);
        assert!(field.cells[1].value().is_mine());
        assert!(field.cells[3].value().is_safe());
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_inference_breaking_updates_clear_and_recompute_inferences() {
        // 3x1 strip: [(0,0), (1,0), (2,0)]
        // Round 1: Reveal (1,0) as Count(2) -> both (0,0) and (2,0) are inferred as Mine.
        let mut field = create_minefield([".2."], 2);
        assert!(field.cells[0].value().is_mine());
        assert!(field.cells[2].value().is_mine());

        // Round 2 (Inference-breaking Case A): Change (1,0) from Count(2) to Count(1).
        // Neither (0,0) nor (2,0) can be inferred anymore; both must reset to InferredState::None!
        update_minefield(&mut field, [".1"]);
        assert!(!field.cells[0].value().is_inferred());
        assert!(!field.cells[2].value().is_inferred());
        assert_eq!(field.cells[1].unknown_neighbors(), W | E);
        assert!(field.cells[1].mine_neighbors().is_empty());
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }

        // Round 3: Change (1,0) to Count(0) -> both (0,0) and (2,0) are inferred as Safe.
        update_minefield(&mut field, [".0"]);
        assert!(field.cells[0].value().is_safe());
        assert!(field.cells[2].value().is_safe());

        // Round 4 (Inference-breaking Case B): Directly violate inference on (0,0) by setting it to Mine,
        // and simultaneously change (1,0) to Count(1).
        // (2,0) must remain/be recomputed as Safe, and all neighbor invariants must hold.
        update_minefield(&mut field, ["*1"]);
        assert!(!field.cells[0].value().is_inferred());
        assert!(field.cells[2].value().is_safe());
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }

        // Round 5 (Inference-breaking Case C): Dissolve the partition completely by resetting
        // (0,0) and (1,0) to Unknown. Verify (2,0)'s Safe inference is cleared even though no
        // new partition is created!
        update_minefield(&mut field, ["__"]);
        assert!(field.partitions.is_empty());
        for cell in &field.cells {
            assert!(!cell.value().is_inferred());
            assert_eq!(cell.unknown_neighbors(), cell.unknown_neighbors());
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_inference_breaking_update_preserves_untouched_partitions() {
        // 5x1 strip: [(0,0)=Count(1), (1,0)=Unknown, (2,0)=Unknown, (3,0)=Unknown, (4,0)=Count(1)]
        // Two independent partitions:
        // - Left partition {(0,0), (1,0)} infers (1,0) is Mine.
        // - Right partition {(3,0), (4,0)} infers (3,0) is Mine.
        let mut field = create_minefield(["1...1"], 2);
        assert!(field.cells[1].value().is_mine());
        assert!(field.cells[3].value().is_mine());
        let right_pid = field.cells[4].partition_id();

        // Trigger an inference-breaking update on the LEFT partition only:
        // change (0,0) from Count(1) to Count(0).
        // Left cell (1,0) must change from InferredValue::Mine to InferredValue::Safe,
        // while the untouched RIGHT partition (and cell (3,0)'s Mine inference) remains intact!
        update_minefield(&mut field, ["0"]);
        assert!(field.cells[1].value().is_safe());
        assert!(field.cells[3].value().is_mine());
        assert_eq!(field.cells[4].partition_id(), right_pid);
        assert_eq!(field.inferred_safe(), vec![Point { x: 1, y: 0 }]);
        assert_eq!(field.inferred_mines(), vec![Point { x: 3, y: 0 }]);
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_invalid_resolved_by_editing_other_cell_or_invalid_cell_itself() {
        // 3x1 strip: [(0,0)=Count(0), (1,0)=Unknown, (2,0)=Count(1)]
        // (0,0) is visited first (mine_count = 0), inferring (1,0) is Safe.
        // (2,0) is visited last (mine_count = 1), and index 2 is added to `field.invalid`.
        let mut field = create_minefield(["0.1"], 1);
        assert_eq!(field.invalid(), vec![Point { x: 2, y: 0 }]);

        // Case A: Resolve the inconsistency by editing (0,0) (index 0, NOT index 2!)
        // from Count(0) to Count(1). Index 2 must be removed from `field.invalid`,
        // and (1,0) must now be inferred as Mine.
        update_minefield(&mut field, ["1"]);
        assert!(field.invalid().is_empty());
        assert!(field.cells[1].value().is_mine());
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }

        // Re-introduce the inconsistency sequentially: set (0,0) back to Count(0).
        // Index 2 (highest mine_count) is again recorded in `field.invalid`.
        update_minefield(&mut field, ["0"]);
        assert_eq!(field.invalid(), vec![Point { x: 2, y: 0 }]);
        assert!(field.cells[1].value().is_safe());

        // Case B: Resolve the inconsistency by editing (2,0) (index 2 itself)
        // from Count(1) to Count(0).
        update_minefield(&mut field, ["..0"]);
        assert!(field.invalid().is_empty());
        assert!(field.cells[1].value().is_safe());
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_invalid_sequential_discovery_and_untouched_partition_preservation() {
        // 7x1 strip with two independent partitions:
        // Left:  [(0,0), (1,0), (2,0)]
        // Right: [(4,0), (5,0), (6,0)]
        // Step 1: Set (2,0) = Count(1) first (infers (1,0) = Mine),
        // and set (4,0) = Count(0), (6,0) = Count(1) (marks index 6 invalid).
        let mut field = create_minefield(["..1.0.1"], 2);
        assert!(field.cells[1].value().is_mine());
        assert!(field.cells[6].is_invalid());

        // Step 2: Reveal (0,0) = Count(0) on the left partition.
        // Even though (0,0) was Unknown with InferredValue::None, its Count(0) conflicts
        // with (1,0)'s prior Mine inference, triggering a reset of the left partition so
        // (0,0) (mine_count = 0) is visited first and (2,0) (mine_count = 1,
        // index 2) is added to `invalid`.
        update_minefield(&mut field, ["0"]);

        // Note: There are two equally valid resolutions to this field.
        //
        // One solution:
        // assert!(field.cells[1].value().is_safe());
        // assert!(field.cells[2].is_invalid());
        // assert!(field.cells[6].is_invalid());
        //
        // A second:
        // assert!(field.cells[0].is_invalid());
        // assert!(field.cells[3].value().is_safe());
        // assert!(field.cells[6].is_invalid());

        // Step 3: Resolve ONLY the left partition by setting (1,0) = Mine and (0,0) = Count(1)
        // (so the left side has no Unknown cells left and is completely valid).
        // Index 2 must be removed from `invalid`, while untouched index 6 remains in `invalid`!
        update_minefield(&mut field, ["1*"]);
        assert_eq!(field.invalid(), vec![Point { x: 6, y: 0 }]);
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_repartition_after_basic_solve_splits_via_inferred_mine() {
        // 5x2 board where a bridge is inferred as Mine during basic_solve,
        // splitting a single 10-cell partition into two independent 3-cell partitions:
        // - (2,0)=Count(3) has only 3 unknown neighbors: (1,1), (2,1), (3,1) -> all 3 inferred as Mine!
        // - (1,0)=Count(3) now has 2 inferred mines at (1,1) and (2,1), leaving 1 mine among {(0,0), (0,1)}!
        // - (3,0)=Count(3) now has 2 inferred mines at (2,1) and (3,1), leaving 1 mine among {(4,0), (4,1)}!
        let field = create_minefield([".333.", "....."], 5);

        // (1,1), (2,1), (3,1) are inferred as Mine and removed from partitions (partition_id == 0).
        // (2,0) has no remaining unknown neighbors and gets partition_id == 0.
        // The board splits into two 3-cell partitions:
        // Left:  {(0,0), (0,1), (1,0)}
        // Right: {(3,0), (4,0), (4,1)}
        assert_eq!(field.partitions.len(), 2);
        assert_eq!(field.cells[2].partition_id(), 0);
        for x in 1..=3 {
            let idx = field.cell_idx(Point { x, y: 1 });
            assert!(field.cells[idx].value().is_mine());
            assert_eq!(field.cells[idx].partition_id(), 0);
        }

        let left_pid = field.cells[0].partition_id();
        let right_pid = field.cells[4].partition_id();
        assert_ne!(left_pid, 0);
        assert_ne!(right_pid, 0);
        assert_ne!(left_pid, right_pid);
        assert_eq!(field.partitions[&left_pid].len(), 3);
        assert_eq!(field.partitions[&right_pid].len(), 3);
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    /// Helper asserting that `cells[*].partition_id` and `partitions` agree: every non-zero
    /// partition id on a cell refers to a live partition that lists that cell as a member, and
    /// every member of every live partition carries that partition's id.
    fn assert_partition_invariants(field: &MineField) {
        for cell in field.cells.iter() {
            let pid = cell.partition_id();
            if pid != 0 {
                let partition = field
                    .partitions
                    .get(&pid)
                    .unwrap_or_else(|| panic!("cell {} has stale partition id {pid}", cell.idx()));
                assert!(partition.members().contains(&cell.idx()));
            }
        }
        for (pid, partition) in &field.partitions {
            for &member in partition.members() {
                assert_eq!(field.cells[member].partition_id(), *pid);
            }
        }
    }

    #[test]
    fn test_index_pending_updates_masks_and_logs_neighbors() {
        let mut field = MineField::new(3, 1, 1).unwrap();
        field
            .set_cell_helper(UserCell {
                point: Point { x: 1, y: 0 },
                value: UserCellValue::Safe(1),
            })
            .unwrap();
        assert_eq!(field.pending.len(), 1);

        field.index_pending();

        // pending is drained and the neighbors' masks reflect the new known cell
        assert!(field.pending.is_empty());
        assert_eq!(field.cells[1].value(), CellValue::for_mine_count(1));
        assert!(field.cells[0].unknown_neighbors().is_empty());
        assert!(field.cells[2].unknown_neighbors().is_empty());
        assert_eq!(field.cells[1].unknown_neighbors(), W | E);

        // both neighbors must be queued for repartitioning, whichever log they landed in
        let mut logged: Vec<usize> = field
            .pending_partition
            .iter()
            .chain(field.breaking_pending_partition.iter())
            .copied()
            .collect();
        logged.sort();
        logged.dedup();
        assert_eq!(logged, vec![0, 1, 2]);
    }

    #[test]
    fn test_index_pending_routes_breaking_changes_to_breaking_log() {
        let mut field = MineField::new(3, 1, 1).unwrap();
        // Simulate a previously inferred-safe cell in the middle.
        field.cells[1].set_value(CellValue::INFERRED_SAFE);

        // Confirming the inference (Safe(n)) is non-breaking...
        field
            .set_cell_helper(UserCell {
                point: Point { x: 1, y: 0 },
                value: UserCellValue::Safe(1),
            })
            .unwrap();
        field.index_pending();
        assert!(field.breaking_pending_partition.is_empty());
        assert!(!field.pending_partition.is_empty());
        field.pending_partition.clear();

        // ...whereas contradicting it (Mine) is breaking.
        field.cells[1].set_value(CellValue::INFERRED_SAFE);
        field
            .set_cell_helper(UserCell {
                point: Point { x: 1, y: 0 },
                value: UserCellValue::Mine,
            })
            .unwrap();
        field.index_pending();
        assert!(field.pending_partition.is_empty());
        assert!(!field.breaking_pending_partition.is_empty());
    }

    #[test]
    fn test_partition_invariants_hold_after_merge() {
        let mut field = create_minefield(["1...1", "....."], 2);
        assert_partition_invariants(&field);
        update_minefield(&mut field, ["..2"]);
        assert_eq!(field.partitions.len(), 1);
        assert_partition_invariants(&field);
        // pending/breaking logs are fully drained after a solve pass
        assert!(field.pending.is_empty());
        assert!(field.pending_partition.is_empty());
        assert!(field.breaking_pending_partition.is_empty());
    }

    #[test]
    fn test_partition_split_leaves_no_stale_partition_ids() {
        // Initially one partition {0..=4} bridged through cell 2.
        let mut field = create_minefield([".1.1."], 2);
        assert_eq!(field.partitions.len(), 1);

        // Reverting cell 1 to Unknown leaves only {2, 3, 4} connected.
        update_minefield(&mut field, ["._"]);
        assert_eq!(field.partitions.len(), 1);
        assert_eq!(field.cells[1].partition_id(), 0);
        assert_partition_invariants(&field);
    }

    #[test]
    fn test_partition_dissolution_leaves_no_stale_partition_ids() {
        let mut field = create_minefield([".1."], 1);
        assert_eq!(field.partitions.len(), 1);

        update_minefield(&mut field, ["._"]);
        assert!(field.partitions.is_empty());
        for cell in field.cells.iter() {
            assert_eq!(cell.partition_id(), 0);
        }
    }

    #[test]
    fn test_unchanged_update_does_not_rebuild_partitions() {
        let mut field = create_minefield(["1...1", "....."], 2);
        let before: Vec<u32> = field.cells.iter().map(|c| c.partition_id()).collect();

        update_minefield(&mut field, ["1...1"]);
        let after: Vec<u32> = field.cells.iter().map(|c| c.partition_id()).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn test_solve_queue_with_duplicates() {
        let field = create_minefield(["2.13", "2.1."], 3);
        assert!(field.cells[1].value().is_mine());
        assert!(field.cells[1].value().is_mine());
    }
}

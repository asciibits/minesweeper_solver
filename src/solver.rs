//! Core Minesweeper solver state and partition orchestration.
use cell::{Cell, UserCell, UserCellValue};
use grid::{Neighbor, Point};
use partition::{CellOrdering, Partition};
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    error::Error,
    fmt::{self, Display},
};

use crate::solver::{
    cell::{CellValue, InferredValue},
    grid::{IndexedOffsets, indexed_neighbor_iter},
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
    /// Map of cell indices pending modification.
    pending: Vec<usize>,
    /// Set of non-dirty neighbor cell indices whose adjacency state changed.
    /// Note: Users of this vector must be prepared for duplicates, as the same
    /// neighbor may be adjacent to multiple dirty cells.
    dirty_neighbors: Vec<usize>,
    /// Active connected components of frontier cells, keyed by partition ID.
    partitions: HashMap<u32, Partition>,
    /// TODO
    inferred: HashSet<usize>,
    /// TODO
    invalid: HashSet<usize>,
    /// Monotonically increasing counter used to assign unique partition IDs.
    next_partition_id: u32,

    indexed_offsets: IndexedOffsets,
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
            dirty_neighbors: Vec::with_capacity(perimeter * 2),
            inferred: HashSet::with_capacity(perimeter),
            invalid: HashSet::new(),
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

    /// Returns the [`Cell`] at the given coordinate `p`.
    ///
    /// # Panics
    ///
    /// Panics if `p` is outside the bounds of the field.
    pub fn cell(&self, p: Point) -> UserCell {
        self.cells[self.cell_idx(p)].cell(self.width)
    }

    #[cfg(test)]
    fn cells(&self) -> &[Cell] {
        &self.cells
    }

    /// Updates a single cell's value and records it in `self.dirty` without running the solver.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CoordinatesOutOfBounds`] if `p.x >= self.width` or `p.y >= self.height`.
    fn set_cell_helper(&mut self, p: Point, value: UserCellValue) -> Result<(), SolverError> {
        if p.x >= self.width || p.y >= self.height {
            return Err(SolverError::CoordinatesOutOfBounds {
                p,
                width: self.width,
                height: self.height,
            });
        }
        let cell = &mut self.cells[self.cell_idx(p)];
        if cell.value().to_user_value() != value {
            // Prepare the value to be set in the next solver pass.
            if cell.staged_value().is_none() {
                self.pending.push(cell.idx());
            }
            cell.set_staged_value(Some(value));
        } else {
            cell.set_staged_value(None);
        }
        Ok(())
    }

    /// Sets the [`CellValue`] at coordinate `p` and triggers a solver pass.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CoordinatesOutOfBounds`] if `p` is outside the board dimensions.
    pub fn set_cell_value(&mut self, p: Point, value: UserCellValue) -> Result<(), SolverError> {
        self.set_cell_helper(p, value)?;
        self.solve();
        Ok(())
    }

    /// Updates the board with the given [`Cell`] and triggers a solver pass.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CoordinatesOutOfBounds`] if `cell.point` is outside the board dimensions.
    pub fn set_cell(&mut self, cell: &UserCell) -> Result<(), SolverError> {
        self.set_cell_value(cell.point, cell.value)
    }

    /// Updates a batch of [`Cell`]s and runs a single solver pass after all updates are applied.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CoordinatesOutOfBounds`] if any cell in `cells` lies outside the board.
    pub fn set_cells(
        &mut self,
        cells: impl IntoIterator<Item = UserCell>,
    ) -> Result<(), SolverError> {
        for cell in cells.into_iter() {
            self.set_cell_helper(cell.point, cell.value)?;
        }
        self.solve();
        Ok(())
    }

    pub fn inferred_safe(&self) -> Vec<Point> {
        self.inferred
            .iter()
            .map(|&idx| self.cells[idx])
            .filter(|c| c.inferred_value() == InferredValue::Safe)
            .map(|c| c.point(self.width))
            .collect()
    }

    pub fn inferred_mines(&self) -> Vec<Point> {
        self.inferred
            .iter()
            .map(|&idx| self.cells[idx])
            .filter(|c| c.inferred_value() == InferredValue::Mine)
            .map(|c| c.point(self.width))
            .collect()
    }

    pub fn invalid(&self) -> Vec<Point> {
        self.invalid
            .iter()
            .map(|&idx| self.cells[idx].point(self.width))
            .collect()
    }

    /// Re-indexes modified cells, rebuilds affected frontier partitions, and runs deduction passes.
    fn solve(&mut self) {
        let inference_broken = self.index_pending();

        // Clear existing dirty partitions and repartition
        let start_id = self.next_partition_id;
        self.partition_dirty(inference_broken);

        // Do another partitioning round if any new inferences were found in the
        // basic_solve phase
        if self.basic_solve(start_id) {
            self.partition_dirty(false);
        }
    }

    /// Clears any solver inference on `cell_idx` and restores its neighbors' `inferred_*_neighbors` masks.
    fn clear_cell_inference(
        cells: &mut [Cell],
        inferred: &mut HashSet<usize>,
        indexed_offsets: &IndexedOffsets,
        cell_idx: usize,
    ) {
        let cell = &mut cells[cell_idx];
        let inferred_state = cell.inferred_value();
        if inferred_state != InferredValue::None {
            cell.set_inferred_value(InferredValue::None);
            inferred.remove(&cell_idx);

            for Neighbor { neighbor, from } in
                indexed_neighbor_iter(cell_idx, cell.neighbors(), indexed_offsets)
            {
                let neighbor_cell = &mut cells[neighbor];
                neighbor_cell.set_unknown(from);
            }
        }
    }

    fn change_value(
        cells: &mut [Cell],
        idx: usize,
        value: UserCellValue,
        inferred: bool,
        indexed_offsets: &IndexedOffsets,
    ) {
        let cell = &mut cells[idx];

        for Neighbor {
            neighbor: n_idx,
            from,
        } in indexed_neighbor_iter(idx, cell.neighbors(), indexed_offsets)
        {
            let mut n_cell = cells[n_idx];
            match value {
                UserCellValue::Unknown => n_cell.set_unknown(from),
                UserCellValue::Safe(_) => n_cell.set_safe(from, inferred),
                UserCellValue::Mine => n_cell.set_mine(from, inferred),
            }
        }
    }

    /// Updates neighbor masks (`unknown_neighbors`, `safe_neighbors`, `mine_neighbors`)
    /// for all cells in `pending` and their immediate neighbors.
    ///
    /// Returns `true` if any inference was broken, `false` otherwise.
    fn index_pending(&mut self) -> bool {
        let mut inference_breaking = false;
        for &pending_idx in &self.pending {
            let dirty_cell = self.cells[pending_idx];
            let Some(new_value) = dirty_cell.staged_value() else {
                continue;
            };
            let new_value = new_value.into();

            let prev_value = dirty_cell.value();
            if prev_value == new_value {
                continue;
            }

            // Transitioning from Safe or Mine breaks previous inference passes,
            // as does a direct violation of existing inferences.
            inference_breaking = inference_breaking
                || prev_value != CellValue::UNKNOWN
                || match dirty_cell.inferred_value() {
                    InferredValue::Mine => !new_value.is_mine(),
                    InferredValue::Safe => !new_value.is_safe(),
                    InferredValue::None => false,
                };

            // If this cell previously held an inference, revert its inferred contribution
            // on all neighbors before applying its new real value.
            Self::clear_cell_inference(
                &mut self.cells,
                &mut self.inferred,
                &self.indexed_offsets,
                pending_idx,
            );

            let count_to_count = matches!(
                (prev_value.mine_count(), new_value.mine_count()),
                (Some(_), Some(_))
            );

            for Neighbor {
                neighbor: neighbor_idx,
                from,
            } in
                indexed_neighbor_iter(pending_idx, dirty_cell.neighbors(), &self.indexed_offsets)
            {
                let neighbor_cell = &mut self.cells[neighbor_idx];
                if !count_to_count {
                    match new_value {
                        CellValue::UNKNOWN => neighbor_cell.set_unknown(from),
                        CellValue::MINE => neighbor_cell.set_mine(from, false),
                        _ => neighbor_cell.set_safe(from, false),
                    };
                }
                if neighbor_cell.staged_value().is_none() {
                    // Note: The same neighbor may be adjacent to multiple dirty
                    // cells, so we may index it multiple times. Future users of
                    // the dirty_neighbors must be prepared for duplicates.
                    self.dirty_neighbors.push(neighbor_cell.idx());
                }
            }

            let dirty_cell = &mut self.cells[pending_idx];
            dirty_cell.set_value(new_value);
        }

        // clear all staged values now that they have been applied
        for &pending_idx in &self.pending {
            self.cells[pending_idx].set_staged_value(None);
        }

        // If any newly updated cell or its immediate neighbor is now impossible to satisfy
        // under existing inferences, treat this update as inference-breaking so the affected
        // partition is reset and re-evaluated from lowest to highest mine_count.
        if !inference_breaking {
            inference_breaking =
                self.pending
                    .iter()
                    .chain(self.dirty_neighbors.iter())
                    .any(|&idx| {
                        CellOrdering::from_cell(&self.cells[idx])
                            .is_some_and(|ord| ord.solution_count() == 0)
                    });
        }

        inference_breaking
    }

    /// Invalidates partitions touched by `dirty` or `dirty_neighbors` and constructs
    /// new connected [`Partition`]s from the updated frontier.
    fn partition_dirty(&mut self, reset_inferred: bool) {
        for &cell_idx in self.pending.iter().chain(self.dirty_neighbors.iter()) {
            let pid = self.cells[cell_idx].partition_id();
            if pid != 0 {
                self.partitions.remove(&pid);
                self.cells[cell_idx].set_partition_id(0);
            }
            if reset_inferred {
                Self::clear_cell_inference(
                    &mut self.cells,
                    &mut self.inferred,
                    &self.indexed_offsets,
                    cell_idx,
                );
                self.invalid.remove(&cell_idx);
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
            if cell.partition_id() == 0 {
                let partition_id = self.next_partition_id;
                if let Some(partition) = Partition::create(
                    &mut self.cells,
                    self.width,
                    self.height,
                    &self.indexed_offsets,
                    cell_idx,
                    partition_id,
                    reset_inferred,
                ) {
                    if reset_inferred {
                        // clear out all inferences from this partition,
                        // updating all neighbors as needed
                        for &member_idx in partition.members() {
                            Self::clear_cell_inference(
                                &mut self.cells,
                                &mut self.inferred,
                                &self.indexed_offsets,
                                member_idx,
                            );
                            self.invalid.remove(&member_idx);
                        }
                    }
                    self.partitions.insert(partition_id, partition);
                    self.next_partition_id += 1;
                } else if let Some(ord) = CellOrdering::from_cell(&self.cells[cell_idx]) {
                    if ord.solution_count() == 0 {
                        self.invalid.insert(cell_idx);
                    }
                }
            }
        }

        self.pending.clear();
        self.dirty_neighbors.clear();
    }

    /// Performs local constraint deductions on the newly created partitions.
    ///
    /// Populates `self.pending` with newly inferred cells and `self.dirty_neighbors`
    /// with their neighbors, returning `true` if at least one cell was inferred.
    fn basic_solve(&mut self, start_partition_id: u32) -> bool {
        // Find all basic logic solutions
        let mut sorted_knowns = BTreeSet::new();

        // work on all partitions simultaneously - load all knowns into the
        // ordered set. As long as the minimum has 1 or 0 solutions, keep
        // popping them off the set and processing their neighbors. As neighbors
        // are processed, first remove them from the set, make the changes, then
        // re-add them.
        for partition_id in start_partition_id..self.next_partition_id {
            let partition = self.partitions.get_mut(&partition_id).unwrap();

            for &cell_idx in partition.members() {
                let cell = &self.cells[cell_idx];
                if let Some(ordered_cell) = CellOrdering::from_cell(cell) {
                    sorted_knowns.insert(ordered_cell);
                }
            }
        }

        while let Some(solve_cell_ord) = sorted_knowns.pop_first() {
            if solve_cell_ord.solution_count() > 1 {
                // We have exhausted the easy deductions for this partition
                sorted_knowns.insert(solve_cell_ord);
                break;
            }

            let solve_cell = self.cells[solve_cell_ord.cell_idx()];

            if solve_cell_ord.solution_count() == 0 {
                // This is indicates an invalid board was presented. This is OK,
                // we will still attempt to solve whatever we can, and report
                // the places where solves were not possible.
                self.invalid.insert(solve_cell.idx());
            }

            // two options here: either the remaining_mines are all in
            // the remaining_neighbors, or none of them are
            let inferred_state = if solve_cell_ord.remaining_mines() > 0 {
                InferredValue::Mine
            } else {
                InferredValue::Safe
            };

            for Neighbor {
                neighbor: inferred_idx,
                ..
            } in indexed_neighbor_iter(
                solve_cell_ord.cell_idx(),
                solve_cell.inferred_unknown_neighbors(),
                &self.indexed_offsets,
            ) {
                let inferred_cell = &mut self.cells[inferred_idx];
                #[cfg(debug_assertions)]
                let inferred_partition = inferred_cell.partition_id();
                inferred_cell.set_inferred_value(inferred_state);
                self.inferred.insert(inferred_idx);
                self.pending.push(inferred_idx);

                for Neighbor {
                    neighbor: neighbor_idx,
                    from,
                } in indexed_neighbor_iter(
                    inferred_idx,
                    inferred_cell.neighbors(),
                    &self.indexed_offsets,
                ) {
                    let neighbor_cell = &mut self.cells[neighbor_idx];
                    if neighbor_cell.partition_id() != 0 && neighbor_cell.is_safe(false) {
                        debug_assert_eq!(neighbor_cell.partition_id(), inferred_partition);
                        self.dirty_neighbors.push(neighbor_idx);
                    }
                    let prev_cell_ord = CellOrdering::from_cell(neighbor_cell);
                    if inferred_state == InferredValue::Mine {
                        neighbor_cell.set_mine(from, true);
                    } else {
                        neighbor_cell.set_safe(from, true);
                    }
                    if let Some(prev_ord_cell) = prev_cell_ord {
                        if sorted_knowns.remove(&prev_ord_cell) {
                            if let Some(new_cell_ord) = CellOrdering::from_cell(neighbor_cell) {
                                sorted_knowns.insert(new_cell_ord);
                            }
                        }
                    }
                }
            }
        }

        !self.pending.is_empty()
    }

    /// Get the cell index for the given x/y coordinates.
    fn cell_idx(&self, p: Point) -> usize {
        p.to_index(self.width)
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
        write!(
            f,
            "pending: [{}],{spacer}",
            self.0
                .pending
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        )?;
        write!(
            f,
            "dirty_neighbors: [{}],{spacer}",
            self.0
                .dirty_neighbors
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<String>>()
                .join(", ")
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
        write!(f, "MineField: [{spacer}\"")?;
        for y in 0..self.height {
            for x in 0..self.width {
                let cell = self.cell(Point { x, y });
                let char = match cell.value {
                    UserCellValue::Mine => '*',
                    UserCellValue::Unknown => '.',
                    UserCellValue::Safe(n) => n.to_string().chars().next().unwrap(),
                };
                write!(f, "{char}")?;
            }
            write!(f, "\",{spacer}")?;
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
        assert!((cell.inferred_unknown_neighbors() & cell.inferred_safe_neighbors()).is_empty());
        assert!((cell.inferred_unknown_neighbors() & cell.inferred_mine_neighbors()).is_empty());
        assert!((cell.inferred_safe_neighbors() & cell.inferred_mine_neighbors()).is_empty());

        // inferred set covers all neighbors
        assert_eq!(
            (cell.inferred_safe_neighbors()
                | cell.inferred_mine_neighbors()
                | cell.inferred_unknown_neighbors()),
            cell.neighbors()
        );

        // inferred safe and mines are a super-set of standard versions
        assert_eq!(
            cell.safe_neighbors() | cell.inferred_safe_neighbors(),
            cell.inferred_safe_neighbors()
        );
        assert_eq!(
            cell.mine_neighbors() | cell.inferred_mine_neighbors(),
            cell.inferred_mine_neighbors()
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
            .set_cell_value(Point { x: 3, y: 0 }, UserCellValue::Safe(1))
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
            .set_cell(&UserCell {
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
        assert_eq!(CellOrdering::from_cell(&c00).unwrap().remaining_mines(), 2);
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

        // Mark (1, 0) as a Mine: (0, 0) moves E from unknown_neighbors to mine_neighbors (remaining_mines == 1)
        update_minefield(&mut field, [".*"]);
        let c00 = field.cells[field.cell_idx(Point { x: 0, y: 0 })];
        assert_eq!(c00.mine_neighbors(), !!E);
        assert!(c00.safe_neighbors().is_empty());
        assert_eq!(c00.unknown_neighbors(), S | SE);
        assert_eq!(CellOrdering::from_cell(&c00).unwrap().remaining_mines(), 1);
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
        assert!(c00.mine_neighbors().is_empty());
        assert_eq!(c00.safe_neighbors(), !!E);
        assert_eq!(c00.unknown_neighbors(), S | SE);
        assert_eq!(c00.inferred_mine_neighbors(), S | SE);
        assert_eq!(CellOrdering::from_cell(&c00).unwrap().remaining_mines(), 0);
        assert_neighbor_invariants(&c00);

        // Revert (1, 0) back to Unknown (inference-breaking): (0, 0) moves E from safe_neighbors
        // back to unknown_neighbors, and the inferences on S and SE are cleared!
        update_minefield(&mut field, ["._"]);
        let c00 = field.cells[field.cell_idx(Point { x: 0, y: 0 })];
        assert!(c00.mine_neighbors().is_empty());
        assert!(c00.safe_neighbors().is_empty());
        assert_eq!(c00.unknown_neighbors(), E | S | SE);
        assert!(c00.inferred_mine_neighbors().is_empty());
        assert_eq!(CellOrdering::from_cell(&c00).unwrap().remaining_mines(), 2);
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
        assert_eq!(field.cells[2].safe_neighbors(), W | E);

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
        assert_eq!(field.cells[2].inferred_value(), InferredValue::Safe);
        assert_eq!(
            CellOrdering::from_cell(&field.cells[1])
                .unwrap()
                .remaining_mines(),
            0
        );
        assert_eq!(field.cells[1].unknown_neighbors(), !!E);
        assert!(field.cells[1].inferred_unknown_neighbors().is_empty());
        assert!(field.cells[1].safe_neighbors().is_empty());
        assert_eq!(field.cells[1].mine_neighbors(), !!W);

        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_batch_adjacent_cells_and_duplicate_dirty_neighbors() {
        // 3x2 board: reveal (1, 0) as Count(2) first
        let mut field = create_minefield([".2.", "..."], 2);
        assert_eq!(
            CellOrdering::from_cell(&field.cells[1])
                .unwrap()
                .remaining_mines(),
            2
        );

        // In one batch, set both (0, 0) and (2, 0) to Mine, and (1, 1) to Count(2).
        // (1, 0) is a non-dirty neighbor of all three pending cells, so it appears 3 times
        // in `dirty_neighbors`.
        update_minefield(&mut field, ["*.*", ".2."]);

        // Both (1, 0) and (1, 1) are adjacent to the two mines at (0, 0) and (2, 0),
        // so both have remaining_mines == 0, basic_solve infers (0, 1) and (2, 1) as Safe,
        // and the post-basic_solve repartition pass dissolves the solved partition.
        let c10 = field.cells[field.cell_idx(Point { x: 1, y: 0 })];
        let c11 = field.cells[field.cell_idx(Point { x: 1, y: 1 })];
        assert_eq!(CellOrdering::from_cell(&c10).unwrap().remaining_mines(), 0);
        assert_eq!(CellOrdering::from_cell(&c11).unwrap().remaining_mines(), 0);

        assert_eq!(c10.mine_neighbors(), W | E);
        assert_eq!(c10.safe_neighbors(), !!S);
        assert_eq!(c10.unknown_neighbors(), SW | SE);
        assert!(c10.inferred_unknown_neighbors().is_empty());

        assert_eq!(c11.mine_neighbors(), NW | NE);
        assert_eq!(c11.safe_neighbors(), !!N);
        assert_eq!(c11.unknown_neighbors(), W | E);
        assert!(c11.inferred_unknown_neighbors().is_empty());

        assert!(field.partitions.is_empty());
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_repeated_and_reverted_staging_on_same_cell() {
        let mut field = create_minefield([".1."], 1);
        assert_eq!(
            CellOrdering::from_cell(&field.cells[1])
                .unwrap()
                .remaining_mines(),
            1
        );

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
        assert_eq!(field.cells[1].unknown_neighbors(), !!E);
        assert!(field.cells[1].safe_neighbors().is_empty());
        assert_eq!(
            CellOrdering::from_cell(&field.cells[1])
                .unwrap()
                .remaining_mines(),
            0
        );
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

        assert_eq!(field.cells[1].inferred_value(), InferredValue::Mine);
        assert_eq!(field.cells[3].inferred_value(), InferredValue::Safe);

        // Verify inferred neighbor sets reflect the deductions
        assert_eq!(field.cells[0].inferred_mine_neighbors(), !!E);
        assert!(field.cells[0].inferred_unknown_neighbors().is_empty());
        assert_eq!(field.cells[2].inferred_mine_neighbors(), !!W);
        assert_eq!(field.cells[2].inferred_safe_neighbors(), !!E);
        assert!(field.cells[2].inferred_unknown_neighbors().is_empty());

        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_basic_solve_detects_impossible_contradictory_clues() {
        // 3x1 strip: [(0,0)=Count(0), (1,0)=Unknown, (2,0)=Count(1)]
        // (0,0)=Count(0) infers (1,0) is Safe, leaving (2,0)=Count(1) with 0 available cells
        // and 1 remaining mine (solution_count == 0 -> recorded in partition.invalid).
        let field = create_minefield(["0.1"], 1);

        assert_eq!(field.cells[1].inferred_value(), InferredValue::Safe);
        assert_eq!(field.invalid.len(), 1);
        assert!(field.invalid.contains(&2));
        // assert_eq!(field.invalid, &[2]);
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
        assert_eq!(field.cells[1].inferred_value(), InferredValue::Mine);
        assert_eq!(field.cells[2].inferred_mine_neighbors(), !!W);

        // Round 2 (inference-compatible): Reveal (2,0) as Count(1).
        // Because (1,0)'s Mine inference is preserved, (2,0) immediately knows its 1 mine is at W
        // and infers (3,0) is Safe!
        update_minefield(&mut field, ["..1"]);
        assert_eq!(field.cells[1].inferred_value(), InferredValue::Mine);
        assert_eq!(field.cells[3].inferred_value(), InferredValue::Safe);
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_inference_breaking_updates_clear_and_recompute_inferences() {
        // 3x1 strip: [(0,0), (1,0), (2,0)]
        // Round 1: Reveal (1,0) as Count(2) -> both (0,0) and (2,0) are inferred as Mine.
        let mut field = create_minefield([".2."], 2);
        assert_eq!(field.cells[0].inferred_value(), InferredValue::Mine);
        assert_eq!(field.cells[2].inferred_value(), InferredValue::Mine);

        // Round 2 (Inference-breaking Case A): Change (1,0) from Count(2) to Count(1).
        // Neither (0,0) nor (2,0) can be inferred anymore; both must reset to InferredState::None!
        update_minefield(&mut field, [".1"]);
        assert_eq!(field.cells[0].inferred_value(), InferredValue::None);
        assert_eq!(field.cells[2].inferred_value(), InferredValue::None);
        assert_eq!(field.cells[1].inferred_unknown_neighbors(), W | E);
        assert!(field.cells[1].inferred_mine_neighbors().is_empty());
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }

        // Round 3: Change (1,0) to Count(0) -> both (0,0) and (2,0) are inferred as Safe.
        update_minefield(&mut field, [".0"]);
        assert_eq!(field.cells[0].inferred_value(), InferredValue::Safe);
        assert_eq!(field.cells[2].inferred_value(), InferredValue::Safe);

        // Round 4 (Inference-breaking Case B): Directly violate inference on (0,0) by setting it to Mine,
        // and simultaneously change (1,0) to Count(1).
        // (2,0) must remain/be recomputed as Safe, and all neighbor invariants must hold.
        update_minefield(&mut field, ["*1"]);
        assert_eq!(field.cells[0].inferred_value(), InferredValue::None);
        assert_eq!(field.cells[2].inferred_value(), InferredValue::Safe);
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }

        // Round 5 (Inference-breaking Case C): Dissolve the partition completely by resetting
        // (0,0) and (1,0) to Unknown. Verify (2,0)'s Safe inference is cleared even though no
        // new partition is created!
        update_minefield(&mut field, ["__"]);
        assert!(field.partitions.is_empty());
        for cell in &field.cells {
            assert_eq!(cell.inferred_value(), InferredValue::None);
            assert_eq!(cell.inferred_unknown_neighbors(), cell.unknown_neighbors());
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
        assert_eq!(field.cells[1].inferred_value(), InferredValue::Mine);
        assert_eq!(field.cells[3].inferred_value(), InferredValue::Mine);
        let right_pid = field.cells[4].partition_id();

        // Trigger an inference-breaking update on the LEFT partition only:
        // change (0,0) from Count(1) to Count(0).
        // Left cell (1,0) must change from InferredValue::Mine to InferredValue::Safe,
        // while the untouched RIGHT partition (and cell (3,0)'s Mine inference) remains intact!
        update_minefield(&mut field, ["0"]);
        assert_eq!(field.cells[1].inferred_value(), InferredValue::Safe);
        assert_eq!(field.cells[3].inferred_value(), InferredValue::Mine);
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
        assert_eq!(field.invalid, HashSet::from([2]));
        assert_eq!(field.invalid(), vec![Point { x: 2, y: 0 }]);

        // Case A: Resolve the inconsistency by editing (0,0) (index 0, NOT index 2!)
        // from Count(0) to Count(1). Index 2 must be removed from `field.invalid`,
        // and (1,0) must now be inferred as Mine.
        update_minefield(&mut field, ["1"]);
        assert!(field.invalid.is_empty());
        assert_eq!(field.cells[1].inferred_value(), InferredValue::Mine);
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }

        // Re-introduce the inconsistency sequentially: set (0,0) back to Count(0).
        // Index 2 (highest mine_count) is again recorded in `field.invalid`.
        update_minefield(&mut field, ["0"]);
        assert_eq!(field.invalid, HashSet::from([2]));
        assert_eq!(field.cells[1].inferred_value(), InferredValue::Safe);

        // Case B: Resolve the inconsistency by editing (2,0) (index 2 itself)
        // from Count(1) to Count(0).
        update_minefield(&mut field, ["..0"]);
        assert!(field.invalid.is_empty());
        assert_eq!(field.cells[1].inferred_value(), InferredValue::Safe);
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
        assert_eq!(field.cells[1].inferred_value(), InferredValue::Mine);
        assert_eq!(field.invalid, HashSet::from([6]));

        // Step 2: Reveal (0,0) = Count(0) on the left partition.
        // Even though (0,0) was Unknown with InferredValue::None, its Count(0) conflicts
        // with (1,0)'s prior Mine inference, triggering a reset of the left partition so
        // (0,0) (mine_count = 0) is visited first and (2,0) (mine_count = 1, index 2) is added to `invalid`.
        update_minefield(&mut field, ["0"]);
        assert_eq!(field.cells[1].inferred_value(), InferredValue::Safe);
        assert_eq!(field.invalid, HashSet::from([2, 6]));

        // Step 3: Resolve ONLY the left partition by setting (1,0) = Mine and (0,0) = Count(1)
        // (so the left side has no Unknown cells left and is completely valid).
        // Index 2 must be removed from `invalid`, while untouched index 6 remains in `invalid`!
        update_minefield(&mut field, ["1*"]);
        assert_eq!(field.invalid, HashSet::from([6]));
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
            assert_eq!(field.cells[idx].inferred_value(), InferredValue::Mine);
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
}

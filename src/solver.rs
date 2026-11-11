//! Core Minesweeper solver state and partition orchestration.
use cell::{Cell, UserCell, UserCellValue};
use grid::Point;
use std::{
    cmp::max,
    collections::HashMap,
    error::Error,
    fmt::{self, Display},
};

#[cfg(feature = "debug")]
use crate::solver::display::DetailMineField;
use crate::{
    InferredValue,
    solver::{
        cell::{
            CellIdx, CellType, CellValue, ChangeStatus, FastNumberHasherBuilder, FlagMask,
            GuardedQueue, IdxAndCell, UserCellWithInference, UserValueWithInference, cell_idx,
            coordinates,
        },
        grid::{DirectionSet, Neighbor},
        metrics::Metrics,
    },
};

pub(crate) mod cell;
#[cfg(any(test, feature = "debug"))]
mod display;
pub(crate) mod grid;
mod metrics;
mod stats;
pub(crate) mod test_utils;

/// Represents the state of a Minesweeper board and its active solving partitions.
#[derive(Debug, Clone)]
pub struct MineField {
    /// Width of the board in cells.
    width: u32,
    /// Height of the board in cells.
    height: u32,
    /// Total number of mines on the board.
    mine_count: Option<u32>,
    /// Row-major contiguous array of all board cells and their precomputed solver metadata.
    cells: Box<[Cell]>,
    /// Active board partitions, mapping a representative seed cell index to the partition size.
    partitions: HashMap<CellIdx, usize, FastNumberHasherBuilder>,
    /// Next unique ID to assign to a newly walked partition.
    next_partition_id: u16,
    /// User-supplied cell updates waiting to be applied by [`MineField::index_pending`].
    pending: Vec<IdxAndCell>,
    /// Queue of seed cells whose breaking changes require inferences in their connected components to be reset.
    desolved_queue: GuardedQueue,
    /// Queue of "count" cells (guaranteed to satisfy [`CellValue::has_count`]) whose neighbor values have become solvable.
    solved_queue: GuardedQueue,
    /// Queue of "count" cells (guaranteed to satisfy [`CellValue::has_count`]) whose connected frontier needs to be partitioned or repartitioned.
    pending_partition_queue: GuardedQueue,
    /// Performance and step counters recorded during solving.
    metrics: Metrics,
    /// Cell indices whose inference changed during the most recent solve cycle.
    recent_inferred: Vec<CellIdx>,
    /// Contradictory "count" cells discovered during the most recent solve cycle.
    recent_invalid: Vec<CellIdx>,
}

const MAX_SIZE: usize = if usize::MAX > u32::MAX as usize {
    usize::MAX >> 9
} else {
    u32::MAX as usize
};

impl MineField {
    /// Creates a new [`MineField`] with the given dimensions and if provided, the total
    /// `mine_count`.
    ///
    /// All cells are initialized to [`crate::CellValue::Unknown`].
    ///
    /// # Errors
    ///
    /// * Returns [`SolverError::AreaTooLarge`] if `width * height` exceeds `usize::MAX >> 9`.
    /// * Returns [`SolverError::MineCountTooLarge`] if `mine_count >= width * height`.
    pub fn new(width: u32, height: u32) -> Result<MineField, SolverError> {
        Self::new_helper(width, height, None)
    }
    pub fn with_mine_count(
        width: u32,
        height: u32,
        mine_count: u32,
    ) -> Result<MineField, SolverError> {
        Self::new_helper(width, height, Some(mine_count))
    }
    fn new_helper(
        width: u32,
        height: u32,
        mine_count: Option<u32>,
    ) -> Result<MineField, SolverError> {
        let area = width as usize * height as usize;
        let perimeter = (width as usize + height as usize) * 2;
        if area > MAX_SIZE {
            return Err(SolverError::AreaTooLarge { width, height });
        }
        if area == 0 {
            return Err(SolverError::AreaTooSmall);
        }
        if let Some(mine_count) = mine_count
            && mine_count as usize >= area
        {
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
            partitions: HashMap::with_capacity_and_hasher(16, FastNumberHasherBuilder),
            next_partition_id: 1,
            pending: Vec::with_capacity(area / 4),
            desolved_queue: GuardedQueue::with_capacity(0, FlagMask::DESOLVED_QUEUE),
            solved_queue: GuardedQueue::with_capacity(perimeter, FlagMask::SOLVED_QUEUE),
            pending_partition_queue: GuardedQueue::with_capacity(
                area / 4,
                FlagMask::PENDING_PARTITION,
            ),
            metrics: Metrics::default(),
            recent_inferred: Vec::with_capacity(perimeter / 2),
            recent_invalid: Vec::new(),
        };

        Ok(field)
    }

    /// Creates a new [`MineField`] with dimensions inferred from `field_spec` and populates its initial cells.
    ///
    /// The board's `width` and `height` are determined by the maximum `x + 1` and `y + 1` coordinates
    /// encountered across all rows in `field_spec` (including `'.'` and `' '` characters).
    /// Leading whitespace on each row is trimmed, so use `'.'` to represent a leading unspecified cell.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minesweeper_solver::{MineField, Point, CellValue};
    /// // Creates a 4x2 MineField with 1 total mine:
    /// // - Safe(1) at (1, 0)
    /// // - Mine at (2, 0)
    /// // - Safe(1) at (3, 1)
    /// // - Unknown everywhere else
    /// let field = MineField::create(
    ///     ".1*.
    ///      ...1",
    /// );
    /// assert_eq!(field.width(), 4);
    /// assert_eq!(field.height(), 2);
    /// assert_eq!(field.cell(Point { x: 1, y: 0 }).value, CellValue::Safe(1));
    /// assert_eq!(field.cell(Point { x: 2, y: 0 }).value, CellValue::Mine);
    /// assert_eq!(field.cell(Point { x: 3, y: 1 }).value, CellValue::Safe(1));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `field_spec` contains any character outside `'0'..='8'`, `'*'`, `'x'`, `'#'`, `'_'`, `'.'`, `' '`, or `'\n'`.
    /// - The inferred board dimensions and `mine_count` fail [`MineField::new`] validation (e.g., an empty
    ///   spec or `mine_count >= width * height`).
    #[track_caller]
    pub fn create(field_spec: &str) -> MineField {
        let (width, height, cell_iter) = Self::cells_and_size_from_spec(field_spec);

        // TODO: implement mine count into the spec
        let mut field = MineField::new(width, height).unwrap();
        field.pending.extend(cell_iter);
        field.solve();

        field
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
    pub fn mine_count(&self) -> Option<u32> {
        self.mine_count
    }

    /// Returns the [`crate::CellWithInference`] at the given coordinate `p`.
    pub fn cell(&self, p: Point) -> UserValueWithInference {
        self.cells[self.cell_idx(p)].value().into()
    }

    /// Applies a batch of cell updates to an existing [`MineField`] using the ASCII `field_spec` format.
    ///
    /// Coordinates with `'.'` or `' '` are filtered out and left unmodified on `field`. Coordinates
    /// with `'_'`, `'x'`, or `'#'` are explicitly set to
    /// [`crate::CellValue::Unknown`]. (Typically `'_'` would be used, but
    /// `'x'` and `'#'` are output from [`MineField`] display formatting indicating "inferred
    /// mine" and "inferred safe" respectively).
    ///
    /// # Examples
    ///
    /// ```
    /// # use minesweeper_solver::{MineField, Point, CellValue};
    /// let mut field = MineField::create(
    ///     "1..1
    ///      ....",
    /// );
    ///
    /// // Reveal (1, 0) as Safe(2) and flag (2, 1) as a Mine, leaving all '.' cells untouched:
    /// field.update(
    ///     ".2
    ///      ..*",
    /// );
    /// assert_eq!(field.cell(Point { x: 0, y: 0 }).value, CellValue::Safe(1));
    /// assert_eq!(field.cell(Point { x: 1, y: 0 }).value, CellValue::Safe(2));
    /// assert_eq!(field.cell(Point { x: 2, y: 1 }).value, CellValue::Mine);
    ///
    /// // Revert (1, 0) back to CellValue::Unknown using '_':
    /// field.update("._");
    /// assert_eq!(field.cell(Point { x: 1, y: 0 }).value, CellValue::Unknown);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `field_spec` contains an invalid character.
    /// - Any non-skipped coordinate in `field_spec` lies outside `field`'s dimensions.
    #[track_caller]
    pub fn update(&mut self, field_spec: &str) {
        self.pending
            .extend(Self::cells_from_spec(field_spec, self.width));
        self.solve();
    }

    /// Updates the board with the given [`crate::Cell`] and triggers a solver pass.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CoordinatesOutOfBounds`] if `cell.point` is outside the board dimensions.
    #[track_caller]
    pub fn set_cell(&mut self, cell: UserCell) -> Result<(), SolverError> {
        self.set_cells([cell])
    }

    /// Updates a batch of [`crate::Cell`]s and runs a single solver pass after all updates are applied.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CoordinatesOutOfBounds`] if any cell in `cells` lies outside the board.
    pub fn set_cells(
        &mut self,
        cells: impl IntoIterator<Item = UserCell>,
    ) -> Result<(), SolverError> {
        for cell in cells.into_iter() {
            if cell.point.x >= self.width || cell.point.y >= self.height {
                return Err(SolverError::CoordinatesOutOfBounds {
                    p: cell.point,
                    width: self.width,
                    height: self.height,
                });
            }
            self.pending.push(IdxAndCell {
                idx: self.cell_idx(cell.point),
                value: cell.value.into(),
            });
        }

        self.pending.sort();
        self.pending.dedup();
        self.solve();
        Ok(())
    }

    fn cell_iterator(&self) -> impl Iterator<Item = (CellIdx, Cell)> {
        self.cells
            .iter()
            .enumerate()
            .map(|(i, &v)| (CellIdx(i as u32), v))
    }

    /// Provide a user view into the cell array. Note: There is no direct, fast access to all
    /// "inferred" cells. This is intentional; For fast access, refer to `recent_inferred` after
    /// each solve pass.
    pub fn cells(&self) -> impl Iterator<Item = UserCellWithInference> {
        self.cell_iterator().map(|(i, c)| {
            let v: UserValueWithInference = c.value().into();
            UserCellWithInference {
                cell: UserCell {
                    point: coordinates(i, self.width),
                    value: v.value,
                },
                inferred: v.inferred,
            }
        })
    }

    pub fn recent_inferred(&self) -> impl Iterator<Item = Point> {
        self.recent_inferred
            .iter()
            .map(|&i| coordinates(i, self.width))
    }

    #[cfg(test)]
    pub(self) fn recent_inferred_cells(&self) -> impl Iterator<Item = CellIdx> {
        self.recent_inferred.iter().copied()
    }

    #[cfg(test)]
    pub(self) fn recent_invalid_cells(&self) -> impl Iterator<Item = CellIdx> {
        self.recent_invalid.iter().copied()
    }

    #[cfg(test)]
    pub(self) fn recent_inferred_mine_cells(&self) -> impl Iterator<Item = CellIdx> {
        self.recent_inferred_cells()
            .filter(|&i| self.cells[i].value().is_mine())
    }

    #[cfg(test)]
    pub(self) fn recent_inferred_safe_cells(&self) -> impl Iterator<Item = CellIdx> {
        self.recent_inferred_cells()
            .filter(|&i| self.cells[i].value().is_safe())
    }

    /// Updates a cell's value and synchronizes neighbor sets, returning a [`ChangeStatus`].
    ///
    /// This function is optimized for typical forward gameplay (non-breaking changes):
    /// - `Unknown -> Mine / InferredMine`: [`Self::unknown_to_mine`]
    /// - `Unknown -> InferredSafe`: [`Self::unknown_to_inferred_safe`]
    /// - `Unknown -> Count`: [`Self::unknown_to_count`]
    /// - `InferredSafe -> Count`: [`Self::inferred_safe_to_count`]
    ///
    /// These transitions update directional neighbor masks incrementally without invalidating
    /// partitions, returning [`ChangeStatus::NonBreaking`].
    ///
    /// Retractions (e.g. cell returning to `Unknown`) or conflicting changes route to
    /// [`Self::inference_breaking_change`], returning [`ChangeStatus::Breaking`].
    fn change_cell_value(&mut self, idx: CellIdx, new_value: CellValue) -> ChangeStatus {
        let prev_value = self.cells[idx].value();

        match prev_value.cell_type() {
            CellType::Unknown => match new_value.cell_type() {
                CellType::Unknown => ChangeStatus::NoChange,
                CellType::InferredMine | CellType::Mine => self.unknown_to_mine(idx, new_value),
                CellType::InferredSafe => self.unknown_to_inferred_safe(idx, new_value),
                CellType::Count => self.unknown_to_count(idx, new_value),
            },
            CellType::InferredMine | CellType::Mine => match new_value.cell_type() {
                CellType::InferredMine | CellType::Mine => {
                    self.cells[idx].set_value(new_value);
                    ChangeStatus::NoChange
                }
                _ => self.inference_breaking_change(idx, prev_value, new_value),
            },
            CellType::InferredSafe => match new_value.cell_type() {
                CellType::InferredSafe => ChangeStatus::NoChange,
                CellType::Count => self.inferred_safe_to_count(idx, new_value),
                _ => self.inference_breaking_change(idx, prev_value, new_value),
            },
            CellType::Count => match new_value.cell_type() {
                CellType::Count => {
                    if prev_value == new_value {
                        ChangeStatus::NoChange
                    } else {
                        // count -> count (diff counts) - it could be de-solved, or solved
                        self.cells[idx].set_value(new_value);
                        self.desolved_queue.push(&mut self.cells, idx);
                        self.solved_queue.push(&mut self.cells, idx);
                        ChangeStatus::Breaking
                    }
                }
                _ => self.inference_breaking_change(idx, prev_value, new_value),
            },
        }
    }

    fn unknown_to_mine(&mut self, idx: CellIdx, new_value: CellValue) -> ChangeStatus {
        // alt (count):  +mine, -alt, +solve
        // safe: +mine, -alt
        // mine: n/a
        // unknown: +mine
        // self: n/a
        let neighbors = self.cells[idx].alt_neighbors();
        for neighbor in Cell::neighbor_iter(idx, self.width, neighbors) {
            let cell = &mut self.cells[neighbor.idx];
            *cell.mine_neighbors_mut() |= neighbor.from;
            *cell.alt_neighbors_mut() -= neighbor.from;
            self.solved_queue.push(&mut self.cells, neighbor.idx);
        }
        let neighbors = self.cells[idx].safe_neighbors();
        for neighbor in Cell::neighbor_iter(idx, self.width, neighbors) {
            let cell = &mut self.cells[neighbor.idx];
            *cell.mine_neighbors_mut() |= neighbor.from;
            *cell.alt_neighbors_mut() -= neighbor.from;
        }
        let cell = &self.cells[idx];
        let neighbors = cell.neighbors()
            - (cell.mine_neighbors() | cell.alt_neighbors() | cell.safe_neighbors());
        for neighbor in Cell::neighbor_iter(idx, self.width, neighbors) {
            let cell = &mut self.cells[neighbor.idx];
            *cell.mine_neighbors_mut() |= neighbor.from;
        }
        self.cells[idx].set_value(new_value);
        ChangeStatus::NonBreaking
    }

    fn unknown_to_inferred_safe(&mut self, idx: CellIdx, new_value: CellValue) -> ChangeStatus {
        // alt (count):  -alt, +solve
        // safe: -alt
        // mine: n/a
        // unknown: +safe
        // self: alt = unknown
        let alt_neighbors = self.cells[idx].alt_neighbors();
        for neighbor in Cell::neighbor_iter(idx, self.width, alt_neighbors) {
            let cell = &mut self.cells[neighbor.idx];
            *cell.alt_neighbors_mut() -= neighbor.from;
            self.solved_queue.push(&mut self.cells, neighbor.idx);
        }
        let safe_neighbors = self.cells[idx].safe_neighbors();
        for neighbor in Cell::neighbor_iter(idx, self.width, safe_neighbors) {
            let cell = &mut self.cells[neighbor.idx];
            *cell.alt_neighbors_mut() -= neighbor.from;
        }
        let cell = &self.cells[idx];
        let unknown_neighbors = cell.neighbors()
            - (cell.mine_neighbors() | cell.alt_neighbors() | cell.safe_neighbors());
        for neighbor in Cell::neighbor_iter(idx, self.width, unknown_neighbors) {
            let cell = &mut self.cells[neighbor.idx];
            *cell.safe_neighbors_mut() |= neighbor.from;
        }
        *self.cells[idx].alt_neighbors_mut() = unknown_neighbors;
        self.cells[idx].set_value(new_value);
        ChangeStatus::NonBreaking
    }

    fn unknown_to_count(&mut self, idx: CellIdx, new_value: CellValue) -> ChangeStatus {
        // alt (count):  -alt, +solve
        // safe: -alt
        // mine: n/a
        // unknown: +alt
        // self: alt = unknown, +solve
        let alt_neighbors = self.cells[idx].alt_neighbors();
        for neighbor in Cell::neighbor_iter(idx, self.width, alt_neighbors) {
            let cell = &mut self.cells[neighbor.idx];
            *cell.alt_neighbors_mut() -= neighbor.from;
            self.solved_queue.push(&mut self.cells, neighbor.idx);
        }
        let safe_neighbors = self.cells[idx].safe_neighbors();
        for neighbor in Cell::neighbor_iter(idx, self.width, safe_neighbors) {
            let cell = &mut self.cells[neighbor.idx];
            *cell.alt_neighbors_mut() -= neighbor.from;
        }
        let cell = &self.cells[idx];
        let unknown_neighbors = cell.neighbors()
            - (cell.mine_neighbors() | cell.alt_neighbors() | cell.safe_neighbors());
        for neighbor in Cell::neighbor_iter(idx, self.width, unknown_neighbors) {
            let cell = &mut self.cells[neighbor.idx];
            *cell.alt_neighbors_mut() |= neighbor.from;
        }
        *self.cells[idx].alt_neighbors_mut() = unknown_neighbors;
        self.solved_queue.push(&mut self.cells, idx);
        self.cells[idx].set_value(new_value);
        ChangeStatus::NonBreaking
    }

    fn inferred_safe_to_count(&mut self, idx: CellIdx, new_value: CellValue) -> ChangeStatus {
        // alt (unknown):  -safe, +alt
        // safe: n/a
        // mine: n/a
        // count: n/a
        // self: +solve
        let alt_neighbors = self.cells[idx].alt_neighbors();
        for neighbor in Cell::neighbor_iter(idx, self.width, alt_neighbors) {
            let cell = &mut self.cells[neighbor.idx];
            *cell.safe_neighbors_mut() -= neighbor.from;
            *cell.alt_neighbors_mut() |= neighbor.from;
        }
        self.solved_queue.push(&mut self.cells, idx);
        self.cells[idx].set_value(new_value);
        ChangeStatus::NonBreaking
    }

    // The slower, not-unrolled general purpose function for the rest of the values. Always involves
    // a loop over all neighbors with a decent amount of branching
    fn inference_breaking_change(
        &mut self,
        idx: CellIdx,
        prev_value: CellValue,
        new_value: CellValue,
    ) -> ChangeStatus {
        let mut mines = DirectionSet::EMPTY;
        let mut count = DirectionSet::EMPTY;
        let mut safe = DirectionSet::EMPTY;
        let mut unknown = DirectionSet::EMPTY;
        for neighbor in Cell::neighbor_iter(idx, self.width, self.cells[idx].neighbors()) {
            let n_value = self.cells[neighbor.idx].value();
            if n_value.is_mine() {
                mines |= neighbor.from.reverse();
            } else if n_value.has_count() {
                count |= neighbor.from.reverse();
                self.desolved_queue.push(&mut self.cells, neighbor.idx);
                self.solved_queue.push(&mut self.cells, neighbor.idx);
            } else if n_value.is_safe() {
                safe |= neighbor.from.reverse();
            } else {
                debug_assert!(n_value.is_unknown());
                unknown |= neighbor.from.reverse();
            }
            match prev_value.cell_type() {
                CellType::Unknown => {
                    if n_value.is_safe() {
                        *self.cells[neighbor.idx].alt_neighbors_mut() -= neighbor.from;
                    }
                }
                CellType::Mine | CellType::InferredMine => {
                    if !n_value.is_mine() {
                        *self.cells[neighbor.idx].mine_neighbors_mut() -= neighbor.from;
                    }
                }
                CellType::Count => {
                    if n_value.is_unknown() {
                        *self.cells[neighbor.idx].alt_neighbors_mut() -= neighbor.from;
                    }
                    self.desolved_queue.push(&mut self.cells, neighbor.idx);
                }
                CellType::InferredSafe => {
                    if n_value.is_unknown() {
                        *self.cells[neighbor.idx].safe_neighbors_mut() -= neighbor.from;
                    }
                }
            }
            match new_value.cell_type() {
                CellType::Unknown => {
                    if n_value.is_safe() {
                        *self.cells[neighbor.idx].alt_neighbors_mut() |= neighbor.from;
                    }
                }
                CellType::Mine | CellType::InferredMine => {
                    if !n_value.is_mine() {
                        *self.cells[neighbor.idx].mine_neighbors_mut() |= neighbor.from;
                    }
                }
                CellType::Count => {
                    if n_value.is_unknown() {
                        *self.cells[neighbor.idx].alt_neighbors_mut() |= neighbor.from;
                    }
                }
                CellType::InferredSafe => {
                    if n_value.is_unknown() {
                        *self.cells[neighbor.idx].safe_neighbors_mut() |= neighbor.from;
                    }
                }
            }
        }
        let cell = &mut self.cells[idx];
        cell.set_value(new_value);
        match new_value.cell_type() {
            CellType::Unknown => {
                *cell.alt_neighbors_mut() = count;
                *cell.safe_neighbors_mut() = safe;
                *cell.mine_neighbors_mut() = mines;
            }
            CellType::Count | CellType::InferredSafe => {
                *cell.alt_neighbors_mut() = unknown;
                *cell.mine_neighbors_mut() = mines;
            }
            _ => (),
        }
        if prev_value.has_count() {
            self.desolved_queue.push(&mut self.cells, idx);
        }
        if new_value.has_count() {
            self.solved_queue.push(&mut self.cells, idx);
        }

        ChangeStatus::Breaking
    }

    /// Compacts partition IDs when `next_partition_id` approaches `u16::MAX`.
    ///
    /// Re-maps all active partition IDs recorded in `self.partitions` to sequential IDs
    /// starting at `1`, updates every cell currently assigned to an active partition,
    /// clears any dangling non-zero partition IDs on cells not in `self.partitions`,
    /// and resets `next_partition_id` to `self.partitions.len() as u16 + 1`.
    pub(super) fn compact_partition_ids(&mut self) {
        let mut old_to_new: HashMap<u16, u16, FastNumberHasherBuilder> =
            HashMap::with_capacity_and_hasher(self.partitions.len(), FastNumberHasherBuilder);
        let mut new_id: u16 = 1;
        for &seed_idx in self.partitions.keys() {
            let old_id = self.cells[seed_idx].partition_id();
            if old_id != 0 && !old_to_new.contains_key(&old_id) {
                old_to_new.insert(old_id, new_id);
                new_id += 1;
            }
        }
        for cell in self.cells.iter_mut() {
            let pid = cell.partition_id();
            if pid != 0 {
                let mapped = old_to_new.get(&pid).copied().unwrap_or(0);
                cell.set_partition_id(mapped);
            }
        }
        self.next_partition_id = new_id;
    }

    fn solve(&mut self) {
        #[cfg(feature = "debug")]
        let debug = true;

        self.metrics.inc_solve_passes();
        self.metrics.mark_now();

        if self.next_partition_id >= 60_000 {
            self.compact_partition_ids();
        }

        // index the new pending changes
        #[cfg(feature = "debug")]
        if debug {
            eprint!("Pre indexing.");
            DetailMineField::eprintln(self);
        }
        self.index_pending();
        self.metrics.inc_indexing_duration();

        // process the "desolved" queue - i.e. reset all inferences attached to the provided cells
        #[cfg(feature = "debug")]
        if debug {
            eprint!("Post indexing, Pre desolved.");
            DetailMineField::eprintln(self);
        }
        self.process_desolved();
        self.metrics.inc_desolved_duration();

        // process the "solved" queue - i.e. find solved cells and mark their neighbors inferred
        #[cfg(feature = "debug")]
        if debug {
            eprint!("Post desolved, Pre solved.");
            DetailMineField::eprintln(self);
        }
        self.process_solved();
        self.metrics.inc_basic_solve_duration();

        // Create final partitions
        #[cfg(feature = "debug")]
        if debug {
            eprint!("Post solved, Pre partition.");
            DetailMineField::eprintln(self);
        }
        self.partition();
        self.metrics.inc_partition_duration();

        #[cfg(feature = "debug")]
        if debug {
            eprint!("Post partition.");
            DetailMineField::eprintln(self);
        }
    }

    fn index_pending(&mut self) {
        let mut has_breaking = false;
        for i in 0..self.pending.len() {
            let IdxAndCell { idx, value } = self.pending[i];
            has_breaking |= matches!(self.change_cell_value(idx, value), ChangeStatus::Breaking);
        }
        if has_breaking {
            // There is a chance that an item landed in the solved queue that was later updated to a
            // non-count cell. Clean up here to keep branching logic off the hot paths
            for i in 0..self.pending.len() {
                let IdxAndCell { idx, .. } = self.pending[i];
                let cell = &mut self.cells[idx];
                if !cell.value().has_count() {
                    self.solved_queue.remove(&mut self.cells, idx);
                    self.pending_partition_queue.remove(&mut self.cells, idx);
                }
            }
        }
        self.pending.clear();
    }

    fn process_desolved(&mut self) {
        let start_desolved_id = self.next_partition_id;
        while let Some(idx) = self.desolved_queue.pop(&mut self.cells) {
            let cell = &mut self.cells[idx];

            if cell.partition_id() >= start_desolved_id {
                continue;
            }
            if cell.value().has_count() {
                self.pending_partition_queue.push(&mut self.cells, idx);
            } else {
                self.partitions.remove(&idx);
            }

            self.desolve_partition(idx);
        }
    }

    fn process_solved(&mut self) {
        self.recent_inferred.clear();
        while let Some(idx) = self.solved_queue.pop(&mut self.cells) {
            let cell = &mut self.cells[idx];
            debug_assert!(cell.value().has_count(), "broke: {idx}");
            if !cell.is_solved()
                && let Some(value) = cell.infer_neighbors()
            {
                self.metrics.inc_neighbor_visits(cell.alt_neighbors());
                for Neighbor { idx: neighbor, .. } in
                    Cell::neighbor_iter(idx, self.width, cell.alt_neighbors())
                {
                    let change_type = self.change_cell_value(neighbor, value);
                    if !matches!(change_type, ChangeStatus::NoChange) {
                        self.recent_inferred.push(neighbor);
                    }
                }
            } else {
                self.pending_partition_queue.push(&mut self.cells, idx);
            }
        }
    }

    fn partition(&mut self) {
        if self.next_partition_id >= 60_000 {
            self.compact_partition_ids();
        }
        let start_partition_id = self.next_partition_id;
        self.recent_invalid.clear();
        while let Some(idx) = self.pending_partition_queue.pop(&mut self.cells) {
            let cell = &mut self.cells[idx];
            debug_assert!(cell.value().has_count());
            if cell.is_invalid() {
                self.recent_invalid.push(idx);
            }
            if cell.partition_id() < start_partition_id {
                self.partitions.remove(&idx);
                let len = self.visit_partition(idx, &mut |field, idx, _| {
                    field.partitions.remove(&idx);
                });
                if len > 1 {
                    self.partitions.insert(idx, len);
                    self.metrics.set_max_partitions(self.partitions.len());
                }
            }
        }
    }

    fn visit_neighbors_helper(
        &mut self,
        idx: CellIdx,
        prev_idx: Option<CellIdx>,
        visit: &mut impl FnMut(&mut MineField, CellIdx, Option<CellIdx>) -> Option<DirectionSet>,
    ) -> usize {
        if self.cells[idx].partition_id() == self.next_partition_id {
            return 0;
        }
        let Some(neighbors) = visit(self, idx, prev_idx) else {
            return 0;
        };
        self.cells[idx].set_partition_id(self.next_partition_id);

        let mut len = 1;
        self.metrics.inc_neighbor_visits(neighbors);
        for Neighbor { idx: n_idx, .. } in Cell::neighbor_iter(idx, self.width, neighbors) {
            len += self.visit_neighbors_helper(n_idx, Some(idx), visit);
        }
        len
    }

    fn visit_partition_helper(
        &mut self,
        idx: CellIdx,
        prev_idx: Option<CellIdx>,
        visit: &mut impl FnMut(&mut MineField, CellIdx, Option<CellIdx>),
    ) -> usize {
        self.visit_neighbors_helper(idx, prev_idx, &mut |field, idx, prev_idx| {
            let cell = field.cells[idx];
            match cell.value().cell_type() {
                CellType::Count | CellType::Unknown => {
                    visit(field, idx, prev_idx);
                    Some(cell.alt_neighbors())
                }
                CellType::Mine | CellType::InferredMine | CellType::InferredSafe => None,
            }
        })
    }

    // Step through the cells of a partition. Note: This function scans assuming the starting cell
    // has been mutated, and is unknown whether it was originally a known vs unknown. So, the
    // starting cell looks for *all* unknown + mine neighbors.
    fn desolve_partition(&mut self, idx: CellIdx) -> usize {
        let len = self.visit_neighbors_helper(idx, None, &mut |field, idx, prev_idx| {
            let cell = field.cells[idx];
            match cell.value().cell_type() {
                CellType::Unknown => {
                    field.partitions.remove(&idx);
                    Some(cell.alt_neighbors())
                }
                CellType::InferredSafe | CellType::InferredMine => {
                    field.partitions.remove(&idx);
                    field.change_cell_value(idx, CellValue::UNKNOWN);
                    let cell = field.cells[idx];
                    Some(cell.alt_neighbors())
                }
                CellType::Count => {
                    if prev_idx.is_none_or(|i| field.cells[i].value().is_unknown()) {
                        field.partitions.remove(&idx);
                        field.pending_partition_queue.push(&mut field.cells, idx);
                        Some(cell.neighbors())
                    } else {
                        None
                    }
                }
                CellType::Mine => None,
            }
        });
        if len <= 1 {
            self.cells[idx].set_partition_id(0);
        } else if len > 1 {
            self.next_partition_id += 1;
        }
        len
    }

    fn visit_partition(
        &mut self,
        member: CellIdx,
        visit: &mut impl FnMut(&mut MineField, CellIdx, Option<CellIdx>),
    ) -> usize {
        let len = self.visit_partition_helper(member, None, visit);
        if len <= 1 {
            self.cells[member].set_partition_id(0);
        } else if len > 1 {
            self.next_partition_id += 1;
            self.metrics.inc_partitions_created();
            self.metrics.set_max_partition_size(len);
        }
        len
    }

    /// Returns the slice of linear cell indices belonging to this partition.
    #[cfg(test)]
    fn members(&mut self, member: CellIdx, capacity: usize) -> Vec<CellIdx> {
        let mut members = Vec::with_capacity(capacity);
        let members_ref = &mut members;

        self.visit_partition(member, &mut |_, idx, _| {
            members_ref.push(idx);
        });
        members
    }

    /// Get the cell index for the given x/y coordinates.
    fn cell_idx(&self, p: Point) -> CellIdx {
        p.to_index(self.width)
    }

    /// Parses `field_spec` into an iterator of [`Cell`] updates, omitting skipped (`'.'` and `' '`) positions.
    fn cells_from_spec(field_spec: &str, width: u32) -> impl Iterator<Item = IdxAndCell> {
        Self::to_cell_iter(Self::spec_to_iter(field_spec), width)
    }

    /// Parses `field_spec` and returns the inferred `(width, height)` board dimensions along with an
    /// iterator of [`Cell`] updates for all non-skipped positions.
    pub(self) fn cells_and_size_from_spec(
        field_spec: &str,
    ) -> (u32, u32, impl Iterator<Item = IdxAndCell>) {
        let cells: Vec<_> = Self::spec_to_iter(field_spec).collect();

        let size = cells
            .iter()
            .map(|(Point { x, y }, _)| (x + 1, y + 1))
            .fold((0, 0), |acc, r| (max(acc.0, r.0), max(acc.1, r.1)));

        let cells = Self::to_cell_iter(cells, size.0);

        (size.0, size.1, cells)
    }

    /// Converts `(x, y, Option<CellValue>)` tuples into [`Cell`]s, filtering out `None` entries.
    fn to_cell_iter(
        iter: impl IntoIterator<Item = (Point, Option<CellValue>)>,
        width: u32,
    ) -> impl Iterator<Item = IdxAndCell> {
        iter.into_iter().filter_map(move |(p, v)| {
            v.map(|value| IdxAndCell {
                idx: cell_idx(p, width),
                value,
            })
        })
    }

    /// Parses `field_spec` into `(x, y, Option<CellValue>)` tuples for every character position in the spec.
    fn spec_to_iter(field_spec: &str) -> impl Iterator<Item = (Point, Option<CellValue>)> {
        fn char_to_value(c: char) -> Option<CellValue> {
            match c {
                '*' => Some(CellValue::MINE),
                n @ '0'..='8' => Some(CellValue::for_mine_count(n as u8 - b'0')),
                '_' | 'x' | '#' => Some(CellValue::UNKNOWN),
                ' ' | '.' => None,
                _ => panic!("Invalid field character: {c:?}"),
            }
        }

        field_spec
            .split("\n")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .enumerate()
            .flat_map(|(y, l)| {
                l.chars()
                    .map(char_to_value)
                    .enumerate()
                    .map(move |(x, value)| {
                        (
                            Point {
                                x: x as u32,
                                y: y as u32,
                            },
                            value,
                        )
                    })
            })
    }
}

impl Display for MineField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for y in 0..self.height {
            for x in 0..self.width {
                let UserValueWithInference { value, inferred } = self.cell(Point { x, y });
                let char = match (value, inferred) {
                    (UserCellValue::Mine, _) => '*',
                    (UserCellValue::Safe(n), _) => n.to_string().chars().next().unwrap(),
                    (UserCellValue::Unknown, InferredValue::None) => '.',
                    (UserCellValue::Unknown, InferredValue::Safe) => '#',
                    (UserCellValue::Unknown, InferredValue::Mine) => 'x',
                };
                write!(f, "{char}")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
impl MineField {
    /// Rebuilds a fresh [`MineField`] from the non-inferred known cells of this field and solves it from scratch.
    pub(super) fn rebuild_from_scratch(&self) -> Self {
        let mut cells = self.cells.clone();
        let mut pending = Vec::with_capacity(self.pending.capacity());
        for (idx, cell) in cells.iter_mut().enumerate() {
            if !cell.value().is_inferred() && cell.value() != CellValue::UNKNOWN {
                pending.push(IdxAndCell {
                    idx: CellIdx(idx as u32),
                    value: cell.value(),
                });
            }
            cell.reset();
        }
        let mut field = Self {
            width: self.width,
            height: self.height,
            mine_count: self.mine_count,
            cells,
            partitions: self.partitions.clone(),
            next_partition_id: 1,
            pending,
            desolved_queue: self.desolved_queue.clone(),
            solved_queue: self.solved_queue.clone(),
            pending_partition_queue: self.pending_partition_queue.clone(),
            metrics: Metrics::default(),
            recent_inferred: self.recent_inferred.clone(),
            recent_invalid: self.recent_invalid.clone(),
        };
        // clear the cloned collections for the new solve
        field.partitions.clear();
        field.desolved_queue.clear();
        field.solved_queue.clear();
        field.pending_partition_queue.clear();
        field.recent_inferred.clear();
        field.recent_invalid.clear();
        field.solve();
        field
    }
}

/// Errors that can occur when constructing or updating a [`MineField`].
#[derive(Debug)]
#[non_exhaustive]
pub enum SolverError {
    /// The requested board dimensions exceed the maximum supported cell count.
    AreaTooLarge { width: u32, height: u32 },
    /// The requested board dimensions include a zero width or height
    AreaTooSmall,
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
            SolverError::AreaTooSmall => write!(f, "The width and height must both be >= 1"),
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
        display::DetailMineField,
        grid::{E, NE, NW, S, SE, W},
    };

    /// Helper asserting neighbor direction set invariants:
    /// - For non-mine cells, `alt_neighbors` and `mine_neighbors` are disjoint subsets of `cell.neighbors`.
    /// - For unknown cells, `safe_neighbors` is also a disjoint subset of `cell.neighbors` (pairwise disjoint
    ///   from `alt_neighbors` and `mine_neighbors`).
    /// - Mine cells do not maintain neighbor direction sets.
    fn assert_neighbor_invariants(cell: &Cell) {
        if !cell.value().is_mine() {
            // all direction sets are subsets of neighbors
            assert_eq!(cell.alt_neighbors() | cell.neighbors(), cell.neighbors());
            assert_eq!(cell.mine_neighbors() | cell.neighbors(), cell.neighbors());

            // all direction sets are disjoint
            assert_eq!(
                cell.alt_neighbors() & cell.mine_neighbors(),
                DirectionSet::EMPTY
            );

            if cell.value().is_unknown() {
                assert_eq!(cell.safe_neighbors() | cell.neighbors(), cell.neighbors());
                assert_eq!(
                    cell.safe_neighbors() & cell.alt_neighbors(),
                    DirectionSet::EMPTY
                );
                assert_eq!(
                    cell.safe_neighbors() & cell.mine_neighbors(),
                    DirectionSet::EMPTY
                );
            }
        }
    }

    #[test]
    fn test_cell_change_value_updates_neighbor_masks() {
        // 3x1 strip, change the middle cell and watch both neighbors
        let mut field = MineField::new(3, 1).unwrap();

        field.change_cell_value(CellIdx(1), CellValue::MINE);
        assert_eq!(field.cells[1].value(), CellValue::MINE);
        assert_eq!(field.cells[0].mine_neighbors(), !!E);
        assert!(field.cells[0].alt_neighbors().is_empty());
        assert!(field.cells[0].safe_neighbors().is_empty());
        assert_eq!(field.cells[2].mine_neighbors(), !!W);
        assert!(field.cells[2].alt_neighbors().is_empty());
        assert!(field.cells[2].safe_neighbors().is_empty());

        // Mine -> Safe(1): neighbors drop the mine but stay known (not unknown)
        field.change_cell_value(CellIdx(1), CellValue::for_mine_count(1));
        assert!(field.cells[0].mine_neighbors().is_empty());
        assert!(field.cells[0].safe_neighbors().is_empty());
        assert_eq!(field.cells[0].alt_neighbors(), !!E);

        // Safe(1) -> Unknown: neighbors regain an unknown neighbor
        field.change_cell_value(CellIdx(1), CellValue::UNKNOWN);
        assert!(field.cells[0].mine_neighbors().is_empty());
        assert!(field.cells[0].alt_neighbors().is_empty());
        assert!(field.cells[2].mine_neighbors().is_empty());
        assert!(field.cells[2].alt_neighbors().is_empty());
    }

    #[test]
    fn test_streamlined_cell_value_transitions() {
        // Grid 3x1: [0, 1, 2]
        let mut field = MineField::new(3, 1).unwrap();

        // 1. Unknown -> Count(1) on cell 0
        let status = field.change_cell_value(CellIdx(0), CellValue::for_mine_count(1));
        assert_eq!(status, ChangeStatus::NonBreaking);
        assert_eq!(field.cells[0].value(), CellValue::for_mine_count(1));
        // Cell 0 is Count, so alt_neighbors points to unknown neighbors (Cell 1)
        assert_eq!(field.cells[0].alt_neighbors(), !!E);
        // Cell 1 is Unknown, so alt_neighbors points to count neighbors (Cell 0)
        assert_eq!(field.cells[1].alt_neighbors(), !!W);

        // 2. Unknown -> InferredSafe on cell 2
        let status = field.change_cell_value(CellIdx(2), CellValue::INFERRED_SAFE);
        assert_eq!(status, ChangeStatus::NonBreaking);
        assert_eq!(field.cells[2].value(), CellValue::INFERRED_SAFE);
        // Cell 1 is Unknown, so safe_neighbors points to inferred safe neighbors (Cell 2)
        assert_eq!(field.cells[1].safe_neighbors(), !!E);

        // 3. InferredSafe -> Count(0) on cell 2
        let status = field.change_cell_value(CellIdx(2), CellValue::for_mine_count(0));
        assert_eq!(status, ChangeStatus::NonBreaking);
        assert_eq!(field.cells[2].value(), CellValue::for_mine_count(0));
        // Cell 1 is Unknown: safe_neighbors has Cell 2 cleared, alt_neighbors has Cell 2 set
        assert!(field.cells[1].safe_neighbors().is_empty());
        assert_eq!(field.cells[1].alt_neighbors(), W | E);

        // 4. Unknown -> Mine on cell 1
        let status = field.change_cell_value(CellIdx(1), CellValue::MINE);
        assert_eq!(status, ChangeStatus::NonBreaking);
        assert_eq!(field.cells[1].value(), CellValue::MINE);
        // Both cell 0 and cell 2 are Count cells: alt_neighbors drops cell 1, mine_neighbors gets cell 1
        assert!(field.cells[0].alt_neighbors().is_empty());
        assert_eq!(field.cells[0].mine_neighbors(), !!E);
        assert!(field.cells[2].alt_neighbors().is_empty());
        assert_eq!(field.cells[2].mine_neighbors(), !!W);

        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_breaking_cell_value_transitions() {
        let mut field = MineField::new(3, 1).unwrap();
        field.change_cell_value(CellIdx(0), CellValue::for_mine_count(1));

        // Count -> different Count is Breaking
        let status = field.change_cell_value(CellIdx(0), CellValue::for_mine_count(2));
        assert_eq!(status, ChangeStatus::Breaking);

        // Count -> Unknown is Breaking
        let status = field.change_cell_value(CellIdx(0), CellValue::UNKNOWN);
        assert_eq!(status, ChangeStatus::Breaking);

        // Unknown -> Mine is NonBreaking
        let status = field.change_cell_value(CellIdx(0), CellValue::MINE);
        assert_eq!(status, ChangeStatus::NonBreaking);

        // Mine -> Unknown is Breaking
        let status = field.change_cell_value(CellIdx(0), CellValue::UNKNOWN);
        assert_eq!(status, ChangeStatus::Breaking);

        // InferredSafe -> Unknown is Breaking
        field.change_cell_value(CellIdx(0), CellValue::INFERRED_SAFE);
        let status = field.change_cell_value(CellIdx(0), CellValue::UNKNOWN);
        assert_eq!(status, ChangeStatus::Breaking);

        // Mine -> Count is Breaking
        field.change_cell_value(CellIdx(0), CellValue::MINE);
        let status = field.change_cell_value(CellIdx(0), CellValue::for_mine_count(1));
        assert_eq!(status, ChangeStatus::Breaking);

        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_create_minefield_basic_example() {
        // " 1* \n   1" -> 4x2 field with:
        // - Count(1) at (1, 0)
        // - Mine at (2, 0)
        // - Count(1) at (3, 1)
        let field = MineField::create(
            ".1* 
             .  1",
        );
        assert_eq!(field.width(), 4);
        assert_eq!(field.height(), 2);
        assert_eq!(field.mine_count(), None);

        assert_eq!(
            field.cell(Point { x: 0, y: 0 }).value,
            UserCellValue::Unknown
        );
        assert_eq!(
            field.cell(Point { x: 1, y: 0 }).value,
            UserCellValue::Safe(1)
        );
        assert_eq!(field.cell(Point { x: 2, y: 0 }).value, UserCellValue::Mine);
        assert_eq!(
            field.cell(Point { x: 3, y: 0 }).value,
            UserCellValue::Unknown
        );

        assert_eq!(
            field.cell(Point { x: 0, y: 1 }).value,
            UserCellValue::Unknown
        );
        assert_eq!(
            field.cell(Point { x: 1, y: 1 }).value,
            UserCellValue::Unknown
        );
        assert_eq!(
            field.cell(Point { x: 2, y: 1 }).value,
            UserCellValue::Unknown
        );
        assert_eq!(
            field.cell(Point { x: 3, y: 1 }).value,
            UserCellValue::Safe(1)
        );

        // Verify the solver ran on the populated cells:
        // Both (1, 0)=Count(1) and (3, 1)=Count(1) are satisfied by the Mine at (2, 0),
        // so all remaining unknown cells are inferred Safe!
        for pt in [
            Point { x: 0, y: 0 },
            Point { x: 3, y: 0 },
            Point { x: 0, y: 1 },
            Point { x: 1, y: 1 },
            Point { x: 2, y: 1 },
        ] {
            let idx = pt.to_index(field.width());
            assert!(field.cells[idx].value().is_safe());
        }
    }

    #[test]
    fn test_create_minefield_all_supported_characters() {
        // Test '0'..='8', '*', 'x', '#', ' ', '.', and '_'
        let field = MineField::create(
            "01234
             5678*
             *x#._",
        );
        assert_eq!(field.width(), 5);
        assert_eq!(field.height(), 3);
        assert_eq!(field.mine_count(), None);

        for x in 0..5u32 {
            assert_eq!(
                field.cell(Point { x, y: 0 }).value,
                UserCellValue::Safe(x as u8)
            );
        }
        for x in 0..4u32 {
            assert_eq!(
                field.cell(Point { x, y: 1 }).value,
                UserCellValue::Safe((x + 5) as u8)
            );
        }
        assert_eq!(field.cell(Point { x: 4, y: 1 }).value, UserCellValue::Mine);

        assert_eq!(field.cell(Point { x: 0, y: 2 }).value, UserCellValue::Mine);
        assert_eq!(
            field.cell(Point { x: 1, y: 2 }).value,
            UserCellValue::Unknown
        );
        assert_eq!(
            field.cell(Point { x: 2, y: 2 }).value,
            UserCellValue::Unknown
        );
        assert_eq!(
            field.cell(Point { x: 3, y: 2 }).value,
            UserCellValue::Unknown
        );
        assert_eq!(
            field.cell(Point { x: 4, y: 2 }).value,
            UserCellValue::Unknown
        );
    }

    #[test]
    fn test_cells_from_spec_and_update_minefield() {
        // Verify ' ' and '.' are filtered out while '_', 'x', '#', '*', and digits emit Cells.
        // Leading unspecified cell must use '.' because leading spaces are trimmed.
        let cells: Vec<IdxAndCell> = MineField::cells_from_spec(". 1_x#*", 7).collect();
        assert_eq!(
            cells,
            vec![
                IdxAndCell {
                    idx: CellIdx(2),
                    value: CellValue::for_mine_count(1)
                },
                IdxAndCell {
                    idx: CellIdx(3),
                    value: CellValue::UNKNOWN
                },
                IdxAndCell {
                    idx: CellIdx(4),
                    value: CellValue::UNKNOWN
                },
                IdxAndCell {
                    idx: CellIdx(5),
                    value: CellValue::UNKNOWN
                },
                IdxAndCell {
                    idx: CellIdx(6),
                    value: CellValue::MINE
                },
            ]
        );

        // Start with a 4x2 field where (0, 0)=Count(1) and (2, 0)=Count(2)
        let mut field = MineField::create(
            "1.2.
             ....",
        );
        assert_eq!(
            field.cell(Point { x: 0, y: 0 }).value,
            UserCellValue::Safe(1)
        );
        assert_eq!(
            field.cell(Point { x: 2, y: 0 }).value,
            UserCellValue::Safe(2)
        );

        // Update with '.' at (0, 0) and (2, 0) (preserving their values),
        // '*' at (1, 0), '_' at (3, 1), and '1' at (0, 1)
        field.update(
            ".*..
             1.._",
        );
        assert_eq!(
            field.cell(Point { x: 0, y: 0 }).value,
            UserCellValue::Safe(1)
        );
        assert_eq!(field.cell(Point { x: 1, y: 0 }).value, UserCellValue::Mine);
        assert_eq!(
            field.cell(Point { x: 2, y: 0 }).value,
            UserCellValue::Safe(2)
        );
        assert_eq!(
            field.cell(Point { x: 0, y: 1 }).value,
            UserCellValue::Safe(1)
        );
        assert_eq!(
            field.cell(Point { x: 3, y: 1 }).value,
            UserCellValue::Unknown
        );

        // Use '_' in `field.update` to explicitly revert (1, 0) and (2, 0) back to Unknown,
        // while '.' at (0, 0) preserves Count(1)
        field.update(".__");
        assert_eq!(
            field.cell(Point { x: 0, y: 0 }).value,
            UserCellValue::Safe(1)
        );
        assert_eq!(
            field.cell(Point { x: 1, y: 0 }).value,
            UserCellValue::Unknown
        );
        assert_eq!(
            field.cell(Point { x: 2, y: 0 }).value,
            UserCellValue::Unknown
        );
    }

    #[test]
    fn test_create_minefield_ragged_lines_and_1d_strips() {
        // Ragged lines: row 0 has length 2, row 1 has length 5 -> produces a 5x2 field
        // where missing trailing cells on row 0 default to Unknown.
        let ragged = MineField::create(
            "1*
             ....2",
        );
        assert_eq!(ragged.width(), 5);
        assert_eq!(ragged.height(), 2);
        assert_eq!(
            ragged.cell(Point { x: 0, y: 0 }).value,
            UserCellValue::Safe(1)
        );
        assert_eq!(ragged.cell(Point { x: 1, y: 0 }).value, UserCellValue::Mine);
        assert_eq!(
            ragged.cell(Point { x: 4, y: 0 }).value,
            UserCellValue::Unknown
        );
        assert_eq!(
            ragged.cell(Point { x: 4, y: 1 }).value,
            UserCellValue::Safe(2)
        );

        // 1x4 vertical strip
        let vertical = MineField::create(
            "1
             .
             *
             1",
        );
        assert_eq!(vertical.width(), 1);
        assert_eq!(vertical.height(), 4);
        assert_eq!(
            vertical.cell(Point { x: 0, y: 0 }).value,
            UserCellValue::Safe(1)
        );
        assert_eq!(
            vertical.cell(Point { x: 0, y: 1 }).value,
            UserCellValue::Unknown
        );
        assert_eq!(
            vertical.cell(Point { x: 0, y: 2 }).value,
            UserCellValue::Mine
        );
        assert_eq!(
            vertical.cell(Point { x: 0, y: 3 }).value,
            UserCellValue::Safe(1)
        );
    }

    #[test]
    #[should_panic(expected = "Invalid field character")]
    fn test_create_minefield_invalid_char_9_panics() {
        MineField::create("19*");
    }

    #[test]
    #[should_panic(expected = "Invalid field character")]
    fn test_create_minefield_invalid_char_letter_panics() {
        MineField::create("1Z*");
    }

    #[test]
    #[should_panic]
    fn test_create_minefield_empty_spec_panics() {
        MineField::create("");
    }

    #[test]
    fn test_new_minefield_validation() {
        let field = MineField::new(10, 10).expect("valid field");
        assert_eq!(field.width(), 10);
        assert_eq!(field.height(), 10);
        assert_eq!(
            field.cell(Point { x: 0, y: 0 }).value,
            UserCellValue::Unknown
        );
        for cell in &field.cells {
            assert!(cell.alt_neighbors().is_empty());
            assert!(cell.safe_neighbors().is_empty());
            assert!(cell.mine_neighbors().is_empty());
            assert_neighbor_invariants(cell);
        }

        // AreaTooLarge
        let err = MineField::new(u32::MAX, u32::MAX).unwrap_err();
        assert!(matches!(err, SolverError::AreaTooLarge { .. }));
        assert!(err.to_string().contains("must be below"));

        // MineCountTooLarge
        let err = MineField::with_mine_count(4, 4, 16).unwrap_err();
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
            MineField::new(0, 0),
            Err(SolverError::AreaTooSmall)
        ));
    }

    #[test]
    fn test_set_cell_out_of_bounds() {
        let mut field = MineField::new(3, 3).unwrap();
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

    fn cell(field: &MineField, x: u32, y: u32) -> Cell {
        field.cells[field.cell_idx(Point { x, y })]
    }

    #[test]
    fn test_neighbor_sets_and_remaining_mines_tracking() {
        // Reveal (0, 0) as Count(2) on a 3x3 board
        let mut field = MineField::create(
            "2..
             ...
             ...",
        );
        let c00 = cell(&field, 0, 0);
        assert_eq!(c00.remaining_mines(), 2);
        assert_eq!(c00.alt_neighbors(), E | S | SE);
        assert!(c00.mine_neighbors().is_empty());
        assert_neighbor_invariants(&c00);

        // Neighbor (1, 0) is Unknown and should move W from unknown_neighbors to safe_neighbors
        let c10 = cell(&field, 1, 0);
        assert_eq!(c10.alt_neighbors(), !!W);
        assert!(c10.mine_neighbors().is_empty());
        assert_neighbor_invariants(&c10);

        // Mark (1, 0) as a Mine: (0, 0) moves E from unknown_neighbors to
        // mine_neighbors (remaining_mines == 1)
        field.update(".*");
        let c00 = cell(&field, 0, 0);
        assert_eq!(c00.mine_neighbors(), !!E);
        assert_eq!(c00.alt_neighbors(), S | SE);
        assert_eq!(c00.remaining_mines(), 1);
        assert_neighbor_invariants(&c00);

        // Change (1, 0) from Mine to Count(1): (0, 0) moves E from mine_neighbors to safe_neighbors.
        // Because (0, 0) is Count(2) and now has only 2 unknown neighbors left (S and SE),
        // basic_solve infers both S and SE are mines, reducing remaining_mines() to 0!
        field.update(".1");
        let c00 = cell(&field, 0, 0);
        assert!(c00.alt_neighbors().is_empty());
        assert_eq!(c00.mine_neighbors(), S | SE);
        assert_eq!(c00.remaining_mines(), 0);
        assert_neighbor_invariants(&c00);

        // Revert (1, 0) back to Unknown (inference-breaking): (0, 0) moves E from safe_neighbors
        // back to unknown_neighbors, and the inferences on S and SE are cleared!
        field.update("._");
        let c00 = cell(&field, 0, 0);
        assert!(c00.mine_neighbors().is_empty());
        assert_eq!(c00.alt_neighbors(), E | S | SE);
        assert_eq!(c00.remaining_mines(), 2);
        assert_neighbor_invariants(&c00);
    }

    #[test]
    fn test_partitioning_and_repartitioning_across_updates() {
        // 5x2 board with Count(1) at top-left (0, 0) and top-right (4, 0)
        // Each corner has 3 unknown neighbors (choose(3, 1) = 3 > 1), so neither is trivially solved.
        let mut field = MineField::create(
            "1...1
             .....",
        );

        // Two disconnected frontier components:
        // Left: {(0,0), (1,0), (0,1), (1,1)} (4 cells)
        // Right: {(4,0), (3,0), (4,1), (3,1)} (4 cells)
        assert_eq!(field.partitions.len(), 2);
        let p_left = field.cells[0].partition_id();
        let p_right = field.cells[4].partition_id();
        assert_ne!(p_left, 0);
        assert_ne!(p_right, 0);
        assert_ne!(p_left, p_right);
        assert_eq!(
            field
                .cells
                .iter()
                .filter(|c| c.partition_id() == p_left)
                .count(),
            4
        );
        assert_eq!(
            field
                .cells
                .iter()
                .filter(|c| c.partition_id() == p_right)
                .count(),
            4
        );
        assert_eq!(field.cells[2].partition_id(), 0);
        assert!(field.cells[2].alt_neighbors().is_empty());
        assert!(field.cells[2].safe_neighbors().is_empty());

        // Reveal the middle top cell (2, 0) as Count(2): now both components are bridged
        // into a single 10-cell partition!
        field.update("..2");
        assert_eq!(field.partitions.len(), 1);
        let merged_pid = field.cells[2].partition_id();
        assert_ne!(merged_pid, 0);
        assert_eq!(
            field
                .cells
                .iter()
                .filter(|c| c.partition_id() == merged_pid)
                .count(),
            10
        );
        assert_eq!(field.cells[1].alt_neighbors(), W | E);
        assert_eq!(field.cells[3].alt_neighbors(), W | E);
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_partition_split_with_distant_non_dirty_cells() {
        // 5x2 board: reveal (1, 0) = Count(1) and (3, 0) = Count(1).
        // Initially all 10 cells are connected via (2, 0) and (2, 1) into a single partition.
        let mut field = MineField::create(
            ".1.1.
             .....",
        );
        assert_eq!(field.partitions.len(), 1);
        let initial_pid = field.cells[0].partition_id();
        assert_ne!(initial_pid, 0);
        assert!(field.cells.iter().all(|c| c.partition_id() == initial_pid));

        // Now reveal the bridge cell (2, 0) as Count(0).
        // `basic_solve` infers (1, 1), (2, 1), and (3, 1) as InferredValue::Safe,
        // and the post-`basic_solve` repartition pass splits the board into two 3-cell
        // partitions: {(0,0), (0,1), (1,0)} and {(3,0), (4,0), (4,1)}, while (2,0)
        // and its inferred-safe neighbors get partition_id == 0.
        field.update("..0");

        assert_eq!(field.partitions.len(), 2);
        assert!(field.cells[2].alt_neighbors().is_empty());

        let left_pid = field.cells[0].partition_id();
        let right_pid = field.cells[4].partition_id();
        assert_ne!(left_pid, 0);
        assert_ne!(right_pid, 0);
        assert_ne!(left_pid, right_pid);
        assert_eq!(field.cells[1].partition_id(), left_pid);
        assert_eq!(cell(&field, 0, 1).partition_id(), left_pid);
        assert_eq!(field.cells[3].partition_id(), right_pid);
        assert_eq!(cell(&field, 4, 1).partition_id(), right_pid);
        assert_eq!(
            field
                .cells
                .iter()
                .filter(|c| c.partition_id() == left_pid)
                .count(),
            3
        );
        assert_eq!(
            field
                .cells
                .iter()
                .filter(|c| c.partition_id() == right_pid)
                .count(),
            3
        );

        // Verify all cells (including distant cell 0) have synced inferred neighbor sets
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_partition_complete_dissolution() {
        // 3x1 board: [(0,0) = Unknown, (1,0) = Count(1), (2,0) = Unknown]
        // (1,0) has 2 unknown neighbors (choose(2, 1) = 2 > 1), so 1 partition of 3 cells remains.
        let mut field = MineField::create(".1.");
        assert_eq!(field.partitions.len(), 1);
        assert_eq!(field.cells[1].alt_neighbors(), W | E);
        assert_eq!(field.cells[0].alt_neighbors(), !!E);
        assert_eq!(field.cells[2].alt_neighbors(), !!W);

        // Flag (0, 0) as a Mine: basic_solve infers (2, 0) is Safe, and the post-basic_solve
        // repartition pass dissolves the partition completely.
        field.update("*");

        assert!(field.partitions.is_empty());
        assert!(field.cells[2].value().is_safe());
        assert_eq!(field.cells[1].remaining_mines(), 0);
        assert!(field.cells[1].alt_neighbors().is_empty());
        assert_eq!(field.cells[1].mine_neighbors(), !!W);

        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_batch_adjacent_cells_and_duplicate_dirty_neighbors() {
        // 3x2 board: reveal (1, 0) as Count(2) first
        let mut field = MineField::create(
            ".2.
             ...",
        );
        assert_eq!(field.cells[1].remaining_mines(), 2);

        // In one batch, set both (0, 0) and (2, 0) to Mine, and (1, 1) to Count(2).
        // (1, 0) is a non-dirty neighbor of all three pending cells, so it appears 3 times
        // in `dirty_neighbors`.
        field.update(
            "*.*
             .2.",
        );

        // Both (1, 0) and (1, 1) are adjacent to the two mines at (0, 0) and (2, 0),
        // so both have remaining_mines == 0, basic_solve infers (0, 1) and (2, 1) as Safe,
        // and the post-basic_solve repartition pass dissolves the solved partition.
        let c10 = cell(&field, 1, 0);
        let c11 = cell(&field, 1, 1);
        assert_eq!(c10.remaining_mines(), 0);
        assert_eq!(c11.remaining_mines(), 0);

        assert_eq!(c10.mine_neighbors(), W | E);
        assert!(c10.alt_neighbors().is_empty());

        assert_eq!(c11.mine_neighbors(), NW | NE);
        assert!(c11.alt_neighbors().is_empty());

        assert!(field.partitions.is_empty());
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_repeated_and_reverted_staging_on_same_cell() {
        let mut field = MineField::create(".1.");
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
        assert!(field.cells[1].alt_neighbors().is_empty());
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
        let field = MineField::create("1.1.");

        assert!(field.cells[1].value().is_mine());
        assert!(field.cells[3].value().is_safe());

        // Verify inferred neighbor sets reflect the deductions
        assert_eq!(field.cells[0].mine_neighbors(), !!E);
        assert!(field.cells[0].alt_neighbors().is_empty());
        assert_eq!(field.cells[2].mine_neighbors(), !!W);
        assert!(field.cells[2].alt_neighbors().is_empty());

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
        let field = MineField::create("1.2.1.");

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
        let field = MineField::create("0.1");

        assert_ne!(field.cells[0].is_invalid(), field.cells[2].is_invalid());
        let invalid: Vec<_> = field.recent_invalid_cells().collect();
        assert_eq!(invalid.len(), 1);
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_inference_compatible_update_preserves_inferences() {
        // 4x1 strip: [(0,0)=Count(1), (1,0)=Unknown, (2,0)=Unknown, (3,0)=Unknown]
        // Round 1: Reveal (0,0)=Count(1) -> infers (1,0) is Mine, and updates (2,0)'s
        // inferred_mine_neighbors to include W.
        let mut field = MineField::create("1...");
        assert!(field.cells[1].value().is_mine());
        assert_eq!(field.cells[2].mine_neighbors(), !!W);

        // Round 2 (inference-compatible): Reveal (2,0) as Count(1).
        // Because (1,0)'s Mine inference is preserved, (2,0) immediately knows its 1 mine is at W
        // and infers (3,0) is Safe!
        field.update("..1");
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
        let mut field = MineField::create(".2.");
        assert!(field.cells[0].value().is_mine());
        assert!(field.cells[2].value().is_mine());

        // Round 2 (Inference-breaking Case A): Change (1,0) from Count(2) to Count(1).
        // Neither (0,0) nor (2,0) can be inferred anymore; both must reset to InferredState::None!
        field.update(".1");
        assert!(!field.cells[0].value().is_inferred());
        assert!(!field.cells[2].value().is_inferred());
        assert_eq!(field.cells[1].alt_neighbors(), W | E);
        assert!(field.cells[1].mine_neighbors().is_empty());
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }

        // Round 3: Change (1,0) to Count(0) -> both (0,0) and (2,0) are inferred as Safe.
        field.update(".0");
        assert!(field.cells[0].value().is_safe());
        assert!(field.cells[2].value().is_safe());

        // Round 4 (Inference-breaking Case B): Directly violate inference on (0,0) by setting it to Mine,
        // and simultaneously change (1,0) to Count(1).
        // (2,0) must remain/be recomputed as Safe, and all neighbor invariants must hold.
        field.update("*1");
        assert!(!field.cells[0].value().is_inferred());
        assert!(field.cells[2].value().is_safe());
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }

        // Round 5 (Inference-breaking Case C): Dissolve the partition completely by resetting
        // (0,0) and (1,0) to Unknown. Verify (2,0)'s Safe inference is cleared even though no
        // new partition is created!
        field.update("__");
        DetailMineField::eprintln(&field);
        assert!(field.partitions.is_empty());
        for cell in &field.cells {
            assert!(!cell.value().is_inferred());
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_inference_breaking_update_preserves_untouched_partitions() {
        // 5x1 strip: [(0,0)=Count(1), (1,0)=Unknown, (2,0)=Unknown, (3,0)=Unknown, (4,0)=Count(1)]
        // Two independent partitions:
        // - Left partition {(0,0), (1,0)} infers (1,0) is Mine.
        // - Right partition {(3,0), (4,0)} infers (3,0) is Mine.
        let mut field = MineField::create("1...1");
        assert!(field.cells[1].value().is_mine());
        assert!(field.cells[3].value().is_mine());
        let right_pid = field.cells[4].partition_id();

        // Trigger an inference-breaking update on the LEFT partition only:
        // change (0,0) from Count(1) to Count(0).
        // Left cell (1,0) must change from InferredValue::Mine to InferredValue::Safe,
        // while the untouched RIGHT partition (and cell (3,0)'s Mine inference) remains intact!
        field.update("0");
        assert!(field.cells[1].value().is_safe());
        assert!(field.cells[3].value().is_mine());
        assert_eq!(field.cells[4].partition_id(), right_pid);
        assert_eq!(
            field
                .cells()
                .filter_map(|v| (v.inferred == InferredValue::Safe).then_some(v.cell.point))
                .collect::<Vec<_>>(),
            vec![Point { x: 1, y: 0 }]
        );
        assert_eq!(
            field
                .cells()
                .filter_map(|v| (v.inferred == InferredValue::Mine).then_some(v.cell.point))
                .collect::<Vec<_>>(),
            vec![Point { x: 3, y: 0 }]
        );
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    #[test]
    fn test_invalid_resolved_by_editing_other_cell_or_invalid_cell_itself() {
        // 3x1 strip: [(0,0)=Count(0), (1,0)=Unknown, (2,0)=Count(1)]
        // (0,0) is visited first (mine_count = 0), inferring (1,0) is Safe.
        // (2,0) is visited last (mine_count = 1), and index 2 is added to `field.invalid`.
        let mut field = MineField::create("0.1");
        let invalid: Vec<_> = field.recent_invalid_cells().collect();
        // solver is free to mark the center as either  a mine or not
        if field.cells[1].value().is_mine() {
            assert_eq!(invalid, vec![CellIdx(0)]);
        } else {
            assert_eq!(invalid, vec![CellIdx(2)]);
        }

        // Case A: Resolve the inconsistency by editing (0,0) (index 0, NOT index 2!)
        // from Count(0) to Count(1). Index 2 must be removed from `field.invalid`,
        // and (1,0) must now be inferred as Mine.
        field.update("1");
        assert!(field.recent_invalid_cells().next().is_none());
        assert!(field.cells[1].value().is_mine());
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }

        // Re-introduce the inconsistency sequentially: set (0,0) back to Count(0).
        // Index 2 (highest mine_count) is again recorded in `field.invalid`.
        field.update("0");
        let invalid: Vec<_> = field.recent_invalid_cells().collect();
        if field.cells[1].value().is_mine() {
            assert_eq!(invalid, vec![CellIdx(0)]);
        } else {
            assert_eq!(invalid, vec![CellIdx(2)]);
            assert!(field.cells[1].value().is_safe());
        }

        // Case B: Resolve the inconsistency by editing (2,0) (index 2 itself)
        // from Count(1) to Count(0).
        field.update("..0");
        assert!(field.recent_invalid_cells().next().is_none());
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
        let mut field = MineField::create("..1.0.1");
        assert!(field.cells[1].value().is_mine());
        // one of id: 4 or id: 6 is invalid
        assert_ne!(field.cells[4].is_invalid(), field.cells[6].is_invalid());

        // Step 2: Reveal (0,0) = Count(0) on the left partition.
        // Even though (0,0) was Unknown with InferredValue::None, its Count(0) conflicts
        // with (1,0)'s prior Mine inference, triggering a reset of the left partition so
        // (0,0) (mine_count = 0) is visited first and (2,0) (mine_count = 1,
        // index 2) is added to `invalid`.
        field.update("0");

        // Note: There are several equally valid resolutions to this field.

        // Step 3: Resolve ONLY the left partition by setting (1,0) = Mine and (0,0) = Count(1)
        // (so the left side has no Unknown cells left and is completely valid).
        // Index 2 must be removed from `invalid`, while untouched index 6 remains in `invalid`!
        field.update("1*");
        assert_ne!(field.cells[4].is_invalid(), field.cells[6].is_invalid());
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
        let field = MineField::create(
            ".333.
             .....",
        );

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
        assert_eq!(
            field
                .cells
                .iter()
                .filter(|c| c.partition_id() == left_pid)
                .count(),
            3
        );
        assert_eq!(
            field
                .cells
                .iter()
                .filter(|c| c.partition_id() == right_pid)
                .count(),
            3
        );
        for cell in &field.cells {
            assert_neighbor_invariants(cell);
        }
    }

    fn assert_partition_invariants(field: &MineField) {
        let mut partition_sizes: HashMap<u16, usize> = HashMap::new();
        for cell in &field.cells {
            let pid = cell.partition_id();
            if pid != 0 {
                *partition_sizes.entry(pid).or_insert(0) += 1;
            }
        }
        for (&seed, &len) in &field.partitions {
            let pid = field.cells[seed].partition_id();
            assert_ne!(pid, 0);
            assert_eq!(partition_sizes.get(&pid), Some(&len));
        }
    }

    #[test]
    fn test_index_pending_updates_masks_and_logs_neighbors() {
        let mut field = MineField::new(3, 1).unwrap();
        field.pending.push(IdxAndCell {
            idx: CellIdx(1),
            value: CellValue::for_mine_count(1),
        });

        field.index_pending();

        // pending is drained and the neighbors' masks reflect the new known cell
        assert!(field.pending.is_empty());
        assert_eq!(field.cells[1].value(), CellValue::for_mine_count(1));
        assert_eq!(field.cells[0].alt_neighbors(), !!E);
        assert_eq!(field.cells[2].alt_neighbors(), !!W);
        assert_eq!(field.cells[1].alt_neighbors(), W | E);

        // only the changed cell will be queued for repartitioning
        let queued: Vec<CellIdx> = field.solved_queue.iter().collect();
        assert_eq!(queued, vec![CellIdx(1)]);
    }

    #[test]
    fn test_index_pending_routes_breaking_changes_to_breaking_log() {
        let mut field = MineField::new(3, 1).unwrap();
        // Simulate a previously inferred-safe cell in the middle.
        field.cells[1].set_value(CellValue::INFERRED_SAFE);
        *field.cells[1].alt_neighbors_mut() = E | W;

        // Confirming the inference (Safe(n)) is non-breaking...
        field.pending.push(IdxAndCell {
            idx: CellIdx(1),
            value: CellValue::for_mine_count(1),
        });
        field.index_pending();
        assert!(field.desolved_queue.is_empty());
        assert!(!field.solved_queue.is_empty());
        field.pending_partition_queue.clear();

        // ...whereas contradicting it (Mine) is breaking.
        field.pending.push(IdxAndCell {
            idx: CellIdx(1),
            value: CellValue::MINE,
        });
        field.index_pending();
        assert!(!field.desolved_queue.is_empty());
    }

    #[test]
    fn test_partition_invariants_hold_after_merge() {
        let mut field = MineField::create(
            "1...1
             .....",
        );
        assert_partition_invariants(&field);
        field.update("..2");
        assert_eq!(field.partitions.len(), 1);
        assert_partition_invariants(&field);
        // pending/desolved queues are fully drained after a solve pass
        assert!(field.pending.is_empty());
        assert!(field.pending_partition_queue.is_empty());
        assert!(field.desolved_queue.is_empty());
    }

    #[test]
    fn test_partition_split_leaves_no_stale_partition_ids() {
        // Initially one partition {0..=4} bridged through cell 2.
        let mut field = MineField::create(".1.1.");
        assert_eq!(field.partitions.len(), 1);

        // Reverting cell 1 to Unknown leaves only {2, 3, 4} connected.
        field.update("._");
        DetailMineField::eprintln(&field);
        assert_eq!(field.partitions.len(), 1);
        assert!(field.cells[1].partition_id() < field.next_partition_id);
        assert_partition_invariants(&field);
    }

    #[test]
    fn test_partition_dissolution_leaves_no_stale_partition_ids() {
        let mut field = MineField::create(".1.");
        assert_eq!(field.partitions.len(), 1);

        field.update("._");
        assert!(field.partitions.is_empty());
        for cell in field.cells.iter() {
            assert!(cell.partition_id() < field.next_partition_id);
        }
    }

    #[test]
    fn test_unchanged_update_does_not_rebuild_partitions() {
        let mut field = MineField::create(
            "1...1
             .....",
        );
        let before: Vec<_> = field.cells.iter().map(|c| c.partition_id()).collect();

        field.update("1...1");
        let after: Vec<_> = field.cells.iter().map(|c| c.partition_id()).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn test_solve_queue_with_duplicates() {
        let field = MineField::create(
            "2.13
             2.1.",
        );
        assert!(field.recent_invalid_cells().next().is_some());
    }

    #[test]
    fn test_partition_create_connected_bipartite_component() {
        // 3x1 board: [Count(1), Unknown, Count(2)]
        let mut field = MineField::new(3, 1).unwrap();
        field.cells[0].set_value(CellValue::for_mine_count(1));
        *field.cells[0].alt_neighbors_mut() = !!E;

        field.cells[1].set_value(CellValue::UNKNOWN);
        *field.cells[1].alt_neighbors_mut() = W | E;

        field.cells[2].set_value(CellValue::for_mine_count(2));
        *field.cells[2].alt_neighbors_mut() = !!W;

        let members = field.members(CellIdx(0), 3);
        assert_eq!(members, vec![CellIdx(0), CellIdx(1), CellIdx(2)]);
        let id = field.cells[0].partition_id();
        assert_ne!(id, 0);
        for c in &field.cells {
            assert_eq!(c.partition_id(), id);
        }
    }

    #[test]
    fn test_compact_partition_ids() {
        let spec = "1.1\n...\n1.1";
        let mut field = MineField::create(spec);
        assert!(!field.partitions.is_empty());
        let orig_partition_count = field.partitions.len();

        // Artificially elevate next_partition_id near u16::MAX
        field.next_partition_id = 60_005;
        // Inject an obsolete cell partition_id that is NOT in field.partitions
        field.cells[4].set_partition_id(50_000);

        field.compact_partition_ids();

        assert_eq!(field.partitions.len(), orig_partition_count);
        assert_eq!(field.next_partition_id, orig_partition_count as u16 + 1);
        // Dangling partition ID on cell 4 was cleared to 0
        assert_eq!(field.cells[4].partition_id(), 0);

        // All active partition members have contiguous IDs in 1..=orig_partition_count
        for &seed in field.partitions.keys() {
            let pid = field.cells[seed].partition_id();
            assert!(pid >= 1 && pid <= orig_partition_count as u16);
        }

        // Solving further when next_partition_id >= 60_000 automatically triggers compaction
        field.next_partition_id = 60_000;
        field.solve();
        assert!(field.next_partition_id <= orig_partition_count as u16 + 2);
    }

    #[test]
    fn test_minefield_clone_and_rebuild_from_scratch() {
        let spec = "111\n1*1\n111";
        let field = MineField::create(spec);

        // Exact clone preserves all internal cell states and partition structures
        let cloned = field.clone();
        assert_eq!(cloned.width, field.width);
        assert_eq!(cloned.height, field.height);
        assert_eq!(cloned.mine_count, field.mine_count);
        assert_eq!(cloned.next_partition_id, field.next_partition_id);
        assert_eq!(cloned.partitions.len(), field.partitions.len());
        for (c1, c2) in field.cells.iter().zip(cloned.cells.iter()) {
            assert_eq!(c1, c2);
        }

        // rebuild_from_scratch resets inferences and re-solves
        let rebuilt = field.rebuild_from_scratch();
        assert_eq!(rebuilt.width, field.width);
        assert_eq!(rebuilt.height, field.height);
        assert_eq!(rebuilt.mine_count, field.mine_count);
        for (c1, c2) in field.cells.iter().zip(rebuilt.cells.iter()) {
            assert_eq!(c1.value(), c2.value());
        }
    }

    #[test]
    fn test_queue_has_count_invariants() {
        // Create a board with counts, unknowns, and mines
        let spec = "1.1\n...\n1.1";
        let mut field = MineField::create(spec);
        for idx in field.pending_partition_queue.iter() {
            assert!(
                field.cells[idx].value().has_count(),
                "pending_partition_queue must only contain cells with has_count"
            );
        }
        for idx in field.solved_queue.iter() {
            assert!(
                field.cells[idx].value().has_count(),
                "solved_queue must only contain cells with has_count"
            );
        }

        // Apply a breaking update changing a count cell into an unknown and another into a mine
        field.update("._*\n...\n1.1");
        for idx in field.pending_partition_queue.iter() {
            assert!(
                field.cells[idx].value().has_count(),
                "pending_partition_queue must strictly maintain has_count after updates"
            );
        }
        for idx in field.solved_queue.iter() {
            assert!(
                field.cells[idx].value().has_count(),
                "solved_queue must strictly maintain has_count after updates"
            );
        }
    }

    #[test]
    fn test_process_desolved_multiple_partitions_avoids_redundant_walks() {
        // Two independent partitions: left (0,0) and right (4,0)
        let mut field = MineField::create("1...1\n.....");
        assert_eq!(field.partitions.len(), 2);
        let pid_before = field.next_partition_id;

        // Break both partitions in the same cycle by reverting both back to unknown
        field.update("_..._\n.....");
        assert!(field.desolved_queue.is_empty());
        // next_partition_id should not experience runaway increments
        assert!(field.next_partition_id <= pid_before + 4);
    }
}

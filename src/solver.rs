//! Core Minesweeper solver state and partition orchestration.
use cell::{UserCell, UserCellValue};
use grid::Point;
use std::{
    cmp::max,
    collections::HashMap,
    error::Error,
    fmt::{self, Display},
};

#[cfg(test)]
use crate::solver::cell::{CellFlags, CellNeighbors};
use crate::{
    InferredValue,
    solver::{
        cell::{
            Cell, CellIdx, CellValue, FastNumberHasherBuilder, FlagMask, GuardedQueue, IndexedCell,
            IndexedCellValue, UserCellWithInference, UserValueWithInference, cell_idx, coordinates,
            create_grid_with_border,
        },
        metrics::Metrics,
        partition::{Partition, PartitionId},
    },
};

pub(crate) mod cell;
#[cfg(any(test, feature = "debug"))]
mod display;
pub(crate) mod grid;
mod metrics;
mod partition;
mod stats;
pub(crate) mod test_utils;
mod utils;

/// Represents the state of a Minesweeper board and its active solving partitions.
#[derive(Debug, Clone)]
pub struct MineField {
    /// Width of the board in cells, including the shared border column of `WALL` cells (`width + 1`).
    width_with_border: u32,
    /// Height of the board in cells, including the top and bottom border rows of `WALL` cells (`height + 2`).
    height_with_border: u32,
    /// Precomputed signed index offsets for all 8 compass neighbors (`NW..SE`).
    neighbor_offsets: [i32; 8],
    /// Total number of mines on the board.
    mine_count: Option<u32>,
    /// Row-major contiguous array of packed 8-byte per-cell state (neighbor masks, partition ID, value, flags).
    cells: Box<[Cell]>,
    /// Active board partitions, mapping each [`PartitionId`] to its [`Partition`] metadata (seed cell and size).
    partitions: HashMap<PartitionId, Partition, FastNumberHasherBuilder>,
    /// Next unique ID to assign to a newly walked partition.
    next_partition_id: PartitionId,
    /// User-supplied cell updates waiting to be applied by [`MineField::index_pending`].
    pending: Vec<IndexedCellValue>,
    /// Queue of opened (or formerly opened) cells whose breaking changes require inferences in their connected components to be reset.
    desolved_queue: GuardedQueue<{ FlagMask::DESOLVE_FLAG.mask() }>,
    /// Reusable scratch buffer used by [`MineField::process_desolved`] (lazily allocated on first desolve).
    desolve_buf: Vec<CellIdx>,
    /// Reusable BFS buffers used by [`MineField::partition`] to collect partition members.
    partition_opened_buf: Vec<CellIdx>,
    partition_unknown_buf: Vec<CellIdx>,
    /// Queue of cells (typically satisfying [`CellValue::is_opened`]) whose neighbors may have become trivially solvable;
    /// non-open cells forwarded from `desolved_queue` are skipped for solving and forwarded to `pending_partition_queue`.
    solved_queue: GuardedQueue<{ FlagMask::SOLVED_QUEUE.mask() }>,
    /// Queue of opened or unknown cells whose connected frontier needs to be partitioned, repartitioned, or reset to unpartitioned.
    pending_partition_queue: GuardedQueue<{ FlagMask::PENDING_PARTITION.mask() }>,
    /// Performance and step counters recorded during solving.
    metrics: Metrics,
    /// Cell indices whose inference changed during the most recent solve cycle.
    recent_inferred: Vec<CellIdx>,
    /// Contradictory open cells discovered during the most recent solve cycle.
    recent_invalid: Vec<CellIdx>,
}

const MAX_SIZE: u32 = u32::MAX >> 9;

impl MineField {
    /// Creates a new [`MineField`] with the given dimensions and no total mine count constraint.
    ///
    /// All cells are initialized to [`crate::CellValue::Unknown`].
    ///
    /// # Errors
    ///
    /// * Returns [`SolverError::AreaTooSmall`] if `width == 0` or `height == 0`.
    /// * Returns [`SolverError::AreaTooLarge`] if the bordered cell count exceeds `MAX_SIZE`.
    pub fn new(width: u32, height: u32) -> Result<MineField, SolverError> {
        Self::new_helper(width, height, None)
    }

    /// Creates a new [`MineField`] with the given dimensions and total `mine_count`.
    ///
    /// All cells are initialized to [`crate::CellValue::Unknown`].
    ///
    /// # Errors
    ///
    /// * Returns [`SolverError::AreaTooSmall`] if `width == 0` or `height == 0`.
    /// * Returns [`SolverError::AreaTooLarge`] if the bordered cell count exceeds `MAX_SIZE`.
    /// * Returns [`SolverError::MineCountTooLarge`] if `mine_count > width * height`.
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
        let area = (width as usize).saturating_mul(height as usize);
        let perimeter = (width as usize + height as usize) * 2;
        // Area with room for WALL cells on the top, bottom, left, plus 1 extra fo last cell's SE
        let border_area = ((width as usize) + 1)
            .saturating_mul((height as usize) + 2)
            .saturating_add(1);
        if border_area > MAX_SIZE as usize {
            return Err(SolverError::AreaTooLarge { width, height });
        }
        if area == 0 {
            return Err(SolverError::AreaTooSmall);
        }
        if let Some(mine_count) = mine_count
            && mine_count as usize > area
        {
            return Err(SolverError::MineCountTooLarge {
                mine_count,
                width,
                height,
            });
        }

        let cells = create_grid_with_border(width, height);

        let width_with_border = width + 1;
        let field = MineField {
            // make room for a set of WALL cells between rows
            width_with_border,
            // make room for a set of WALL cells above and below the interior cells
            height_with_border: height + 2,
            neighbor_offsets: CellIdx::neighbor_offsets(width_with_border),
            mine_count,
            cells,
            partitions: HashMap::with_capacity_and_hasher(16, FastNumberHasherBuilder),
            next_partition_id: PartitionId::ONE,
            pending: Vec::with_capacity(area / 4),
            desolved_queue: GuardedQueue::with_capacity(0),
            desolve_buf: Vec::new(),
            partition_opened_buf: Vec::new(),
            partition_unknown_buf: Vec::new(),
            solved_queue: GuardedQueue::with_capacity(perimeter),
            pending_partition_queue: GuardedQueue::with_capacity(area / 4),
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
    /// // - Opened(1) at (1, 0)
    /// // - Mine at (2, 0)
    /// // - Opened(1) at (3, 1)
    /// // - Unknown everywhere else
    /// let field = MineField::create(
    ///     ".1*.
    ///      ...1",
    /// );
    /// assert_eq!(field.width(), 4);
    /// assert_eq!(field.height(), 2);
    /// assert_eq!(field.cell(Point { x: 1, y: 0 }).value, CellValue::Opened(1));
    /// assert_eq!(field.cell(Point { x: 2, y: 0 }).value, CellValue::Mine);
    /// assert_eq!(field.cell(Point { x: 3, y: 1 }).value, CellValue::Opened(1));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `field_spec` contains any character outside `'0'..='8'`, `'*'`, `'x'`, `'#'`, `'_'`, `'@'`, `'.'`, `' '`, or `'\n'`.
    /// * The parsed board dimensions or `mine_count` fail [`MineField::new`] / [`MineField::with_mine_count`]
    ///   validation (e.g., an empty spec or `mine_count > width * height`).
    #[track_caller]
    pub fn create(field_spec: &str) -> MineField {
        let (width, height, cell_iter) = Self::cells_and_size_from_spec(field_spec);

        // TODO: implement mine count into the spec
        let mut field = MineField::new(width, height).unwrap();
        for cell in cell_iter {
            field.pending.push(IndexedCellValue {
                idx: field.cell_idx(cell.point),
                value: cell.value.into(),
            });
        }
        field.solve();

        field
    }

    /// Returns the width of the board in cells.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width_with_border - 1
    }

    /// Returns the height of the board in cells.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height_with_border - 2
    }

    /// Returns the total number of mines on the board.
    #[inline]
    pub fn mine_count(&self) -> Option<u32> {
        self.mine_count
    }

    /// Returns the [`UserValueWithInference`] at the given coordinate `p`.
    ///
    /// # Panics
    ///
    /// Panics if `p.x >= self.width()` or `p.y >= self.height()`.
    pub fn cell(&self, p: Point) -> UserValueWithInference {
        assert!(p.x < self.width());
        assert!(p.y < self.height());
        self.cells[self.cell_idx(p)].value.into()
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
    /// // Reveal (1, 0) as Opened(2) and flag (2, 1) as a Mine, leaving all '.' cells untouched:
    /// field.update(
    ///     ".2
    ///      ..*",
    /// );
    /// assert_eq!(field.cell(Point { x: 0, y: 0 }).value, CellValue::Opened(1));
    /// assert_eq!(field.cell(Point { x: 1, y: 0 }).value, CellValue::Opened(2));
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
        for cell in Self::cells_from_spec(field_spec) {
            assert!(
                cell.point.x < self.width() && cell.point.y < self.height(),
                "Coordinates ({}, {}) out of bounds for {}x{} field",
                cell.point.x,
                cell.point.y,
                self.width(),
                self.height()
            );
            self.pending.push(IndexedCellValue {
                idx: self.cell_idx(cell.point),
                value: cell.value.into(),
            });
        }
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
            if cell.point.x >= self.width() || cell.point.y >= self.height() {
                return Err(SolverError::CoordinatesOutOfBounds {
                    p: cell.point,
                    width: self.width(),
                    height: self.height(),
                });
            }
            self.pending.push(IndexedCellValue {
                idx: self.cell_idx(cell.point),
                value: cell.value.into(),
            });
        }

        self.solve();
        Ok(())
    }

    /// Provide a user view into the cell array. Note: There is no direct, fast access to all
    /// "inferred" cells. This is intentional; For fast access, refer to `recent_inferred` after
    /// each solve pass.
    pub fn cells(&self) -> impl Iterator<Item = UserCellWithInference> + use<'_> {
        (0..self.height()).flat_map(move |y| {
            (0..self.width()).map(move |x| {
                let point = Point { x, y };
                let idx = self.cell_idx(point);
                let v: UserValueWithInference = self.cells[idx].value.into();
                UserCellWithInference {
                    cell: UserCell {
                        point,
                        value: v.value,
                    },
                    inferred: v.inferred,
                }
            })
        })
    }

    /// Returns an iterator over [`Point`] coordinates whose inferred value was deduced
    /// during the most recent solve pass.
    ///
    /// Cleared at the start of every `process_solved` step.
    pub fn recent_inferred(&self) -> impl Iterator<Item = Point> {
        self.recent_inferred.iter().map(|&i| self.coordinates(i))
    }

    #[cfg(test)]
    pub(self) fn recent_invalid(&self) -> impl Iterator<Item = Point> {
        self.recent_invalid.iter().map(|&i| self.coordinates(i))
    }

    #[cfg(test)]
    pub(self) fn recent_inferred_mines(&self) -> impl Iterator<Item = Point> {
        self.recent_inferred
            .iter()
            .filter(|&&i| self.cells[i].value.is_mine())
            .map(|&i| self.coordinates(i))
    }

    #[cfg(test)]
    pub(self) fn recent_inferred_safe(&self) -> impl Iterator<Item = Point> {
        self.recent_inferred
            .iter()
            .filter(|&&i| self.cells[i].value.is_safe())
            .map(|&i| self.coordinates(i))
    }

    /// Updates `self.cells[idx].value` to `new_value`, synchronizes neighbor direction bitmasks,
    /// and queues affected cells onto `desolved_queue` and `solved_queue`.
    fn change_cell_value(&mut self, idx: CellIdx, new_value: CellValue) {
        #[cfg(feature = "debug")]
        let pre_change = self.clone();

        let prev_value = self.cells[idx].value;
        if prev_value == new_value {
            return;
        }
        #[cfg(feature = "debug")]
        {
            eprintln!("Applying change: ({idx}): {prev_value} -> {new_value}");
        }

        // Update the new value and all neighbor back references
        self.cells[idx].value = new_value;
        CellValue::update_neighbors(
            &mut self.cells,
            self.width_with_border,
            idx,
            prev_value,
            new_value,
        );
        #[cfg(feature = "debug")]
        {
            // This does a full board validation with all neighbor references after each change.
            // This is quite expensive.
            Self::validate_board(self, &pre_change, idx, new_value);
        }

        // If `idx` was previously opened, any partition/solved state recorded under it may be stale.
        if prev_value.is_opened() {
            self.desolved_queue.push(&mut self.cells, idx);
        }
        if new_value.is_opened() {
            self.solved_queue.push(&mut self.cells, idx);
        }

        let neighbors = self.cells[idx].neighbors;
        let mine_changed = prev_value.is_mine() != new_value.is_mine();
        // Queue neighbor desolves if we changed our neighbors' known mine count or cleared an
        // opened/mine cell; whenever `desolve_neighbors` holds, `mine_changed` also requires a
        // neighbor re-solve, so both queues can share a single `neighbors.opened` pass.
        let desolve_neighbors = !prev_value.is_unknown() && mine_changed;
        if (prev_value.is_unknown() || new_value.is_unknown() || mine_changed)
            && !neighbors.opened().is_empty()
        {
            let offsets = self.neighbor_offsets;
            for n_idx in idx.neighbor_iter(&offsets, neighbors.opened()) {
                if desolve_neighbors {
                    self.desolved_queue.push(&mut self.cells, n_idx);
                }
                self.solved_queue.push(&mut self.cells, n_idx);
            }
        }
    }

    #[cfg(feature = "debug")]
    fn validate_board(&self, pre_change: &MineField, source_idx: CellIdx, new_value: CellValue) {
        fn failure(
            field: &MineField,
            pre_change: &MineField,
            source_idx: CellIdx,
            n_idx: CellIdx,
            infer_idx: CellIdx,
            new_value: CellValue,
        ) {
            use crate::solver::display::ColoredMineField;

            eprintln!(
                "\nBoard validation failure (source idx: {source_idx}, neighbor idx: {n_idx}, infer idx: {infer_idx}, new_value: {new_value}.\nPost-change:"
            );
            ColoredMineField::eprintln(&field);
            DetailMineField::eprintln(&field);
            eprintln!("Pre-change:\n");
            ColoredMineField::eprintln(&pre_change);
            DetailMineField::eprintln(&pre_change);
            panic!();
        }
        use crate::solver::display::DetailMineField;
        for (idx, cell) in self.cells.iter().copied().enumerate() {
            use crate::solver::grid::DirectionSet;

            if cell.value.is_wall() {
                continue;
            }
            let n_idx = CellIdx(idx as u32);
            let neighbors = self.cells[n_idx].neighbors;
            for infer_idx in n_idx.neighbor_iter(&self.neighbor_offsets, neighbors.opened()) {
                if !self.cells[infer_idx].value.is_opened() {
                    failure(self, pre_change, source_idx, n_idx, infer_idx, new_value);
                }
            }
            for infer_idx in n_idx.neighbor_iter(&self.neighbor_offsets, neighbors.unknown()) {
                if !self.cells[infer_idx].value.is_unknown() {
                    failure(self, pre_change, source_idx, n_idx, infer_idx, new_value);
                }
            }
            for infer_idx in n_idx.neighbor_iter(&self.neighbor_offsets, neighbors.mines()) {
                if !self.cells[infer_idx].value.is_mine() {
                    failure(self, pre_change, source_idx, n_idx, infer_idx, new_value);
                }
            }
            for infer_idx in n_idx.neighbor_iter(&self.neighbor_offsets, neighbors.inferred()) {
                if !self.cells[infer_idx].value.is_inferred() {
                    failure(self, pre_change, source_idx, n_idx, infer_idx, new_value);
                }
            }
            for infer_idx in
                n_idx.neighbor_iter(&self.neighbor_offsets, DirectionSet::ALL - neighbors.all())
            {
                if !self.cells[infer_idx].value.is_wall() {
                    failure(self, pre_change, source_idx, n_idx, infer_idx, new_value);
                }
            }
        }
    }

    pub(super) fn compact_partition(&mut self) {
        let mut id_map =
            HashMap::with_capacity_and_hasher(self.partitions.len(), FastNumberHasherBuilder);
        let mut new_partitions =
            HashMap::with_capacity_and_hasher(self.partitions.len(), FastNumberHasherBuilder);
        let mut new_pid = PartitionId::ONE;
        for (old_pid, partition) in self.partitions.drain() {
            id_map.insert(old_pid, new_pid);
            new_partitions.insert(new_pid, partition);
            new_pid = new_pid.inc();
        }
        self.partitions = new_partitions;
        for cell in self.cells.iter_mut() {
            if cell.partition_id > PartitionId::ZERO {
                cell.partition_id = id_map
                    .get(&cell.partition_id)
                    .copied()
                    .unwrap_or(PartitionId::ZERO);
            }
        }
        self.next_partition_id = new_pid;
    }

    fn solve(&mut self) {
        #[cfg(feature = "debug")]
        use crate::solver::display::{ColoredMineField, DetailMineField};

        #[cfg(feature = "debug")]
        let debug = true;

        if self.next_partition_id > PartitionId::COMPACT_THRESHOLD {
            // in reality this should never happen, but just in case...
            self.compact_partition();
        }

        self.metrics.inc_solve_passes();
        self.metrics.mark_now();

        // index the new pending changes
        #[cfg(feature = "debug")]
        if debug {
            eprintln!("New Solve Pass ({})", self.metrics.solve_passes());
            eprintln!("Pre indexing:");
            DetailMineField::eprintln(self);
            ColoredMineField::eprintln(self);
        }
        self.index_pending();
        self.metrics.inc_indexing_duration();

        // process the "desolved" queue - i.e. reset all inferences attached to the provided cells
        #[cfg(feature = "debug")]
        if debug {
            eprintln!("Post indexing, Pre desolved:");
            DetailMineField::eprintln(self);
            ColoredMineField::eprintln(self);
        }
        self.process_desolved();
        self.metrics.inc_desolved_duration();

        // process the "solved" queue - i.e. find solved cells and mark their neighbors inferred
        #[cfg(feature = "debug")]
        if debug {
            eprintln!("Post desolved, Pre solved:");
            DetailMineField::eprintln(self);
            ColoredMineField::eprintln(self);
        }
        self.process_solved();
        self.metrics.inc_basic_solve_duration();

        // Create final partitions
        #[cfg(feature = "debug")]
        if debug {
            eprintln!("Post solved, Pre partition:");
            DetailMineField::eprintln(self);
            ColoredMineField::eprintln(self);
        }
        self.partition();
        self.metrics.inc_partition_duration();

        #[cfg(feature = "debug")]
        if debug {
            eprintln!("Post partition:");
            DetailMineField::eprintln(self);
            ColoredMineField::eprintln(self);
        }
    }

    fn index_pending(&mut self) {
        for i in 0..self.pending.len() {
            let IndexedCellValue { idx, value } = self.pending[i];
            self.change_cell_value(idx, value);
        }
        self.pending.clear();
    }

    fn process_desolved(&mut self) {
        if !self.desolved_queue.is_empty() {
            let mut queue = std::mem::take(&mut self.desolve_buf);
            queue.clear();
            let est_cap = 2 * (self.width_with_border + self.height_with_border) as usize;
            if queue.capacity() < est_cap {
                queue.reserve(est_cap);
            }
            while let Some(IndexedCell { idx, cell }) = self.desolved_queue.pop(&mut self.cells) {
                if cell.partition_id != self.next_partition_id {
                    self.enqueue_desolve(&mut queue, idx, cell.partition_id);
                    while let Some(idx) = queue.pop() {
                        self.desolve_opened(idx, &mut queue);
                    }
                }
            }
            self.desolve_buf = queue;
            self.next_partition_id = self.next_partition_id.inc();
        }
    }

    #[inline]
    fn enqueue_desolve(&mut self, queue: &mut Vec<CellIdx>, idx: CellIdx, pid: PartitionId) {
        if pid != PartitionId::ZERO {
            self.partitions.remove(&pid);
        }
        self.cells[idx].partition_id = self.next_partition_id;
        queue.push(idx);
    }

    fn desolve_opened(&mut self, idx: CellIdx, queue: &mut Vec<CellIdx>) {
        let neighbors = self.cells[idx].neighbors;
        let offsets = self.neighbor_offsets;
        #[cfg(feature = "debug")]
        {
            eprintln!("Desolving {idx} (inferred: {})", neighbors.inferred());
        }

        self.solved_queue.push(&mut self.cells, idx);

        self.metrics.inc_neighbor_visits(neighbors.unknown());
        for n_idx in idx.neighbor_iter(&offsets, neighbors.unknown()) {
            self.pending_partition_queue.push(&mut self.cells, n_idx);
        }

        // reset all inferred neighbors
        self.metrics.inc_neighbor_visits(neighbors.inferred());
        for inferred_idx in idx.neighbor_iter(&offsets, neighbors.inferred()) {
            let inferred_cell = self.cells[inferred_idx];
            if inferred_cell.partition_id != self.next_partition_id {
                self.cells[inferred_idx].partition_id = self.next_partition_id;
                let inferred_neighbors = inferred_cell.neighbors;
                #[cfg(feature = "debug")]
                {
                    eprintln!(
                        "Resetting inferred {inferred_idx} (opened: {})",
                        inferred_neighbors.opened()
                    );
                }

                self.pending_partition_queue
                    .push(&mut self.cells, inferred_idx);

                // reset inferred to unknown
                self.change_cell_value(inferred_idx, CellValue::UNKNOWN);

                // walk all opened neighbors
                self.metrics
                    .inc_neighbor_visits(inferred_neighbors.opened());
                for opened_idx in
                    inferred_idx.neighbor_iter(&offsets, inferred_neighbors.opened())
                {
                    let opened_cell = self.cells[opened_idx];
                    if opened_cell.partition_id != self.next_partition_id {
                        self.enqueue_desolve(queue, opened_idx, opened_cell.partition_id);
                    }
                }
            }
        }
    }

    fn process_solved(&mut self) {
        self.recent_inferred.clear();
        let offsets = self.neighbor_offsets;
        while let Some(idx) = self.solved_queue.pop_keep_mask() {
            let cell = self.cells[idx];
            self.pending_partition_queue.push(&mut self.cells, idx);
            if cell.value.is_opened() {
                let unknown_neighbors = cell.neighbors.unknown();
                if !unknown_neighbors.is_empty()
                    && let Some(value) = cell.value.infer_neighbors(cell.neighbors)
                {
                    self.metrics.inc_neighbor_visits(unknown_neighbors);
                    for n_idx in idx.neighbor_iter(&offsets, unknown_neighbors) {
                        self.change_cell_value(n_idx, value);
                        self.recent_inferred.push(n_idx);
                    }
                }
            }
            self.solved_queue.clear_mask(&mut self.cells, idx);
        }
    }

    fn partition(&mut self) {
        let start_partition_id = self.next_partition_id;
        self.recent_invalid.clear();
        if self.pending_partition_queue.is_empty() {
            return;
        }

        let mut opened_buf = std::mem::take(&mut self.partition_opened_buf);
        let mut unknown_buf = std::mem::take(&mut self.partition_unknown_buf);
        let est_cap = (self.width_with_border + self.height_with_border) as usize;
        if opened_buf.capacity() < est_cap {
            opened_buf.reserve(est_cap);
        }
        if unknown_buf.capacity() < est_cap {
            unknown_buf.reserve(est_cap);
        }

        while let Some(IndexedCell { idx, cell }) =
            self.pending_partition_queue.pop(&mut self.cells)
        {
            let is_opened = if cell.value.is_opened() {
                if cell.value.is_invalid(cell.neighbors) {
                    self.recent_invalid.push(idx);
                }
                if cell.neighbors.unknown().is_empty() {
                    if cell.partition_id != PartitionId::ZERO {
                        self.partitions.remove(&cell.partition_id);
                        self.cells[idx].partition_id = PartitionId::ZERO;
                    }
                    continue;
                }
                true
            } else if cell.value.is_unknown() {
                if cell.neighbors.opened().is_empty() {
                    if cell.partition_id != PartitionId::ZERO {
                        self.partitions.remove(&cell.partition_id);
                        self.cells[idx].partition_id = PartitionId::ZERO;
                    }
                    continue;
                }
                false
            } else {
                if cell.partition_id < start_partition_id {
                    self.cells[idx].partition_id = PartitionId::ZERO;
                }
                continue;
            };

            if cell.partition_id < start_partition_id {
                self.build_partition(
                    idx,
                    cell.partition_id,
                    is_opened,
                    &mut opened_buf,
                    &mut unknown_buf,
                );
            }
        }

        self.partition_opened_buf = opened_buf;
        self.partition_unknown_buf = unknown_buf;
    }

    fn build_partition(
        &mut self,
        root_idx: CellIdx,
        root_pid: PartitionId,
        is_opened: bool,
        opened_buf: &mut Vec<CellIdx>,
        unknown_buf: &mut Vec<CellIdx>,
    ) {
        opened_buf.clear();
        unknown_buf.clear();

        let next_pid = self.next_partition_id;
        let offsets = self.neighbor_offsets;
        self.cells[root_idx].partition_id = next_pid;
        if is_opened {
            if root_pid != PartitionId::ZERO {
                self.partitions.remove(&root_pid);
            }
            opened_buf.push(root_idx);
        } else {
            unknown_buf.push(root_idx);
        }

        let mut opened_cursor = 0;
        let mut unknown_cursor = 0;

        while opened_cursor < opened_buf.len() || unknown_cursor < unknown_buf.len() {
            while opened_cursor < opened_buf.len() {
                let idx = opened_buf[opened_cursor];
                opened_cursor += 1;
                let unknown = self.cells[idx].neighbors.unknown();
                self.metrics.inc_neighbor_visits(unknown);
                for n_idx in idx.neighbor_iter(&offsets, unknown) {
                    if self.cells[n_idx].partition_id != next_pid {
                        self.cells[n_idx].partition_id = next_pid;
                        unknown_buf.push(n_idx);
                    }
                }
            }
            while unknown_cursor < unknown_buf.len() {
                let idx = unknown_buf[unknown_cursor];
                unknown_cursor += 1;
                let opened = self.cells[idx].neighbors.opened();
                self.metrics.inc_neighbor_visits(opened);
                for n_idx in idx.neighbor_iter(&offsets, opened) {
                    let pid = self.cells[n_idx].partition_id;
                    if pid != next_pid {
                        if pid != PartitionId::ZERO {
                            self.partitions.remove(&pid);
                        }
                        self.cells[n_idx].partition_id = next_pid;
                        opened_buf.push(n_idx);
                    }
                }
            }
        }

        let len = (opened_buf.len() + unknown_buf.len()) as u32;
        if len <= 1 {
            self.cells[root_idx].partition_id = PartitionId::ZERO;
        } else {
            self.next_partition_id = next_pid.inc();
            self.metrics.inc_partitions_created();
            self.metrics.set_max_partition_size(len);
            self.partitions.insert(
                next_pid,
                Partition::new(opened_buf.as_slice(), unknown_buf.as_slice()),
            );
            self.metrics.set_max_partitions(self.partitions.len());
        }
    }

    /// Get the cell index for the given x/y coordinates.
    fn cell_idx(&self, Point { x, y }: Point) -> CellIdx {
        CellIdx(cell_idx(
            Point { x: x + 1, y: y + 1 },
            self.width_with_border,
        ))
    }

    fn coordinates(&self, idx: CellIdx) -> Point {
        let Point { x, y } = coordinates(idx.0, self.width_with_border);
        Point { x: x - 1, y: y - 1 }
    }

    /// Parses `field_spec` into an iterator of [`Cell`] updates, omitting skipped (`'.'` and `' '`) positions.
    fn cells_from_spec(field_spec: &str) -> impl Iterator<Item = UserCell> {
        Self::to_cell_iter(Self::spec_to_iter(field_spec))
    }

    /// Parses `field_spec` and returns the inferred `(width, height)` board dimensions along with an
    /// iterator of [`Cell`] updates for all non-skipped positions.
    pub(self) fn cells_and_size_from_spec(
        field_spec: &str,
    ) -> (u32, u32, impl Iterator<Item = UserCell> + use<>) {
        let cells: Vec<_> = Self::spec_to_iter(field_spec).collect();

        let size = cells
            .iter()
            .map(|(Point { x, y }, _)| (x + 1, y + 1))
            .fold((0, 0), |acc, r| (max(acc.0, r.0), max(acc.1, r.1)));

        let cells = Self::to_cell_iter(cells);

        (size.0, size.1, cells)
    }

    /// Converts `(x, y, Option<CellValue>)` tuples into [`Cell`]s, filtering out `None` entries.
    fn to_cell_iter(
        iter: impl IntoIterator<Item = (Point, Option<UserCellValue>)>,
    ) -> impl Iterator<Item = UserCell> {
        iter.into_iter()
            .filter_map(move |(point, v)| v.map(|value| UserCell { point, value }))
    }

    /// Parses `field_spec` into `(x, y, Option<CellValue>)` tuples for every character position in the spec.
    fn spec_to_iter(field_spec: &str) -> impl Iterator<Item = (Point, Option<UserCellValue>)> {
        fn char_to_value(c: char) -> Option<UserCellValue> {
            match c {
                '*' => Some(UserCellValue::Mine),
                n @ '0'..='8' => Some(UserCellValue::Opened(n as u8 - b'0')),
                '_' | 'x' | '#' => Some(UserCellValue::Unknown),
                '@' => Some(UserCellValue::Wall),
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
        for y in 0..self.height() {
            for x in 0..self.width() {
                let UserValueWithInference { value, inferred } = self.cell(Point { x, y });
                let char = match (value, inferred) {
                    (UserCellValue::Mine, _) => '*',
                    (UserCellValue::Wall, _) => '@',
                    (UserCellValue::Opened(n), _) => char::from(b'0' + n),
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

impl MineField {
    /// Rebuilds a fresh [`MineField`] from the non-inferred known cells of this field and solves it from scratch.
    #[cfg(test)]
    pub(super) fn rebuild_from_scratch(&self) -> Self {
        let mut cells = self.cells.clone();
        let mut pending = Vec::with_capacity(self.pending.capacity());
        for y in 0..self.height() {
            for x in 0..self.width() {
                let idx = self.cell_idx(Point { x, y });
                let cell = cells[idx].value;
                if !cell.is_inferred() && cell != CellValue::UNKNOWN {
                    pending.push(IndexedCellValue { idx, value: cell });
                }
                cells[idx].value = CellValue::UNKNOWN;
            }
        }
        for cell in cells.iter_mut() {
            cell.neighbors.reset();
            cell.partition_id = PartitionId::ZERO;
            cell.flags = CellFlags::new();
        }

        let mut field = Self {
            width_with_border: self.width_with_border,
            height_with_border: self.height_with_border,
            neighbor_offsets: self.neighbor_offsets,
            mine_count: self.mine_count,
            cells,
            partitions: self.partitions.clone(),
            next_partition_id: PartitionId::ONE,
            pending,
            desolved_queue: self.desolved_queue.clone(),
            desolve_buf: Vec::new(),
            partition_opened_buf: Vec::new(),
            partition_unknown_buf: Vec::new(),
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
    /// The requested board dimensions include a zero width or height.
    AreaTooSmall,
    /// The requested mine count is greater than the total number of cells.
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
                "The mine field area ({width} x {height}) must be below {} (with a little padding for borders).",
                MAX_SIZE
            ),
            SolverError::AreaTooSmall => write!(f, "The width and height must both be >= 1"),
            SolverError::MineCountTooLarge {
                mine_count,
                width,
                height,
            } => {
                write!(
                    f,
                    "The mine count ({mine_count}) must be less than or equal to the area of the mine field ({width} x {height})."
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
        grid::{DirectionSet, E, NE, NW, S, SE, W},
    };

    /// Asserts structural invariants across the 4 neighbor bitmasks in [`CellNeighbors`]:
    /// - `unknown`, `opened`, `mines`, and `inferred` are subsets of `all()`.
    /// - `unknown`, `opened`, and `mines` are pairwise disjoint.
    /// - `inferred` cannot overlap with `unknown` or `opened` (only `Cleared` and `Flag` are inferred).
    fn assert_neighbor_invariants(neighbors: CellNeighbors) {
        let all_neighbors = neighbors.all();
        // all direction sets are subsets of all_neighbors
        assert_eq!(neighbors.unknown() | all_neighbors, all_neighbors);
        assert_eq!(neighbors.mines() | all_neighbors, all_neighbors);
        assert_eq!(neighbors.opened() | all_neighbors, all_neighbors);
        assert_eq!(neighbors.inferred() | all_neighbors, all_neighbors);

        // unknown, opened, and mines are pairwise disjoint
        assert_eq!(neighbors.mines() & neighbors.unknown(), DirectionSet::EMPTY);
        assert_eq!(neighbors.opened() & neighbors.mines(), DirectionSet::EMPTY);
        assert_eq!(
            neighbors.opened() & neighbors.unknown(),
            DirectionSet::EMPTY
        );

        // no such thing as an "inferred unknown" or an "inferred opened"
        assert_eq!(
            neighbors.inferred() & neighbors.unknown(),
            DirectionSet::EMPTY
        );
        assert_eq!(
            neighbors.opened() & neighbors.inferred(),
            DirectionSet::EMPTY
        );
    }

    fn assert_partition_invariants(field: &MineField) {
        let mut partition_sizes: HashMap<PartitionId, u16> = HashMap::new();
        for cell in &field.cells {
            if cell.partition_id != PartitionId::ZERO {
                assert!(cell.partition_id < field.next_partition_id);
                *partition_sizes.entry(cell.partition_id).or_insert(0) += 1;
            }
        }
        for (&pid, partition) in &field.partitions {
            assert_ne!(pid, PartitionId::ZERO);
            assert_eq!(partition_sizes.get(&pid), Some(&partition.len()));
        }
    }

    fn assert_all_invariants(field: &MineField) {
        for cell in field.cells.iter() {
            assert_neighbor_invariants(cell.neighbors);
        }
        assert_partition_invariants(field);
    }

    fn partition_cell_count(field: &MineField, pid: PartitionId) -> usize {
        field.cells.iter().filter(|c| c.partition_id == pid).count()
    }

    fn change_cell_value(field: &mut MineField, idx: CellIdx, value: CellValue) {
        field.change_cell_value(idx, value);
    }

    #[test]
    fn test_cell_change_value_updates_neighbor_masks() {
        // 3x1 strip, change the middle cell and watch both neighbors
        let mut field = MineField::new(3, 1).unwrap();
        let c0 = field.cell_idx(Point::new(0, 0));
        let c1 = field.cell_idx(Point::new(1, 0));
        let c2 = field.cell_idx(Point::new(2, 0));

        change_cell_value(&mut field, c1, CellValue::MINE);
        assert_eq!(field.cells[c1].value, CellValue::MINE);
        assert_eq!(field.cells[c0].neighbors.mines(), !!E);
        assert!(field.cells[c0].neighbors.opened().is_empty());
        assert!(field.cells[c0].neighbors.inferred().is_empty());
        assert_eq!(field.cells[c2].neighbors.mines(), !!W);
        assert!(field.cells[c2].neighbors.opened().is_empty());
        assert!(field.cells[c2].neighbors.inferred().is_empty());

        // Mine -> Safe(1): neighbors drop the mine but stay known (not unknown)
        change_cell_value(&mut field, c1, CellValue::for_mine_count(1));
        assert!(field.cells[c0].neighbors.mines().is_empty());
        assert!(field.cells[c0].neighbors.inferred().is_empty());
        assert_eq!(field.cells[c0].neighbors.opened(), !!E);

        // Safe(1) -> Unknown: neighbors regain an unknown neighbor
        change_cell_value(&mut field, c1, CellValue::UNKNOWN);
        assert!(field.cells[c0].neighbors.mines().is_empty());
        assert!(field.cells[c0].neighbors.opened().is_empty());
        assert!(field.cells[c2].neighbors.mines().is_empty());
        assert!(field.cells[c2].neighbors.opened().is_empty());
    }

    #[test]
    fn test_streamlined_cell_value_transitions() {
        // Grid 3x1: [0, 1, 2]
        let mut field = MineField::new(3, 1).unwrap();
        let c0 = field.cell_idx(Point::new(0, 0));
        let c1 = field.cell_idx(Point::new(1, 0));
        let c2 = field.cell_idx(Point::new(2, 0));

        // 1. Unknown -> Opened(1) on cell 0
        change_cell_value(&mut field, c0, CellValue::for_mine_count(1));
        assert_eq!(field.cells[c0].value, CellValue::for_mine_count(1));
        // Cell 0's `unknown` mask points to Cell 1 (E), and Cell 1's `opened` mask points to Cell 0 (W)
        assert_eq!(field.cells[c0].neighbors.unknown(), !!E);
        assert_eq!(field.cells[c1].neighbors.opened(), !!W);

        // 2. Unknown -> Cleared on cell 2
        change_cell_value(&mut field, c2, CellValue::CLEARED);
        assert_eq!(field.cells[c2].value, CellValue::CLEARED);
        // Cell 1's `inferred` mask points to Cell 2 (E)
        assert_eq!(field.cells[c1].neighbors.inferred(), !!E);

        // 3. Cleared -> Opened(0) on cell 2
        change_cell_value(&mut field, c2, CellValue::for_mine_count(0));
        assert_eq!(field.cells[c2].value, CellValue::for_mine_count(0));
        // Cell 1 drops Cell 2 from `inferred` and adds Cell 2 to `opened`
        assert!(field.cells[c1].neighbors.inferred().is_empty());
        assert_eq!(field.cells[c1].neighbors.opened(), W | E);

        // 4. Unknown -> Mine on cell 1
        change_cell_value(&mut field, c1, CellValue::MINE);
        assert_eq!(field.cells[c1].value, CellValue::MINE);
        // Both cell 0 and cell 2 drop cell 1 from `unknown` and set cell 1 in `mines`
        assert!(field.cells[c0].neighbors.unknown().is_empty());
        assert_eq!(field.cells[c0].neighbors.mines(), !!E);
        assert!(field.cells[c2].neighbors.unknown().is_empty());
        assert_eq!(field.cells[c2].neighbors.mines(), !!W);

        for cell in field.cells.iter() {
            assert_neighbor_invariants(cell.neighbors);
        }
    }

    #[test]
    fn test_wall_cells_creation_and_solving() {
        // 3x2 board:
        // 1 @ .
        // . * .
        // (0,0) has count 1.
        // Neighbors of (0,0) are (1,0)=Wall, (0,1)=Unknown, (1,1)=Mine.
        // Because (1,0) is a Wall, it is neither an unknown nor a mine neighbor.
        // The Mine at (1,1) satisfies the count 1 at (0,0), so (0,1) must be inferred CLEAR!
        let field = MineField::create(
            "1@.
             .*.",
        );
        assert_eq!(field.cell(Point { x: 1, y: 0 }).value, UserCellValue::Wall);
        assert_eq!(
            field.cells[field.cell_idx(Point { x: 1, y: 0 })].value,
            CellValue::WALL
        );
        assert!(
            field.cells[field.cell_idx(Point { x: 0, y: 1 })]
                .value
                .is_safe()
        );

        // 3x1 board:
        // 1 @ _
        // (0,0) is count 1. Neighbor (1,0) is Wall. (2,0) is Unknown but not adjacent to (0,0).
        // (0,0) needs 1 mine but has 0 available unknown neighbors -> invalid!
        let field = MineField::create("1@_");
        assert_eq!(field.recent_invalid().next(), Some(Point { x: 0, y: 0 }));
    }

    #[test]
    fn test_wall_cell_value_transitions() {
        let mut field = MineField::new(3, 1).unwrap();
        let c1 = field.cell_idx(Point::new(1, 0));

        // Unknown -> Wall is non-breaking
        change_cell_value(&mut field, c1, CellValue::WALL);
        assert_eq!(field.cells[c1].value, CellValue::WALL);

        // Wall -> Unknown is Breaking
        change_cell_value(&mut field, c1, CellValue::UNKNOWN);
        assert_eq!(field.cells[c1].value, CellValue::UNKNOWN);

        for cell in field.cells.iter() {
            assert_neighbor_invariants(cell.neighbors);
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
            UserCellValue::Opened(1)
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
            UserCellValue::Opened(1)
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
            let idx = field.cell_idx(pt);
            assert!(field.cells[idx].value.is_safe());
        }
    }

    #[test]
    fn test_create_minefield_all_supported_characters() {
        // Test '0'..='8', '*', 'x', '#', ' ', '.', '_', and '@'
        //  ..***
        //  ..78*
        //  *.***
        //  @....
        let field = MineField::create(
            "01234
             5678*
             *x#._
             @....",
        );
        assert_eq!(field.width(), 5);
        assert_eq!(field.height(), 4);
        assert_eq!(field.mine_count(), None);

        for x in 0..5u32 {
            assert_eq!(
                field.cell(Point { x, y: 0 }).value,
                UserCellValue::Opened(x as u8)
            );
        }
        for x in 0..4u32 {
            assert_eq!(
                field.cell(Point { x, y: 1 }).value,
                UserCellValue::Opened((x + 5) as u8)
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

        assert_eq!(field.cell(Point { x: 0, y: 3 }).value, UserCellValue::Wall);
        for x in 1..5u32 {
            assert_eq!(field.cell(Point { x, y: 3 }).value, UserCellValue::Unknown);
        }
    }

    #[test]
    fn test_cells_from_spec_and_update_minefield() {
        // Verify ' ' and '.' are filtered out while '_', 'x', '#', '*', '@', and digits emit Cells.
        // Leading unspecified cell must use '.' because leading spaces are trimmed.
        let cells: Vec<UserCell> = MineField::cells_from_spec(". 1_x#*@").collect();
        assert_eq!(
            cells,
            vec![
                UserCell {
                    point: Point { x: 2, y: 0 },
                    value: UserCellValue::Opened(1)
                },
                UserCell {
                    point: Point { x: 3, y: 0 },
                    value: UserCellValue::Unknown
                },
                UserCell {
                    point: Point { x: 4, y: 0 },
                    value: UserCellValue::Unknown
                },
                UserCell {
                    point: Point { x: 5, y: 0 },
                    value: UserCellValue::Unknown
                },
                UserCell {
                    point: Point { x: 6, y: 0 },
                    value: UserCellValue::Mine
                },
                UserCell {
                    point: Point { x: 7, y: 0 },
                    value: UserCellValue::Wall
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
            UserCellValue::Opened(1)
        );
        assert_eq!(
            field.cell(Point { x: 2, y: 0 }).value,
            UserCellValue::Opened(2)
        );

        // Update with '.' at (0, 0) and (2, 0) (preserving their values),
        // '*' at (1, 0), '_' at (3, 1), and '1' at (0, 1)
        field.update(
            ".*..
             1.._",
        );
        assert_eq!(
            field.cell(Point { x: 0, y: 0 }).value,
            UserCellValue::Opened(1)
        );
        assert_eq!(field.cell(Point { x: 1, y: 0 }).value, UserCellValue::Mine);
        assert_eq!(
            field.cell(Point { x: 2, y: 0 }).value,
            UserCellValue::Opened(2)
        );
        assert_eq!(
            field.cell(Point { x: 0, y: 1 }).value,
            UserCellValue::Opened(1)
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
            UserCellValue::Opened(1)
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
            UserCellValue::Opened(1)
        );
        assert_eq!(ragged.cell(Point { x: 1, y: 0 }).value, UserCellValue::Mine);
        assert_eq!(
            ragged.cell(Point { x: 4, y: 0 }).value,
            UserCellValue::Unknown
        );
        assert_eq!(
            ragged.cell(Point { x: 4, y: 1 }).value,
            UserCellValue::Opened(2)
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
            UserCellValue::Opened(1)
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
            UserCellValue::Opened(1)
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
        for cell in field.cells.iter() {
            let neighbors = cell.neighbors;
            assert!(neighbors.opened().is_empty());
            assert!(neighbors.inferred().is_empty());
            assert!(neighbors.mines().is_empty());
            assert_neighbor_invariants(neighbors);
        }

        // AreaTooLarge
        let err = MineField::new(u32::MAX, u32::MAX).unwrap_err();
        assert!(matches!(err, SolverError::AreaTooLarge { .. }));
        assert!(err.to_string().contains("must be below"));

        // MineCountTooLarge
        let err = MineField::with_mine_count(4, 4, 17).unwrap_err();
        assert!(matches!(
            err,
            SolverError::MineCountTooLarge {
                mine_count: 17,
                width: 4,
                height: 4
            }
        ));
        assert!(err.to_string().contains("must be less than or equal to"));

        // 0x0 field rejected as AreaTooSmall
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
                value: UserCellValue::Opened(1),
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
                value: UserCellValue::Opened(1),
            })
            .unwrap_err();
        assert!(matches!(err_y, SolverError::CoordinatesOutOfBounds { .. }));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_update_x_out_of_bounds_panics() {
        let mut field = MineField::new(2, 2).unwrap();
        field.update("..1");
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_update_y_out_of_bounds_panics() {
        let mut field = MineField::new(2, 2).unwrap();
        field.update("..\n..\n1");
    }

    fn cell(field: &MineField, x: u32, y: u32) -> (CellValue, CellNeighbors) {
        let idx = field.cell_idx(Point { x, y });
        (field.cells[idx].value, field.cells[idx].neighbors)
    }

    #[test]
    fn test_neighbor_sets_and_remaining_mines_tracking() {
        // Reveal (0, 0) as Count(2) on a 3x3 board
        let mut field = MineField::create(
            "2..
             ...
             ...",
        );
        let (c00, n00) = cell(&field, 0, 0);
        assert_eq!(c00.remaining_mines(n00), 2);
        assert_eq!(n00.unknown(), E | S | SE);
        assert!(n00.mines().is_empty());
        assert_neighbor_invariants(n00);

        // Neighbor (1, 0) is Unknown and moves W from unknown to opened
        let (_, n10) = cell(&field, 1, 0);
        assert_eq!(n10.opened(), !!W);
        assert!(n10.mines().is_empty());
        assert_neighbor_invariants(n10);

        // Mark (1, 0) as a Mine: (0, 0) moves E from unknown to
        // mines (remaining_mines == 1)
        field.update(".*");
        let (c00, n00) = cell(&field, 0, 0);
        assert_eq!(n00.mines(), !!E);
        assert_eq!(n00.unknown(), S | SE);
        assert_eq!(c00.remaining_mines(n00), 1);
        assert_neighbor_invariants(n00);

        // Change (1, 0) from Mine to Opened(1): (0, 0) moves E from mines to opened.
        // Because (0, 0) is Opened(2) and now has only 2 unknown neighbors left (S and SE),
        // process_solved infers both S and SE are mines, reducing remaining_mines() to 0.
        field.update(".1");
        let (c00, n00) = cell(&field, 0, 0);
        assert!(n00.unknown().is_empty());
        assert_eq!(n00.mines(), S | SE);
        assert_eq!(c00.remaining_mines(n00), 0);
        assert_neighbor_invariants(n00);

        // Revert (1, 0) back to Unknown (inference-breaking): (0, 0) moves E from opened
        // back to unknown, and the inferences on S and SE are cleared.
        field.update("._");
        let (c00, n00) = cell(&field, 0, 0);
        assert!(n00.mines().is_empty());
        assert_eq!(n00.unknown(), E | S | SE);
        assert_eq!(c00.remaining_mines(n00), 2);
        assert_neighbor_invariants(n00);
    }

    #[test]
    fn test_partitioning_and_repartitioning_across_updates() {
        // 5x2 board with Opened(1) at top-left (0, 0) and top-right (4, 0)
        // Each corner has 3 unknown neighbors (choose(3, 1) = 3 > 1), so neither is trivially solved.
        let mut field = MineField::create(
            "1...1
             .....",
        );
        let c00 = field.cell_idx(Point::new(0, 0));
        let c40 = field.cell_idx(Point::new(4, 0));
        let c20 = field.cell_idx(Point::new(2, 0));
        let c10 = field.cell_idx(Point::new(1, 0));
        let c30 = field.cell_idx(Point::new(3, 0));

        // Two disconnected frontier components:
        // Left: {(0,0), (1,0), (0,1), (1,1)} (4 cells)
        // Right: {(4,0), (3,0), (4,1), (3,1)} (4 cells)
        assert_eq!(field.partitions.len(), 2);
        let p_left = field.cells[c00].partition_id;
        let p_right = field.cells[c40].partition_id;
        assert_ne!(p_left, PartitionId::ZERO);
        assert_ne!(p_right, PartitionId::ZERO);
        assert_ne!(p_left, p_right);
        assert_eq!(partition_cell_count(&field, p_left), 4);
        assert_eq!(partition_cell_count(&field, p_right), 4);
        assert_eq!(field.cells[c20].partition_id, PartitionId::ZERO);
        assert!(field.cells[c20].neighbors.opened().is_empty());
        assert!(field.cells[c20].neighbors.inferred().is_empty());
        assert_all_invariants(&field);

        // Reveal the middle top cell (2, 0) as Opened(2): now both components are bridged
        // into a single 10-cell partition!
        field.update("..2");
        assert_eq!(field.partitions.len(), 1);
        let merged_pid = field.cells[c20].partition_id;
        assert_ne!(merged_pid, PartitionId::ZERO);
        assert_eq!(partition_cell_count(&field, merged_pid), 10);
        assert_eq!(field.cells[c10].neighbors.opened(), W | E);
        assert_eq!(field.cells[c30].neighbors.opened(), W | E);
        assert_all_invariants(&field);
    }

    #[test]
    fn test_partition_split_with_distant_non_dirty_cells() {
        // 5x2 board: reveal (1, 0) = Opened(1) and (3, 0) = Opened(1).
        // Initially all 10 cells are connected via (2, 0) and (2, 1) into a single partition.
        let mut field = MineField::create(
            ".1.1.
             .....",
        );
        let c00 = field.cell_idx(Point::new(0, 0));
        let c10 = field.cell_idx(Point::new(1, 0));
        let c20 = field.cell_idx(Point::new(2, 0));
        let c30 = field.cell_idx(Point::new(3, 0));
        let c40 = field.cell_idx(Point::new(4, 0));
        let c01 = field.cell_idx(Point::new(0, 1));
        let c41 = field.cell_idx(Point::new(4, 1));

        assert_eq!(field.partitions.len(), 1);
        let initial_pid = field.cells[c00].partition_id;
        assert_ne!(initial_pid, PartitionId::ZERO);
        for y in 0..field.height() {
            for x in 0..field.width() {
                assert_eq!(
                    field.cells[field.cell_idx(Point::new(x, y))].partition_id,
                    initial_pid
                );
            }
        }

        // Now reveal the bridge cell (2, 0) as Opened(0).
        // `process_solved` infers (1, 1), (2, 1), and (3, 1) as InferredValue::Safe,
        // and the post-`process_solved` partition pass splits the board into two 3-cell
        // partitions: {(0,0), (0,1), (1,0)} and {(3,0), (4,0), (4,1)}, while (2,0)
        // and its inferred-safe neighbors get partition_id == 0.
        field.update("..0");

        assert_eq!(field.partitions.len(), 2);
        assert!(field.cells[c20].neighbors.unknown().is_empty());

        let left_pid = field.cells[c00].partition_id;
        let right_pid = field.cells[c40].partition_id;
        assert_ne!(left_pid, PartitionId::ZERO);
        assert_ne!(right_pid, PartitionId::ZERO);
        assert_ne!(left_pid, right_pid);
        assert_eq!(field.cells[c10].partition_id, left_pid);
        assert_eq!(field.cells[c01].partition_id, left_pid);
        assert_eq!(field.cells[c30].partition_id, right_pid);
        assert_eq!(field.cells[c41].partition_id, right_pid);
        assert_eq!(partition_cell_count(&field, left_pid), 3);
        assert_eq!(partition_cell_count(&field, right_pid), 3);

        // Verify all cells and partitions remain consistent
        assert_all_invariants(&field);
    }

    #[test]
    fn test_partition_complete_dissolution() {
        // 3x1 board: [(0,0) = Unknown, (1,0) = Opened(1), (2,0) = Unknown]
        // (1,0) has 2 unknown neighbors (choose(2, 1) = 2 > 1), so 1 partition of 3 cells remains.
        let mut field = MineField::create(".1.");
        let c0 = field.cell_idx(Point::new(0, 0));
        let c1 = field.cell_idx(Point::new(1, 0));
        let c2 = field.cell_idx(Point::new(2, 0));

        assert_eq!(field.partitions.len(), 1);
        assert_eq!(field.cells[c1].neighbors.unknown(), W | E);
        assert_eq!(field.cells[c0].neighbors.opened(), !!E);
        assert_eq!(field.cells[c2].neighbors.opened(), !!W);

        // Flag (0, 0) as a Mine: basic_solve infers (2, 0) is Safe, and the post-basic_solve
        // repartition pass dissolves the partition completely.
        field.update("*");

        assert!(field.partitions.is_empty());
        assert!(field.cells[c2].value.is_safe());
        assert_eq!(
            field.cells[c1]
                .value
                .remaining_mines(field.cells[c1].neighbors),
            0
        );
        assert!(field.cells[c1].neighbors.unknown().is_empty());
        assert_eq!(field.cells[c1].neighbors.mines(), !!W);

        for cell in field.cells.iter() {
            assert_neighbor_invariants(cell.neighbors);
        }
    }

    #[test]
    fn test_batch_adjacent_cells_with_shared_opened_neighbor() {
        // 3x2 board: reveal (1, 0) as Opened(2) first
        let mut field = MineField::create(
            ".2.
             ...",
        );
        let c10 = field.cell_idx(Point::new(1, 0));
        assert_eq!(
            field.cells[c10]
                .value
                .remaining_mines(field.cells[c10].neighbors),
            2
        );

        // In one batch, set both (0, 0) and (2, 0) to Mine, and (1, 1) to Opened(2).
        // (1, 0) neighbors all three pending cells, so dedup keeps it queued at most once.
        field.update(
            "*.*
             .2.",
        );

        // Both (1, 0) and (1, 1) are adjacent to the two mines at (0, 0) and (2, 0),
        // so both have remaining_mines == 0, process_solved infers (0, 1) and (2, 1) as Safe,
        // and the post-process_solved partition pass dissolves the solved partition.
        let (c10, n10) = cell(&field, 1, 0);
        let (c11, n11) = cell(&field, 1, 1);
        assert_eq!(c10.remaining_mines(n10), 0);
        assert_eq!(c11.remaining_mines(n11), 0);

        assert_eq!(n10.mines(), W | E);
        assert!(n10.unknown().is_empty());

        assert_eq!(n11.mines(), NW | NE);
        assert!(n11.unknown().is_empty());

        assert!(field.partitions.is_empty());
        assert_all_invariants(&field);
    }

    #[test]
    fn test_repeated_and_reverted_staging_on_same_cell() {
        let mut field = MineField::create(".1.");
        let c1 = field.cell_idx(Point::new(1, 0));
        assert_eq!(
            field.cells[c1]
                .value
                .remaining_mines(field.cells[c1].neighbors),
            1
        );

        // Stage (0, 0) as Mine -> revert to Unknown -> stage again as Mine in one batch.
        // Neighbor (1, 0)'s mines and unknown masks must update cleanly.
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
        assert_eq!(field.cells[c1].neighbors.mines(), !!W);
        // process_solved now infers (2,0) is safe. Inferred cells are folded into the neighbor
        // masks, so it moves out of `unknown`.
        assert_eq!(
            field.cell(Point { x: 2, y: 0 }).inferred,
            InferredValue::Safe
        );
        assert!(field.cells[c1].neighbors.unknown().is_empty());
        assert_eq!(
            field.cells[c1]
                .value
                .remaining_mines(field.cells[c1].neighbors),
            0
        );
        assert_all_invariants(&field);
    }

    #[test]
    fn test_basic_solve_mine_and_safe_inferences_and_cascading() {
        // 4x1 strip: [(0,0)=Opened(1), (1,0)=Unknown, (2,0)=Opened(1), (3,0)=Unknown]
        // 1. (0,0)=Opened(1) has only 1 unknown neighbor (1,0) -> infers (1,0) is Mine.
        // 2. That satisfies (2,0)=Opened(1), which then infers (3,0) is Safe!
        let field = MineField::create("1.1.");
        let c0 = field.cell_idx(Point::new(0, 0));
        let c1 = field.cell_idx(Point::new(1, 0));
        let c2 = field.cell_idx(Point::new(2, 0));
        let c3 = field.cell_idx(Point::new(3, 0));

        assert!(field.cells[c1].value.is_mine());
        assert!(field.cells[c3].value.is_safe());

        // Verify inferred neighbor sets reflect the deductions
        assert_eq!(field.cells[c0].neighbors.mines(), !!E);
        assert!(field.cells[c0].neighbors.unknown().is_empty());
        assert_eq!(field.cells[c2].neighbors.mines(), !!W);
        assert!(field.cells[c2].neighbors.unknown().is_empty());

        assert_all_invariants(&field);
    }

    #[test]
    fn test_basic_solve_cascade_multi_step_chain() {
        // 6x1 strip: [(0,0)=1, (1,0)=., (2,0)=2, (3,0)=., (4,0)=1, (5,0)=.]
        // Hop 1: (0,0)=1 has 1 unknown neighbor (1,0) -> infers (1,0) is Mine.
        // Hop 2: (2,0)=2 now has 1 mine at (1,0), so 1 remaining mine for (3,0) -> infers (3,0) is Mine.
        // Hop 3: (4,0)=1 now has 1 mine at (3,0), so 0 remaining mines for (5,0) -> infers (5,0) is Safe.
        let field = MineField::create("1.2.1.");
        let c1 = field.cell_idx(Point::new(1, 0));
        let c3 = field.cell_idx(Point::new(3, 0));
        let c5 = field.cell_idx(Point::new(5, 0));

        assert!(field.cells[c1].value.is_mine() && field.cells[c1].value.is_inferred());
        assert!(field.cells[c3].value.is_mine() && field.cells[c3].value.is_inferred());
        assert!(field.cells[c5].value.is_safe() && field.cells[c5].value.is_inferred());

        assert_all_invariants(&field);
    }

    #[test]
    fn test_basic_solve_detects_impossible_contradictory_clues() {
        // 3x1 strip: [(0,0)=Opened(0), (1,0)=Unknown, (2,0)=Opened(1)]
        // (0,0)=Opened(0) infers (1,0) is Safe, leaving (2,0)=Opened(1) with 0 available cells
        // and 1 remaining mine.
        let field = MineField::create("0.1");
        let c0 = field.cell_idx(Point::new(0, 0));
        let c2 = field.cell_idx(Point::new(2, 0));

        assert_ne!(
            field.cells[c0].value.is_invalid(field.cells[c0].neighbors),
            field.cells[c2].value.is_invalid(field.cells[c2].neighbors)
        );
        let invalid: Vec<_> = field.recent_invalid().collect();
        assert_eq!(invalid.len(), 1);
        assert_all_invariants(&field);
    }

    #[test]
    fn test_inference_compatible_update_preserves_inferences() {
        // 4x1 strip: [(0,0)=Opened(1), (1,0)=Unknown, (2,0)=Unknown, (3,0)=Unknown]
        // Round 1: Reveal (0,0)=Opened(1) -> infers (1,0) is Mine, and updates (2,0)'s
        // `mines` neighbor mask to include W.
        let mut field = MineField::create("1...");
        let c1 = field.cell_idx(Point::new(1, 0));
        let c2 = field.cell_idx(Point::new(2, 0));
        let c3 = field.cell_idx(Point::new(3, 0));
        assert!(field.cells[c1].value.is_mine());
        assert_eq!(field.cells[c2].neighbors.mines(), !!W);

        // Round 2 (inference-compatible): Reveal (2,0) as Opened(1).
        // Because (1,0)'s Mine inference is preserved, (2,0) immediately knows its 1 mine is at W
        // and infers (3,0) is Safe!
        field.update("..1");
        assert!(field.cells[c1].value.is_mine());
        assert!(field.cells[c3].value.is_safe());
        assert_all_invariants(&field);
    }

    #[test]
    fn test_inference_breaking_updates_clear_and_recompute_inferences() {
        // 3x1 strip: [(0,0), (1,0), (2,0)]
        // Round 1: Reveal (1,0) as Opened(2) -> both (0,0) and (2,0) are inferred as Mine.
        let mut field = MineField::create(".2.");
        let c0 = field.cell_idx(Point::new(0, 0));
        let c1 = field.cell_idx(Point::new(1, 0));
        let c2 = field.cell_idx(Point::new(2, 0));
        assert!(field.cells[c0].value.is_mine());
        assert!(field.cells[c2].value.is_mine());

        // Round 2 (Inference-breaking Case A): Change (1,0) from Opened(2) to Opened(1).
        // Neither (0,0) nor (2,0) can be inferred anymore; both reset to InferredValue::None.
        field.update(".1");
        assert!(!field.cells[c0].value.is_inferred());
        assert!(!field.cells[c2].value.is_inferred());
        assert_eq!(field.cells[c1].neighbors.unknown(), W | E);
        assert!(field.cells[c1].neighbors.mines().is_empty());
        assert_all_invariants(&field);

        // Round 3: Change (1,0) to Opened(0) -> both (0,0) and (2,0) are inferred as Safe.
        field.update(".0");
        assert!(field.cells[c0].value.is_safe());
        assert!(field.cells[c2].value.is_safe());

        // Round 4 (Inference-breaking Case B): Directly violate inference on (0,0) by setting it to Mine,
        // and simultaneously change (1,0) to Opened(1).
        // (2,0) must remain/be recomputed as Safe, and all neighbor/partition invariants must hold.
        field.update("*1");
        assert!(!field.cells[c0].value.is_inferred());
        assert!(field.cells[c2].value.is_safe());
        assert_all_invariants(&field);

        // Round 5 (Inference-breaking Case C): Dissolve the partition completely by resetting
        // (0,0) and (1,0) to Unknown. Verify (2,0)'s Safe inference is cleared even though no
        // new partition is created!
        field.update("__");
        assert!(field.partitions.is_empty());
        for cell in field.cells.iter() {
            assert!(!cell.value.is_inferred());
        }
        assert_all_invariants(&field);
    }

    #[test]
    fn test_inference_breaking_update_preserves_untouched_partitions() {
        // 5x1 strip: [(0,0)=Opened(1), (1,0)=Unknown, (2,0)=Unknown, (3,0)=Unknown, (4,0)=Opened(1)]
        // Two independent partitions:
        // - Left partition {(0,0), (1,0)} infers (1,0) is Mine.
        // - Right partition {(3,0), (4,0)} infers (3,0) is Mine.
        let mut field = MineField::create("1...1");
        let c1 = field.cell_idx(Point::new(1, 0));
        let c3 = field.cell_idx(Point::new(3, 0));
        let c4 = field.cell_idx(Point::new(4, 0));
        assert!(field.cells[c1].value.is_mine());
        assert!(field.cells[c3].value.is_mine());
        let right_pid = field.cells[c4].partition_id;

        // Trigger an inference-breaking update on the LEFT partition only:
        // change (0,0) from Opened(1) to Opened(0).
        // Left cell (1,0) must change from InferredValue::Mine to InferredValue::Safe,
        // while the untouched RIGHT partition (and cell (3,0)'s Mine inference) remains intact!
        field.update("0");
        assert!(field.cells[c1].value.is_safe());
        assert!(field.cells[c3].value.is_mine());
        assert_eq!(field.cells[c4].partition_id, right_pid);
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
        assert_all_invariants(&field);
    }

    #[test]
    fn test_invalid_resolved_by_editing_other_cell_or_invalid_cell_itself() {
        // 3x1 strip: [(0,0)=Opened(0), (1,0)=Unknown, (2,0)=Opened(1)]
        // One of (0,0) or (2,0) infers (1,0) first, leaving the other in `recent_invalid`.
        let mut field = MineField::create("0.1");
        let c1 = field.cell_idx(Point::new(1, 0));
        let invalid: Vec<_> = field.recent_invalid().collect();
        // Solver is free to infer the center cell from either clue first.
        if field.cells[c1].value.is_mine() {
            assert_eq!(invalid, vec![(0, 0).into()]);
        } else {
            assert_eq!(invalid, vec![(2, 0).into()]);
        }

        // Case A: Resolve the inconsistency by editing (0,0) (NOT (2,0))
        // from Opened(0) to Opened(1). `recent_invalid` is now empty and (1,0) is Mine.
        field.update("1");
        assert!(field.recent_invalid().next().is_none());
        assert!(field.cells[c1].value.is_mine());
        assert_all_invariants(&field);

        // Re-introduce the inconsistency sequentially: set (0,0) back to Opened(0).
        // One of the conflicting clues is again recorded in `recent_invalid`.
        field.update("0");
        let invalid: Vec<_> = field.recent_invalid().collect();
        if field.cells[c1].value.is_mine() {
            assert_eq!(invalid, vec![(0, 0).into()]);
        } else {
            assert_eq!(invalid, vec![(2, 0).into()]);
            assert!(field.cells[c1].value.is_safe());
        }

        // Case B: Resolve the inconsistency by editing (2,0) itself from Opened(1) to Opened(0).
        field.update("..0");
        assert!(field.recent_invalid().next().is_none());
        assert!(field.cells[c1].value.is_safe());
        assert_all_invariants(&field);
    }

    #[test]
    fn test_invalid_sequential_discovery_and_untouched_partition_preservation() {
        // 7x1 strip with two independent contradictory regions:
        // Left:  [(0,0), (1,0), (2,0)]
        // Right: [(4,0), (5,0), (6,0)]
        // Step 1: Set (2,0) = Opened(1) (infers (1,0) = Mine),
        // and set (4,0) = Opened(0), (6,0) = Opened(1) (invalidates one of (4,0)/(6,0)).
        let mut field = MineField::create("..1.0.1");
        let c1 = field.cell_idx(Point::new(1, 0));
        let c4 = field.cell_idx(Point::new(4, 0));
        let c6 = field.cell_idx(Point::new(6, 0));
        assert!(field.cells[c1].value.is_mine());
        assert_ne!(
            field.cells[c4].value.is_invalid(field.cells[c4].neighbors),
            field.cells[c6].value.is_invalid(field.cells[c6].neighbors)
        );
        assert_eq!(field.recent_invalid().count(), 1);

        // Step 2: Reveal (0,0) = Opened(0) on the left side.
        // Its Opened(0) conflicts with (2,0)=Opened(1), so the left solve pass produces an invalid cell
        // while the untouched right side keeps its contradiction.
        field.update("0");
        assert_eq!(field.recent_invalid().count(), 1);
        assert_ne!(
            field.cells[c4].value.is_invalid(field.cells[c4].neighbors),
            field.cells[c6].value.is_invalid(field.cells[c6].neighbors)
        );

        // Step 3: Resolve ONLY the left side by setting (0,0) = Opened(1) and (1,0) = Mine.
        // Because `recent_invalid` only tracks the latest solve pass (which touched only the left side),
        // `recent_invalid` is empty while the untouched right side's contradiction remains intact.
        field.update("1*");
        assert!(field.recent_invalid().next().is_none());
        assert_ne!(
            field.cells[c4].value.is_invalid(field.cells[c4].neighbors),
            field.cells[c6].value.is_invalid(field.cells[c6].neighbors)
        );
        assert_all_invariants(&field);
    }

    #[test]
    fn test_repartition_after_basic_solve_splits_via_inferred_mine() {
        // 5x2 board where a bridge is inferred as Mine during process_solved,
        // splitting a single 10-cell partition into two independent 3-cell partitions:
        // - (2,0)=Opened(3) has only 3 unknown neighbors: (1,1), (2,1), (3,1) -> all 3 inferred as Mine!
        // - (1,0)=Opened(3) now has 2 inferred mines at (1,1) and (2,1), leaving 1 mine among {(0,0), (0,1)}!
        // - (3,0)=Opened(3) now has 2 inferred mines at (2,1) and (3,1), leaving 1 mine among {(4,0), (4,1)}!
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
        let c20 = field.cell_idx(Point::new(2, 0));
        assert_eq!(field.cells[c20].partition_id, PartitionId::ZERO);
        for x in 1..=3 {
            let idx = field.cell_idx(Point { x, y: 1 });
            assert!(field.cells[idx].value.is_mine());
            assert_eq!(field.cells[idx].partition_id, PartitionId::ZERO);
        }

        let c00 = field.cell_idx(Point::new(0, 0));
        let c40 = field.cell_idx(Point::new(4, 0));
        let left_pid = field.cells[c00].partition_id;
        let right_pid = field.cells[c40].partition_id;
        assert_ne!(left_pid, PartitionId::ZERO);
        assert_ne!(right_pid, PartitionId::ZERO);
        assert_ne!(left_pid, right_pid);
        assert_eq!(partition_cell_count(&field, left_pid), 3);
        assert_eq!(partition_cell_count(&field, right_pid), 3);
        assert_all_invariants(&field);
    }

    #[test]
    fn test_index_pending_updates_masks_and_logs_neighbors() {
        let mut field = MineField::new(3, 1).unwrap();
        let c0 = field.cell_idx(Point::new(0, 0));
        let c1 = field.cell_idx(Point::new(1, 0));
        let c2 = field.cell_idx(Point::new(2, 0));
        field.pending.push(IndexedCellValue {
            idx: c1,
            value: CellValue::for_mine_count(1),
        });

        field.index_pending();

        // pending is drained and the neighbors' masks reflect the new known cell
        assert!(field.pending.is_empty());
        assert_eq!(field.cells[c1].value, CellValue::for_mine_count(1));
        assert_eq!(field.cells[c0].neighbors.opened(), !!E);
        assert_eq!(field.cells[c2].neighbors.opened(), !!W);
        assert_eq!(field.cells[c1].neighbors.unknown(), W | E);

        // only the changed cell is queued for solving
        let queued: Vec<CellIdx> = field.solved_queue.iter().collect();
        assert_eq!(queued, vec![c1]);
    }

    #[test]
    fn test_partition_invariants_hold_after_merge() {
        let mut field = MineField::create(
            "1...1
             .....",
        );
        assert_all_invariants(&field);
        field.update("..2");
        assert_eq!(field.partitions.len(), 1);
        assert_all_invariants(&field);
        // pending/desolved queues are fully drained after a solve pass
        assert!(field.pending.is_empty());
        assert!(field.pending_partition_queue.is_empty());
        assert!(field.desolved_queue.is_empty());
        assert!(field.solved_queue.is_empty());
    }

    #[test]
    fn test_partition_split_leaves_no_stale_partition_ids() {
        // Initially one partition {0..=4} bridged through cell 2.
        let mut field = MineField::create(".1.1.");
        assert_eq!(field.partitions.len(), 1);

        // Reverting cell 1 to Unknown leaves only {2, 3, 4} connected.
        field.update("._");
        assert_eq!(field.partitions.len(), 1);
        let c1 = field.cell_idx(Point::new(1, 0));
        assert_eq!(field.cells[c1].partition_id, PartitionId::ZERO);
        assert_all_invariants(&field);
    }

    #[test]
    fn test_partition_dissolution_leaves_no_stale_partition_ids() {
        let mut field = MineField::create(".1.");
        assert_eq!(field.partitions.len(), 1);

        field.update("._");
        assert!(field.partitions.is_empty());
        for cell in field.cells.iter() {
            assert_eq!(cell.partition_id, PartitionId::ZERO);
        }
        assert_all_invariants(&field);
    }

    #[test]
    fn test_unchanged_update_does_not_rebuild_partitions() {
        let mut field = MineField::create(
            "1...1
             .....",
        );
        let before: Vec<_> = field.cells.iter().map(|c| c.partition_id).collect();

        field.update("1...1");
        let after: Vec<_> = field.cells.iter().map(|c| c.partition_id).collect();
        assert_eq!(before, after);
        assert_all_invariants(&field);
    }

    #[test]
    fn test_solve_queue_with_duplicates() {
        let field = MineField::create(
            "2.13
             2.1.",
        );
        assert!(field.recent_invalid().next().is_some());
        assert_all_invariants(&field);
    }

    #[test]
    fn test_partition_create_connected_bipartite_component() {
        // 3x1 board: [Opened(1), Unknown, Opened(2)]
        let mut field = MineField::new(3, 1).unwrap();
        let c0 = field.cell_idx(Point::new(0, 0));
        let c1 = field.cell_idx(Point::new(1, 0));
        let c2 = field.cell_idx(Point::new(2, 0));
        field.cells[c0].value = CellValue::for_mine_count(1);

        field.cells[c1].value = CellValue::UNKNOWN;
        field.cells[c1].neighbors.set_opened(W | E);
        field.cells[c1].neighbors.set_unknown(DirectionSet::EMPTY);

        field.cells[c2].value = CellValue::for_mine_count(2);

        field.pending_partition_queue.push(&mut field.cells, c0);
        field.partition();
        let id = field.cells[c0].partition_id;
        assert_ne!(id, PartitionId::ZERO);
        assert_eq!(field.cells[c1].partition_id, id);
        assert_eq!(field.cells[c2].partition_id, id);

        let mut members = field.partitions[&id].members().collect::<Vec<_>>();
        members.sort();
        assert_eq!(members, vec![c0, c1, c2]);
    }

    #[test]
    fn test_minefield_clone_and_rebuild_from_scratch() {
        let spec = "111\n1*1\n111";
        let field = MineField::create(spec);

        // Exact clone preserves all internal cell states and partition structures
        let cloned = field.clone();
        assert_eq!(cloned.width_with_border, field.width_with_border);
        assert_eq!(cloned.height_with_border, field.height_with_border);
        assert_eq!(cloned.mine_count, field.mine_count);
        assert_eq!(cloned.next_partition_id, field.next_partition_id);
        assert_eq!(cloned.partitions.len(), field.partitions.len());
        for (c1, c2) in field.cells.iter().zip(cloned.cells.iter()) {
            assert_eq!(c1, c2);
        }

        // rebuild_from_scratch resets inferences and re-solves
        let rebuilt = field.rebuild_from_scratch();
        assert_eq!(rebuilt.width_with_border, field.width_with_border);
        assert_eq!(rebuilt.height_with_border, field.height_with_border);
        assert_eq!(rebuilt.mine_count, field.mine_count);
        for (c1, c2) in field.cells.iter().zip(rebuilt.cells.iter()) {
            assert_eq!(c1.value, c2.value);
        }
    }

    #[test]
    fn test_queue_draining_and_solved_queue_flag_lifecycle() {
        let mut field = MineField::create("1.1\n...\n1.1");
        assert!(field.pending.is_empty());
        assert!(field.desolved_queue.is_empty());
        assert!(field.solved_queue.is_empty());
        assert!(field.pending_partition_queue.is_empty());
        for cell in field.cells.iter() {
            assert!(!cell.flags.is_marked(FlagMask::DESOLVE_FLAG));
            assert!(!cell.flags.is_marked(FlagMask::SOLVED_QUEUE));
            assert!(!cell.flags.is_marked(FlagMask::PENDING_PARTITION));
        }

        // Before `process_solved` runs, `index_pending` queues opened cells on `solved_queue`.
        let c00 = field.cell_idx(Point::new(0, 0));
        field.pending.push(IndexedCellValue {
            idx: c00,
            value: CellValue::UNKNOWN,
        });
        field.index_pending();
        assert!(!field.desolved_queue.is_empty());
        for idx in field.solved_queue.iter() {
            assert!(field.cells[idx].value.is_opened());
            assert!(field.cells[idx].flags.is_marked(FlagMask::SOLVED_QUEUE));
        }

        field.process_desolved();
        field.process_solved();
        assert!(field.solved_queue.is_empty());
        assert!(!field.cells[c00].flags.is_marked(FlagMask::SOLVED_QUEUE));
        field.partition();
        assert!(field.pending_partition_queue.is_empty());
        assert_all_invariants(&field);
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
        assert!(field.next_partition_id <= pid_before.inc());
        assert_all_invariants(&field);
    }

    #[test]
    fn test_non_count_on_solved_queue() {
        let mut field = MineField::create(".11.");
        // Reset both `1` cells to Unknown. Processing the first puts both into `desolved_queue`
        // and cell #2 into `solved_queue`; resetting the second leaves a non-opened cell in `solved_queue`.
        field.update(".__");
        assert!(field.partitions.is_empty());
        for cell in field.cells() {
            assert_eq!(cell.cell.value, UserCellValue::Unknown);
            assert_eq!(cell.inferred, InferredValue::None);
        }
        assert_all_invariants(&field);

        let mut field = MineField::create(".111.\n.....");
        field.update(".__");
        assert_eq!(field.partitions.len(), 1);
        assert_eq!(
            field.cell(Point { x: 1, y: 0 }).value,
            UserCellValue::Unknown
        );
        assert_eq!(
            field.cell(Point { x: 2, y: 0 }).value,
            UserCellValue::Unknown
        );
        assert_eq!(
            field.cell(Point { x: 3, y: 0 }).value,
            UserCellValue::Opened(1)
        );
        assert_all_invariants(&field);
    }

    #[test]
    fn test_adjacent_walls_solve_isolation_and_reversion() {
        let mut field = MineField::new(3, 1).unwrap();
        let c0 = field.cell_idx((0, 0).into());
        let c1 = field.cell_idx((1, 0).into());
        change_cell_value(&mut field, c0, CellValue::WALL);
        change_cell_value(&mut field, c1, CellValue::WALL);
        change_cell_value(&mut field, c0, CellValue::UNKNOWN);

        // Solving an open cell adjacent to a wall must not overwrite the wall
        change_cell_value(&mut field, c0, CellValue::for_mine_count(0));
        field.solve();
        assert_eq!(
            field.cells[c1].value,
            CellValue::WALL,
            "Wall at (1, 0) was overwritten by solver!"
        );

        // Reverting the second wall restores bidirectional connections between playable cells
        change_cell_value(&mut field, c1, CellValue::UNKNOWN);
        assert!(
            !(field.cells[c0].neighbors.unknown() & crate::solver::grid::E).is_empty(),
            "Cell (0, 0) should see (1, 0) as Unknown (E)"
        );
        assert!(
            !(field.cells[c1].neighbors.opened() & crate::solver::grid::W).is_empty(),
            "Cell (1, 0) should see (0, 0) as Opened (W)"
        );
        for cell in field.cells.iter() {
            assert_neighbor_invariants(cell.neighbors);
        }
    }

    #[test]
    fn test_revert_cleared_neighbor_reinfers_from_other_clue() {
        // (0,0)=0 infers (1,0) as Cleared; (2,0)=1 has Mine at (3,0) so remaining_mines == 0.
        let mut field = MineField::create("0.1*");
        let p1 = Point::new(1, 0);
        assert_eq!(field.cell(p1).inferred, InferredValue::Safe);

        // Reverting (0,0) to Unknown runs desolve on (1,0) (Cleared -> Unknown).
        // (2,0)=1 must be re-queued on solved_queue and re-infer (1,0) as Safe!
        field.update("_");
        assert_eq!(field.cell(p1).inferred, InferredValue::Safe);
        for cell in field.cells.iter() {
            assert_neighbor_invariants(cell.neighbors);
        }
    }

    #[test]
    fn test_revert_opened_to_unknown_triggers_neighbor_solve() {
        // Both cells opened: (0,0)=0 and (1,0)=1
        let mut field = MineField::create("01");
        let p1 = Point::new(1, 0);
        assert_eq!(field.cell(p1).value, UserCellValue::Opened(1));

        // Reverting (1,0) to Unknown gives (0,0)=0 an Unknown neighbor, so (0,0) must
        // solve (1,0) as Safe.
        field.update("._");
        assert_eq!(field.cell(p1).value, UserCellValue::Unknown);
        assert_eq!(field.cell(p1).inferred, InferredValue::Safe);
        for cell in field.cells.iter() {
            assert_neighbor_invariants(cell.neighbors);
        }
    }

    #[test]
    fn test_1x1_and_zero_dimension_boards() {
        assert!(matches!(
            MineField::new(5, 0),
            Err(SolverError::AreaTooSmall)
        ));
        assert!(matches!(
            MineField::new(0, 5),
            Err(SolverError::AreaTooSmall)
        ));

        let single_mine = MineField::with_mine_count(1, 1, 1).unwrap();
        assert_eq!(single_mine.mine_count(), Some(1));
        assert_eq!(single_mine.width(), 1);
        assert_eq!(single_mine.height(), 1);

        let f_zero = MineField::create("0");
        assert_eq!(
            f_zero.cell(Point::new(0, 0)).value,
            UserCellValue::Opened(0)
        );
        assert!(f_zero.partitions.is_empty());

        let f_unknown = MineField::create(".");
        assert_eq!(
            f_unknown.cell(Point::new(0, 0)).value,
            UserCellValue::Unknown
        );
        assert_eq!(
            f_unknown.cell(Point::new(0, 0)).inferred,
            InferredValue::None
        );
    }

    #[test]
    fn test_recent_inferred_and_cells_public_iterators() {
        let mut field = MineField::create("1...");
        let recent_1: Vec<_> = field.recent_inferred().collect();
        assert_eq!(recent_1, vec![Point::new(1, 0)]);

        // Confirming the inferred mine directly keeps inferences unchanged and clears recent_inferred
        field.update(".*..");
        assert!(field.recent_inferred().next().is_none());

        // Revealing (2, 0)=1 uses confirmed mine at (1, 0) and newly infers (3, 0) as Safe
        field.update("..1.");
        let recent_3: Vec<_> = field.recent_inferred().collect();
        assert_eq!(recent_3, vec![Point::new(3, 0)]);

        let all_cells: Vec<_> = field.cells().collect();
        assert_eq!(all_cells.len(), 4);
        assert_eq!(all_cells[0].cell.value, UserCellValue::Opened(1));
        assert_eq!(all_cells[1].cell.value, UserCellValue::Mine);
        assert_eq!(all_cells[2].cell.value, UserCellValue::Opened(1));
        assert_eq!(all_cells[3].cell.value, UserCellValue::Unknown);
        assert_eq!(all_cells[3].inferred, InferredValue::Safe);
    }

    #[test]
    fn test_confirming_inferred_cells_is_non_breaking() {
        // 1.1.. -> (1,0)=Flagged, (3,0)=Cleared
        let mut field = MineField::create("1.1..");
        let c1 = field.cell_idx(Point::new(1, 0));
        let c3 = field.cell_idx(Point::new(3, 0));
        assert_eq!(field.cells[c1].value, CellValue::FLAGGED);
        assert_eq!(field.cells[c3].value, CellValue::CLEARED);

        // Confirming Flagged -> Mine and Cleared -> Opened(1) are both NonBreaking.
        // Because (3,0)=1 now has 1 unknown neighbor (4,0) and 0 mines, (4,0) is inferred as Mine.
        change_cell_value(&mut field, c1, CellValue::MINE);
        change_cell_value(&mut field, c3, CellValue::for_mine_count(1));
        field.solve();
        assert_eq!(field.cell(Point::new(4, 0)).inferred, InferredValue::Mine);
    }

    #[test]
    fn test_wall_partition_split_and_rebridge() {
        // Two '1' clues bridged across (2,0)=2 into one 10-cell partition
        let mut field = MineField::create(
            "1...1
             .....",
        );
        field.update("..2");
        assert_eq!(field.partitions.len(), 1);

        // Replacing the bridge (2,0) with an interior wall splits into two 4-cell partitions
        field.update("..@");
        assert_eq!(field.partitions.len(), 2);
        let c00 = field.cell_idx(Point::new(0, 0));
        let c40 = field.cell_idx(Point::new(4, 0));
        assert_ne!(field.cells[c00].partition_id, field.cells[c40].partition_id);

        // Restoring (2,0) as Opened(2) bridges the two partitions back into one
        field.update("..2");
        assert_eq!(field.partitions.len(), 1);
        assert_eq!(field.cells[c00].partition_id, field.cells[c40].partition_id);
        assert_all_invariants(&field);
    }

    #[test]
    fn test_set_cell_updates_board_and_infers() {
        let mut field = MineField::new(3, 1).unwrap();
        field
            .set_cell(UserCell {
                point: Point { x: 0, y: 0 },
                value: UserCellValue::Opened(1),
            })
            .unwrap();
        assert_eq!(
            field.cell(Point { x: 0, y: 0 }).value,
            UserCellValue::Opened(1)
        );
        assert_eq!(
            field.cell(Point { x: 1, y: 0 }).inferred,
            InferredValue::Mine
        );
        assert_all_invariants(&field);
    }

    #[test]
    fn test_set_cells_out_of_bounds_returns_error_without_solving() {
        let mut field = MineField::create("1..\n...");
        let snapshot = field.clone();

        let err = field
            .set_cells([
                UserCell {
                    point: Point { x: 2, y: 0 },
                    value: UserCellValue::Opened(0),
                },
                UserCell {
                    point: Point { x: 3, y: 0 },
                    value: UserCellValue::Opened(0),
                },
            ])
            .unwrap_err();
        assert!(matches!(err, SolverError::CoordinatesOutOfBounds { .. }));
        assert_eq!(field.cells.as_ref(), snapshot.cells.as_ref());
        assert_eq!(field.partitions.len(), snapshot.partitions.len());
    }

    #[test]
    fn test_cell_out_of_bounds_panics() {
        let field = MineField::new(2, 2).unwrap();
        assert!(std::panic::catch_unwind(|| field.cell(Point { x: 2, y: 0 })).is_err());
        assert!(std::panic::catch_unwind(|| field.cell(Point { x: 0, y: 2 })).is_err());
    }

    #[test]
    fn test_minefield_display_formatting() {
        // 6x2 board exercising Opened ('1', '0'), Inferred Mine ('x'), Wall ('@'), Mine ('*'),
        // Inferred Safe ('#'), and Unknown ('.')
        let field = MineField::create("1.@*..\n@@...0");
        let expected = "\
            1x@*##\n\
            @@..#0\n";
        assert_eq!(field.to_string(), expected);
    }

    #[test]
    fn test_singleton_partition_cleanup_resets_nonzero_partition_id() {
        // (0,0)=1 and (1,0)=1 share 2 unknown cells {(0,1), (1,1)} in a 4-cell partition
        let mut field = MineField::create("11.\n...");
        let c10 = field.cell_idx(Point::new(1, 0));
        let pid = field.cells[c10].partition_id;
        assert_ne!(pid, PartitionId::ZERO);

        // Reveal (2,0)=0 -> infers (1,1) Safe. Now (1,0)=1 has only 1 unknown neighbor (0,1),
        // which infers (0,1) Mine, dissolving the entire partition and resetting partition_ids.
        field.update("..0");
        assert!(field.partitions.is_empty());
        assert_eq!(field.cells[c10].partition_id, PartitionId::ZERO);
        assert_all_invariants(&field);
    }

    #[test]
    fn test_excess_mines_marks_recent_invalid_and_recovers() {
        // Opened(1) at (1,0) already touches two Mines -> remaining_mines == -1
        let mut field = MineField::create("*1*");
        let c1 = field.cell_idx(Point::new(1, 0));
        assert_eq!(
            field.cells[c1]
                .value
                .remaining_mines(field.cells[c1].neighbors),
            -1
        );
        assert_eq!(
            field.recent_invalid().collect::<Vec<_>>(),
            vec![Point { x: 1, y: 0 }]
        );

        // Removing one mine brings (1,0) back to valid and satisfies it cleanly
        field.update("_");
        assert!(field.recent_invalid().next().is_none());
        assert_eq!(
            field.cells[c1]
                .value
                .remaining_mines(field.cells[c1].neighbors),
            0
        );
        assert_all_invariants(&field);
    }
}

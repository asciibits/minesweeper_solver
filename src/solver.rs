//! Core Minesweeper solver state and partition orchestration.
use cell::{UserCell, UserCellValue};
use grid::Point;
use std::{
    cmp::max,
    collections::HashMap,
    error::Error,
    fmt::{self, Display},
    ops::{Index, IndexMut},
};

use crate::{
    InferredValue,
    solver::{
        cell::{
            CellFlags, CellIdx, CellNeighbors, CellValue, ChangeStatus, FastNumberHasherBuilder,
            FlagMask, GuardedQueue, IndexedCell, UserCellWithInference, UserValueWithInference,
            cell_idx, coordinates, create_grid_with_border,
        },
        grid::DirectionSet,
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
    /// Width of the board in cells, including the shared border column of `WALL` cells (`width + 1`).
    width_with_border: u32,
    /// Height of the board in cells, including the top and bottom border rows of `WALL` cells (`height + 2`).
    height_with_border: u32,
    /// Precomputed signed index offsets for all 8 compass neighbors (`NW..SE`).
    neighbor_offsets: [i32; 8],
    /// Total number of mines on the board.
    mine_count: Option<u32>,
    /// Row-major contiguous array of all board cell values.
    cells: Box<[CellValue]>,
    /// Row-major contiguous array of partition IDs for each cell. 0 indicates unpartitioned;
    /// non-zero values group cells into connected bipartite frontier components.
    partition_ids: Box<[PartitionId]>,
    /// Row-major contiguous array of bitflags tracking membership in solver queues ([`GuardedQueue`]).
    cell_flags: Box<[CellFlags]>,
    /// Row-major contiguous array of precomputed and dynamically maintained neighbor direction masks for each cell ([`CellNeighbors`]).
    cell_neighbors: Box<[CellNeighbors]>,
    /// Active board partitions, mapping each [`PartitionId`] to its [`Partition`] metadata (seed cell and size).
    partitions: HashMap<PartitionId, Partition, FastNumberHasherBuilder>,
    /// Next unique ID to assign to a newly walked partition.
    next_partition_id: PartitionId,
    /// User-supplied cell updates waiting to be applied by [`MineField::index_pending`].
    pending: Vec<IndexedCell>,
    /// Queue of opened (or formerly opened) cells whose breaking changes require inferences in their connected components to be reset.
    desolved_queue: GuardedQueue<{ FlagMask::DESOLVE_FLAG.mask() }>,
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

        let (cells, cell_neighbors) = create_grid_with_border(width, height);

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
            partition_ids: vec![PartitionId(0); border_area].into_boxed_slice(),
            cell_flags: vec![CellFlags::new(); border_area].into_boxed_slice(),
            cell_neighbors,
            next_partition_id: PartitionId(1),
            pending: Vec::with_capacity(area / 4),
            desolved_queue: GuardedQueue::with_capacity(0),
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
    /// - The inferred board dimensions and `mine_count` fail [`MineField::new`] validation (e.g., an empty
    ///   spec or `mine_count >= width * height`).
    #[track_caller]
    pub fn create(field_spec: &str) -> MineField {
        let (width, height, cell_iter) = Self::cells_and_size_from_spec(field_spec);

        // TODO: implement mine count into the spec
        let mut field = MineField::new(width, height).unwrap();
        for cell in cell_iter {
            field.pending.push(IndexedCell {
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

    /// Returns the [`crate::CellWithInference`] at the given coordinate `p`.
    pub fn cell(&self, p: Point) -> UserValueWithInference {
        assert!(p.x < self.width());
        assert!(p.y < self.height());
        self.cells[self.cell_idx(p)].into()
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
            self.pending.push(IndexedCell {
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
            self.pending.push(IndexedCell {
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
                let v: UserValueWithInference = self.cells[idx].into();
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
            .filter(|&&i| self.cells[i].is_mine())
            .map(|&i| self.coordinates(i))
    }

    #[cfg(test)]
    pub(self) fn recent_inferred_safe(&self) -> impl Iterator<Item = Point> {
        self.recent_inferred
            .iter()
            .filter(|&&i| self.cells[i].is_safe())
            .map(|&i| self.coordinates(i))
    }

    #[cfg(any(test, feature = "debug"))]
    pub(self) fn cell_data_iter(
        &self,
    ) -> impl Iterator<Item = (CellValue, CellFlags, CellNeighbors, PartitionId)> {
        (0..self.cells.len()).map(|i| {
            (
                self.cells[i],
                self.cell_flags[i],
                self.cell_neighbors[i],
                self.partition_ids[i],
            )
        })
    }

    /// Updates a cell's value, synchronizes neighbor direction bitmasks via SIMD, queues affected
    /// cells for desolving/solving, and returns a [`ChangeStatus`] indicating whether existing
    /// neighbor inferences remain valid.
    ///
    /// Delegates directly to [`CellValue::apply_change`].
    fn change_cell_value(&mut self, idx: CellIdx, new_value: CellValue) -> ChangeStatus {
        CellValue::apply_change(self, idx, new_value)
    }

    pub(super) fn compact_partition(&mut self) {
        let mut id_map =
            HashMap::with_capacity_and_hasher(self.partitions.len(), FastNumberHasherBuilder);
        let mut new_partitions =
            HashMap::with_capacity_and_hasher(self.partitions.len(), FastNumberHasherBuilder);
        let mut new_idx = 1;
        for (old_pid, partition) in self.partitions.drain() {
            let new_pid = PartitionId(new_idx);
            id_map.insert(old_pid, new_pid);
            new_partitions.insert(new_pid, partition);
            new_idx += 1;
        }
        self.partitions = new_partitions;
        for pid in self.partition_ids.iter_mut() {
            if pid.0 > 0 {
                *pid = id_map.get(pid).copied().unwrap_or(PartitionId(0));
            }
        }
        self.next_partition_id = PartitionId(new_idx);
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
            let IndexedCell { idx, value } = self.pending[i];
            self.change_cell_value(idx, value);
        }
        self.pending.clear();
    }

    fn process_desolved(&mut self) {
        if !self.desolved_queue.is_empty() {
            while let Some(idx) = self.desolved_queue.pop(&mut self.cell_flags) {
                if self.partition_ids[idx] != self.next_partition_id {
                    self.desolve_partition(idx);
                }
            }
            self.next_partition_id.inc();
        }
    }

    // Kept out-of-line so `hashbrown`'s SIMD probing registers and stack spills do not
    // bloat every frame of the recursive `desolve_partition` and `visit_partition` walkers.
    #[inline(never)]
    fn remove_partition(&mut self, pid: PartitionId) {
        self.partitions.remove(&pid);
    }

    // Step through the cells of a partition. Note: This function assumes the starting cell is
    // opened, or was prior to the recent change
    fn desolve_partition(&mut self, idx: CellIdx) {
        // Takes `&[i32; 8]` separately from `field` so `idx.neighbor_iter` only borrows `offsets`
        // (8-byte pointer) instead of copying `[i32; 8]` onto every recursive stack frame.
        fn visit_opened(field: &mut MineField, offsets: &[i32; 8], idx: CellIdx) {
            let neighbors = field.cell_neighbors[idx];
            #[cfg(feature = "debug")]
            {
                eprintln!("Desolving {idx} (inferred: {})", neighbors.inferred());
            }

            field.solved_queue.push(&mut field.cell_flags, idx);
            let pid = field.partition_ids[idx];
            if pid.0 != 0 {
                field.remove_partition(pid);
            }

            field.metrics.inc_neighbor_visits(neighbors.unknown());
            for n_idx in idx.neighbor_iter(offsets, neighbors.unknown()) {
                field
                    .pending_partition_queue
                    .push(&mut field.cell_flags, n_idx);
            }

            field.partition_ids[idx] = field.next_partition_id;

            // reset all inferred neighbors
            field.metrics.inc_neighbor_visits(neighbors.inferred());
            for n_idx in idx.neighbor_iter(offsets, neighbors.inferred()) {
                if field.partition_ids[n_idx] != field.next_partition_id {
                    visit_inferred(field, offsets, n_idx);
                }
            }
        }

        fn visit_inferred(field: &mut MineField, offsets: &[i32; 8], idx: CellIdx) {
            field.partition_ids[idx] = field.next_partition_id;
            let neighbors = field.cell_neighbors[idx];
            #[cfg(feature = "debug")]
            {
                eprintln!("Resetting inferred {idx} (opened: {})", neighbors.opened());
            }

            field
                .pending_partition_queue
                .push(&mut field.cell_flags, idx);

            // reset inferred to unknown
            field.change_cell_value(idx, CellValue::UNKNOWN);

            // walk all opened neighbors
            field.metrics.inc_neighbor_visits(neighbors.opened());
            for n_idx in idx.neighbor_iter(offsets, neighbors.opened()) {
                if field.partition_ids[n_idx] != field.next_partition_id {
                    visit_opened(field, offsets, n_idx);
                }
            }
        }

        let offsets = self.neighbor_offsets;
        visit_opened(self, &offsets, idx);
    }

    fn process_solved(&mut self) {
        self.recent_inferred.clear();
        while let Some(idx) = self.solved_queue.pop(&mut self.cell_flags) {
            let cell = self.cells[idx];
            self.pending_partition_queue.push(&mut self.cell_flags, idx);
            if !cell.is_opened() {
                continue;
            }
            let neighbors = self.cell_neighbors[idx];
            let unknown_neighbors = neighbors.unknown();
            if !unknown_neighbors.is_empty()
                && let Some(value) = cell.infer_neighbors(neighbors)
            {
                for n_idx in self.neighbor_iter(idx, unknown_neighbors) {
                    let change_type = self.change_cell_value(n_idx, value);
                    if change_type == ChangeStatus::NonBreaking {
                        self.recent_inferred.push(n_idx);
                    }
                }
            }
        }
    }

    fn partition(&mut self) {
        let start_partition_id = self.next_partition_id;
        self.recent_invalid.clear();
        while let Some(idx) = self.pending_partition_queue.pop(&mut self.cell_flags) {
            let cell = self.cells[idx];
            if cell.is_opened() && cell.is_invalid(self.cell_neighbors[idx]) {
                self.recent_invalid.push(idx);
            }
            if self.partition_ids[idx] < start_partition_id {
                let next_partition_id = self.next_partition_id;
                let len = self.visit_partition(
                    idx,
                    &mut |field, idx| {
                        if field.partition_ids[idx] == field.next_partition_id {
                            return false;
                        }
                        if field.partition_ids[idx].0 != 0 {
                            field.remove_partition(field.partition_ids[idx]);
                        }
                        field.partition_ids[idx] = field.next_partition_id;
                        true
                    },
                    &mut |field, idx| {
                        if field.partition_ids[idx] == field.next_partition_id {
                            return false;
                        }
                        field.partition_ids[idx] = field.next_partition_id;
                        true
                    },
                );

                if len <= 1 {
                    self.partition_ids[idx].set(0);
                } else {
                    self.next_partition_id.inc();
                    self.metrics.inc_partitions_created();
                    self.metrics.set_max_partition_size(len);
                    self.partitions
                        .insert(next_partition_id, Partition { member: idx, len });
                    self.metrics.set_max_partitions(self.partitions.len());
                }
            }
        }
    }

    fn neighbor_iter(
        &mut self,
        idx: CellIdx,
        neighbors: DirectionSet,
    ) -> impl Iterator<Item = CellIdx> + use<> {
        self.metrics.inc_neighbor_visits(neighbors);
        idx.neighbor_iter(self.neighbor_offsets, neighbors)
    }

    fn visit_partition(
        &mut self,
        member: CellIdx,
        visit_opened: &mut impl FnMut(&mut MineField, CellIdx) -> bool,
        visit_unknown: &mut impl FnMut(&mut MineField, CellIdx) -> bool,
    ) -> u32 {
        // Pass `offsets: &[i32; 8]` separately from `field` to avoid copying the 32-byte array
        // per recursive frame, and pass `prev_neighbor` as a sentinel `CellIdx` rather than
        // `Option<CellIdx>` so it fits in one register without a discriminant spill.
        fn walk_partition_opened(
            field: &mut MineField,
            offsets: &[i32; 8],
            idx: CellIdx,
            prev_neighbor: CellIdx,
            visit_opened: &mut impl FnMut(&mut MineField, CellIdx) -> bool,
            visit_unknown: &mut impl FnMut(&mut MineField, CellIdx) -> bool,
        ) -> u32 {
            let mut len = 1;
            let neighbors = field.cell_neighbors[idx].unknown();
            field.metrics.inc_neighbor_visits(neighbors);
            for n_idx in idx.neighbor_iter(offsets, neighbors) {
                if prev_neighbor == n_idx {
                    continue;
                }
                if visit_unknown(field, n_idx) {
                    len += walk_partition_unknown(
                        field,
                        offsets,
                        n_idx,
                        idx,
                        visit_opened,
                        visit_unknown,
                    );
                }
            }
            len
        }

        fn walk_partition_unknown(
            field: &mut MineField,
            offsets: &[i32; 8],
            idx: CellIdx,
            prev_neighbor: CellIdx,
            visit_opened: &mut impl FnMut(&mut MineField, CellIdx) -> bool,
            visit_unknown: &mut impl FnMut(&mut MineField, CellIdx) -> bool,
        ) -> u32 {
            let mut len = 1;
            let neighbors = field.cell_neighbors[idx].opened();
            field.metrics.inc_neighbor_visits(neighbors);
            for n_idx in idx.neighbor_iter(offsets, neighbors) {
                if prev_neighbor == n_idx {
                    continue;
                }
                if visit_opened(field, n_idx) {
                    len += walk_partition_opened(
                        field,
                        offsets,
                        n_idx,
                        idx,
                        visit_opened,
                        visit_unknown,
                    );
                }
            }
            len
        }

        let mut len = 0;
        let offsets = self.neighbor_offsets;

        if self.cells[member].is_opened() {
            if visit_opened(self, member) {
                len = walk_partition_opened(
                    self,
                    &offsets,
                    member,
                    CellIdx::NONE,
                    visit_opened,
                    visit_unknown,
                );
            }
        } else if self.cells[member].is_unknown() && visit_unknown(self, member) {
            len = walk_partition_unknown(
                self,
                &offsets,
                member,
                CellIdx::NONE,
                visit_opened,
                visit_unknown,
            )
        };

        len
    }

    /// Returns the slice of linear cell indices belonging to this partition.
    #[cfg(test)]
    fn members(&mut self, member: CellIdx, capacity: u32) -> Vec<CellIdx> {
        use std::collections::HashSet;
        use std::{cell::RefCell, rc::Rc};

        let member_map = Rc::new(RefCell::new(HashSet::with_capacity_and_hasher(
            capacity as usize,
            FastNumberHasherBuilder,
        )));
        self.visit_partition(
            member,
            &mut |_, idx| member_map.borrow_mut().insert(idx),
            &mut |_, idx| member_map.borrow_mut().insert(idx),
        );
        let mut members = Vec::with_capacity(capacity as usize);
        members.extend(Rc::into_inner(member_map).unwrap().into_inner().iter());
        members
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
                let cell = cells[idx];
                if !cell.is_inferred() && cell != CellValue::UNKNOWN {
                    pending.push(IndexedCell { idx, value: cell });
                }
                cells[idx] = CellValue::UNKNOWN;
            }
        }
        let mut cell_neighbors = self.cell_neighbors.clone();
        for neighbors in cell_neighbors.iter_mut() {
            neighbors.reset();
        }

        let mut field = Self {
            width_with_border: self.width_with_border,
            height_with_border: self.height_with_border,
            neighbor_offsets: self.neighbor_offsets,
            mine_count: self.mine_count,
            cells,
            partitions: self.partitions.clone(),
            partition_ids: self.partition_ids.clone(),
            cell_flags: self.cell_flags.clone(),
            cell_neighbors,
            next_partition_id: PartitionId(1),
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
        field.partition_ids.fill(PartitionId(0));
        field.cell_flags.fill(CellFlags::new());
        field.desolved_queue.clear();
        field.solved_queue.clear();
        field.pending_partition_queue.clear();
        field.recent_inferred.clear();
        field.recent_invalid.clear();
        field.solve();
        field
    }
}

#[derive(Debug, Copy, Clone)]
struct Partition {
    #[allow(dead_code)]
    member: CellIdx,
    #[allow(dead_code)]
    len: u32,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PartitionId(u32);

impl PartitionId {
    #[cfg(test)]
    const MAX: PartitionId = PartitionId(u32::MAX);
    const COMPACT_THRESHOLD: PartitionId = PartitionId(u32::MAX >> 1);

    fn inc(&mut self) {
        self.0 += 1;
    }

    fn set(&mut self, id: u32) {
        self.0 = id;
    }
}

impl Index<CellIdx> for [PartitionId] {
    type Output = PartitionId;

    #[inline]
    fn index(&self, index: CellIdx) -> &Self::Output {
        &self[index.0 as usize]
    }
}

impl IndexMut<CellIdx> for [PartitionId] {
    #[inline]
    fn index_mut(&mut self, index: CellIdx) -> &mut Self::Output {
        &mut self[index.0 as usize]
    }
}

impl Display for PartitionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
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

    #[test]
    fn test_cell_change_value_updates_neighbor_masks() {
        // 3x1 strip, change the middle cell and watch both neighbors
        let mut field = MineField::new(3, 1).unwrap();
        let c0 = field.cell_idx(Point::new(0, 0));
        let c1 = field.cell_idx(Point::new(1, 0));
        let c2 = field.cell_idx(Point::new(2, 0));

        field.change_cell_value(c1, CellValue::MINE);
        assert_eq!(field.cells[c1], CellValue::MINE);
        assert_eq!(field.cell_neighbors[c0].mines(), !!E);
        assert!(field.cell_neighbors[c0].opened().is_empty());
        assert!(field.cell_neighbors[c0].inferred().is_empty());
        assert_eq!(field.cell_neighbors[c2].mines(), !!W);
        assert!(field.cell_neighbors[c2].opened().is_empty());
        assert!(field.cell_neighbors[c2].inferred().is_empty());

        // Mine -> Safe(1): neighbors drop the mine but stay known (not unknown)
        field.change_cell_value(c1, CellValue::for_mine_count(1));
        assert!(field.cell_neighbors[c0].mines().is_empty());
        assert!(field.cell_neighbors[c0].inferred().is_empty());
        assert_eq!(field.cell_neighbors[c0].opened(), !!E);

        // Safe(1) -> Unknown: neighbors regain an unknown neighbor
        field.change_cell_value(c1, CellValue::UNKNOWN);
        assert!(field.cell_neighbors[c0].mines().is_empty());
        assert!(field.cell_neighbors[c0].opened().is_empty());
        assert!(field.cell_neighbors[c2].mines().is_empty());
        assert!(field.cell_neighbors[c2].opened().is_empty());
    }

    #[test]
    fn test_streamlined_cell_value_transitions() {
        // Grid 3x1: [0, 1, 2]
        let mut field = MineField::new(3, 1).unwrap();
        let c0 = field.cell_idx(Point::new(0, 0));
        let c1 = field.cell_idx(Point::new(1, 0));
        let c2 = field.cell_idx(Point::new(2, 0));

        // 1. Unknown -> Opened(1) on cell 0
        let status = field.change_cell_value(c0, CellValue::for_mine_count(1));
        assert_eq!(status, ChangeStatus::NonBreaking);
        assert_eq!(field.cells[c0], CellValue::for_mine_count(1));
        // Cell 0's `unknown` mask points to Cell 1 (E), and Cell 1's `opened` mask points to Cell 0 (W)
        assert_eq!(field.cell_neighbors[c0].unknown(), !!E);
        assert_eq!(field.cell_neighbors[c1].opened(), !!W);

        // 2. Unknown -> Cleared on cell 2
        let status = field.change_cell_value(c2, CellValue::CLEARED);
        assert_eq!(status, ChangeStatus::NonBreaking);
        assert_eq!(field.cells[c2], CellValue::CLEARED);
        // Cell 1's `inferred` mask points to Cell 2 (E)
        assert_eq!(field.cell_neighbors[c1].inferred(), !!E);

        // 3. Cleared -> Opened(0) on cell 2
        let status = field.change_cell_value(c2, CellValue::for_mine_count(0));
        assert_eq!(status, ChangeStatus::NonBreaking);
        assert_eq!(field.cells[c2], CellValue::for_mine_count(0));
        // Cell 1 drops Cell 2 from `inferred` and adds Cell 2 to `opened`
        assert!(field.cell_neighbors[c1].inferred().is_empty());
        assert_eq!(field.cell_neighbors[c1].opened(), W | E);

        // 4. Unknown -> Mine on cell 1
        let status = field.change_cell_value(c1, CellValue::MINE);
        assert_eq!(status, ChangeStatus::NonBreaking);
        assert_eq!(field.cells[c1], CellValue::MINE);
        // Both cell 0 and cell 2 drop cell 1 from `unknown` and set cell 1 in `mines`
        assert!(field.cell_neighbors[c0].unknown().is_empty());
        assert_eq!(field.cell_neighbors[c0].mines(), !!E);
        assert!(field.cell_neighbors[c2].unknown().is_empty());
        assert_eq!(field.cell_neighbors[c2].mines(), !!W);

        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
        }
    }

    #[test]
    fn test_breaking_cell_value_transitions() {
        let mut field = MineField::new(3, 1).unwrap();
        let c0 = field.cell_idx(Point::new(0, 0));
        field.change_cell_value(c0, CellValue::for_mine_count(1));

        // Count -> different Count is triggers a desolve for that cell, but is not Breaking for the
        // (neighbors their mine_count and unknown_count are unchanged)
        let status = field.change_cell_value(c0, CellValue::for_mine_count(2));
        assert_eq!(status, ChangeStatus::NonBreaking);

        // Count -> Unknown is Breaking
        let status = field.change_cell_value(c0, CellValue::UNKNOWN);
        assert_eq!(status, ChangeStatus::Breaking);

        // Unknown -> Mine is NonBreaking
        let status = field.change_cell_value(c0, CellValue::MINE);
        assert_eq!(status, ChangeStatus::NonBreaking);

        // Mine -> Unknown is Breaking
        let status = field.change_cell_value(c0, CellValue::UNKNOWN);
        assert_eq!(status, ChangeStatus::Breaking);

        // InferredSafe -> Unknown is Breaking
        field.change_cell_value(c0, CellValue::CLEARED);
        let status = field.change_cell_value(c0, CellValue::UNKNOWN);
        assert_eq!(status, ChangeStatus::Breaking);

        // Mine -> Count is Breaking
        field.change_cell_value(c0, CellValue::MINE);
        let status = field.change_cell_value(c0, CellValue::for_mine_count(1));
        assert_eq!(status, ChangeStatus::Breaking);

        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
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
            field.cells[field.cell_idx(Point { x: 1, y: 0 })],
            CellValue::WALL
        );
        assert!(field.cells[field.cell_idx(Point { x: 0, y: 1 })].is_safe());

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
        let status = field.change_cell_value(c1, CellValue::WALL);
        assert_eq!(status, ChangeStatus::NonBreaking);
        assert_eq!(field.cells[c1], CellValue::WALL);

        // Wall -> Unknown is Breaking
        let status = field.change_cell_value(c1, CellValue::UNKNOWN);
        assert_eq!(status, ChangeStatus::Breaking);
        assert_eq!(field.cells[c1], CellValue::UNKNOWN);

        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
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
            assert!(field.cells[idx].is_safe());
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
        for &neighbors in field.cell_neighbors.iter() {
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

    fn cell(field: &MineField, x: u32, y: u32) -> (CellValue, CellNeighbors) {
        let idx = field.cell_idx(Point { x, y });
        (field.cells[idx], field.cell_neighbors[idx])
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

        // Neighbor (1, 0) is Unknown and should move W from unknown_neighbors to safe_neighbors
        let (_, n10) = cell(&field, 1, 0);
        assert_eq!(n10.opened(), !!W);
        assert!(n10.mines().is_empty());
        assert_neighbor_invariants(n10);

        // Mark (1, 0) as a Mine: (0, 0) moves E from unknown_neighbors to
        // mine_neighbors (remaining_mines == 1)
        field.update(".*");
        let (c00, n00) = cell(&field, 0, 0);
        assert_eq!(n00.mines(), !!E);
        assert_eq!(n00.unknown(), S | SE);
        assert_eq!(c00.remaining_mines(n00), 1);
        assert_neighbor_invariants(n00);

        // Change (1, 0) from Mine to Count(1): (0, 0) moves E from mine_neighbors to safe_neighbors().
        // Because (0, 0) is Count(2) and now has only 2 unknown neighbors left (S and SE),
        // basic_solve infers both S and SE are mines, reducing remaining_mines() to 0!
        field.update(".1");
        let (c00, n00) = cell(&field, 0, 0);
        assert!(n00.unknown().is_empty());
        assert_eq!(n00.mines(), S | SE);
        assert_eq!(c00.remaining_mines(n00), 0);
        assert_neighbor_invariants(n00);

        // Revert (1, 0) back to Unknown (inference-breaking): (0, 0) moves E from safe_neighbors
        // back to unknown_neighbors, and the inferences on S and SE are cleared!
        field.update("._");
        let (c00, n00) = cell(&field, 0, 0);
        assert!(n00.mines().is_empty());
        assert_eq!(n00.unknown(), E | S | SE);
        assert_eq!(c00.remaining_mines(n00), 2);
        assert_neighbor_invariants(n00);
    }

    #[test]
    fn test_partitioning_and_repartitioning_across_updates() {
        // 5x2 board with Count(1) at top-left (0, 0) and top-right (4, 0)
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
        let p_left = field.partition_ids[c00].0;
        let p_right = field.partition_ids[c40].0;
        assert_ne!(p_left, 0);
        assert_ne!(p_right, 0);
        assert_ne!(p_left, p_right);
        assert_eq!(
            field.partition_ids.iter().filter(|p| p.0 == p_left).count(),
            4
        );
        assert_eq!(
            field
                .partition_ids
                .iter()
                .filter(|p| p.0 == p_right)
                .count(),
            4
        );
        assert_eq!(field.partition_ids[c20].0, 0);
        assert!(field.cell_neighbors[c20].opened().is_empty());
        assert!(field.cell_neighbors[c20].inferred().is_empty());

        // Reveal the middle top cell (2, 0) as Count(2): now both components are bridged
        // into a single 10-cell partition!
        field.update("..2");
        assert_eq!(field.partitions.len(), 1);
        let merged_pid = field.partition_ids[c20];
        assert_ne!(merged_pid.0, 0);
        assert_eq!(
            field
                .partition_ids
                .iter()
                .filter(|&&p| p == merged_pid)
                .count(),
            10
        );
        assert_eq!(field.cell_neighbors[c10].opened(), W | E);
        assert_eq!(field.cell_neighbors[c30].opened(), W | E);
        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
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
        let c00 = field.cell_idx(Point::new(0, 0));
        let c10 = field.cell_idx(Point::new(1, 0));
        let c20 = field.cell_idx(Point::new(2, 0));
        let c30 = field.cell_idx(Point::new(3, 0));
        let c40 = field.cell_idx(Point::new(4, 0));
        let c01 = field.cell_idx(Point::new(0, 1));
        let c41 = field.cell_idx(Point::new(4, 1));

        assert_eq!(field.partitions.len(), 1);
        let initial_pid = field.partition_ids[c00];
        assert_ne!(initial_pid.0, 0);
        for y in 0..field.height() {
            for x in 0..field.width() {
                assert_eq!(
                    field.partition_ids[field.cell_idx(Point::new(x, y))],
                    initial_pid
                );
            }
        }

        // Now reveal the bridge cell (2, 0) as Count(0).
        // `basic_solve` infers (1, 1), (2, 1), and (3, 1) as InferredValue::Safe,
        // and the post-`basic_solve` repartition pass splits the board into two 3-cell
        // partitions: {(0,0), (0,1), (1,0)} and {(3,0), (4,0), (4,1)}, while (2,0)
        // and its inferred-safe neighbors get partition_id == 0.
        field.update("..0");

        assert_eq!(field.partitions.len(), 2);
        assert!(field.cell_neighbors[c20].unknown().is_empty());

        let left_pid = field.partition_ids[c00];
        let right_pid = field.partition_ids[c40];
        assert_ne!(left_pid.0, 0);
        assert_ne!(right_pid.0, 0);
        assert_ne!(left_pid, right_pid);
        assert_eq!(field.partition_ids[c10], left_pid);
        assert_eq!(field.partition_ids[c01], left_pid);
        assert_eq!(field.partition_ids[c30], right_pid);
        assert_eq!(field.partition_ids[c41], right_pid);
        assert_eq!(
            field
                .partition_ids
                .iter()
                .filter(|&&p| p == left_pid)
                .count(),
            3
        );
        assert_eq!(
            field
                .partition_ids
                .iter()
                .filter(|&&p| p == right_pid)
                .count(),
            3
        );

        // Verify all cells (including distant cell 0) have synced inferred neighbor sets
        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
        }
    }

    #[test]
    fn test_partition_complete_dissolution() {
        // 3x1 board: [(0,0) = Unknown, (1,0) = Count(1), (2,0) = Unknown]
        // (1,0) has 2 unknown neighbors (choose(2, 1) = 2 > 1), so 1 partition of 3 cells remains.
        let mut field = MineField::create(".1.");
        let c0 = field.cell_idx(Point::new(0, 0));
        let c1 = field.cell_idx(Point::new(1, 0));
        let c2 = field.cell_idx(Point::new(2, 0));

        assert_eq!(field.partitions.len(), 1);
        assert_eq!(field.cell_neighbors[c1].unknown(), W | E);
        assert_eq!(field.cell_neighbors[c0].opened(), !!E);
        assert_eq!(field.cell_neighbors[c2].opened(), !!W);

        // Flag (0, 0) as a Mine: basic_solve infers (2, 0) is Safe, and the post-basic_solve
        // repartition pass dissolves the partition completely.
        field.update("*");

        assert!(field.partitions.is_empty());
        assert!(field.cells[c2].is_safe());
        assert_eq!(field.cells[c1].remaining_mines(field.cell_neighbors[c1]), 0);
        assert!(field.cell_neighbors[c1].unknown().is_empty());
        assert_eq!(field.cell_neighbors[c1].mines(), !!W);

        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
        }
    }

    #[test]
    fn test_batch_adjacent_cells_and_duplicate_dirty_neighbors() {
        // 3x2 board: reveal (1, 0) as Count(2) first
        let mut field = MineField::create(
            ".2.
             ...",
        );
        let c10 = field.cell_idx(Point::new(1, 0));
        assert_eq!(
            field.cells[c10].remaining_mines(field.cell_neighbors[c10]),
            2
        );

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
        let (c10, n10) = cell(&field, 1, 0);
        let (c11, n11) = cell(&field, 1, 1);
        assert_eq!(c10.remaining_mines(n10), 0);
        assert_eq!(c11.remaining_mines(n11), 0);

        assert_eq!(n10.mines(), W | E);
        assert!(n10.unknown().is_empty());

        assert_eq!(n11.mines(), NW | NE);
        assert!(n11.unknown().is_empty());

        assert!(field.partitions.is_empty());
        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
        }
    }

    #[test]
    fn test_repeated_and_reverted_staging_on_same_cell() {
        let mut field = MineField::create(".1.");
        let c1 = field.cell_idx(Point::new(1, 0));
        assert_eq!(field.cells[c1].remaining_mines(field.cell_neighbors[c1]), 1);

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
        assert_eq!(field.cell_neighbors[c1].mines(), !!W);
        // basic_solve now infers (2,0) is safe. Inferred cells are folded into the neighbor
        // masks, so it moves from `unknown_neighbors` to `safe_neighbors`.
        assert_eq!(
            field.cell(Point { x: 2, y: 0 }).inferred,
            InferredValue::Safe
        );
        assert!(field.cell_neighbors[c1].unknown().is_empty());
        assert_eq!(field.cells[c1].remaining_mines(field.cell_neighbors[c1]), 0);
        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
        }
    }

    #[test]
    fn test_basic_solve_mine_and_safe_inferences_and_cascading() {
        // 4x1 strip: [(0,0)=Count(1), (1,0)=Unknown, (2,0)=Count(1), (3,0)=Unknown]
        // 1. (0,0)=Count(1) has only 1 unknown neighbor (1,0) -> infers (1,0) is Mine.
        // 2. That satisfies (2,0)=Count(1), which then infers (3,0) is Safe!
        let field = MineField::create("1.1.");
        let c0 = field.cell_idx(Point::new(0, 0));
        let c1 = field.cell_idx(Point::new(1, 0));
        let c2 = field.cell_idx(Point::new(2, 0));
        let c3 = field.cell_idx(Point::new(3, 0));

        assert!(field.cells[c1].is_mine());
        assert!(field.cells[c3].is_safe());

        // Verify inferred neighbor sets reflect the deductions
        assert_eq!(field.cell_neighbors[c0].mines(), !!E);
        assert!(field.cell_neighbors[c0].unknown().is_empty());
        assert_eq!(field.cell_neighbors[c2].mines(), !!W);
        assert!(field.cell_neighbors[c2].unknown().is_empty());

        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
        }
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

        assert!(field.cells[c1].is_mine() && field.cells[c1].is_inferred());
        assert!(field.cells[c3].is_mine() && field.cells[c3].is_inferred());
        assert!(field.cells[c5].is_safe() && field.cells[c5].is_inferred());

        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
        }
    }

    #[test]
    fn test_basic_solve_detects_impossible_contradictory_clues() {
        // 3x1 strip: [(0,0)=Count(0), (1,0)=Unknown, (2,0)=Count(1)]
        // (0,0)=Count(0) infers (1,0) is Safe, leaving (2,0)=Count(1) with 0 available cells
        // and 1 remaining mine (solution_count == 0).
        let field = MineField::create("0.1");
        let c0 = field.cell_idx(Point::new(0, 0));
        let c2 = field.cell_idx(Point::new(2, 0));

        assert_ne!(
            field.cells[c0].is_invalid(field.cell_neighbors[c0]),
            field.cells[c2].is_invalid(field.cell_neighbors[c2])
        );
        let invalid: Vec<_> = field.recent_invalid().collect();
        assert_eq!(invalid.len(), 1);
        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
        }
    }

    #[test]
    fn test_inference_compatible_update_preserves_inferences() {
        // 4x1 strip: [(0,0)=Count(1), (1,0)=Unknown, (2,0)=Unknown, (3,0)=Unknown]
        // Round 1: Reveal (0,0)=Count(1) -> infers (1,0) is Mine, and updates (2,0)'s
        // inferred_mine_neighbors to include W.
        let mut field = MineField::create("1...");
        let c1 = field.cell_idx(Point::new(1, 0));
        let c2 = field.cell_idx(Point::new(2, 0));
        let c3 = field.cell_idx(Point::new(3, 0));
        assert!(field.cells[c1].is_mine());
        assert_eq!(field.cell_neighbors[c2].mines(), !!W);

        // Round 2 (inference-compatible): Reveal (2,0) as Count(1).
        // Because (1,0)'s Mine inference is preserved, (2,0) immediately knows its 1 mine is at W
        // and infers (3,0) is Safe!
        field.update("..1");
        assert!(field.cells[c1].is_mine());
        assert!(field.cells[c3].is_safe());
        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
        }
    }

    #[test]
    fn test_inference_breaking_updates_clear_and_recompute_inferences() {
        // 3x1 strip: [(0,0), (1,0), (2,0)]
        // Round 1: Reveal (1,0) as Count(2) -> both (0,0) and (2,0) are inferred as Mine.
        let mut field = MineField::create(".2.");
        let c0 = field.cell_idx(Point::new(0, 0));
        let c1 = field.cell_idx(Point::new(1, 0));
        let c2 = field.cell_idx(Point::new(2, 0));
        assert!(field.cells[c0].is_mine());
        assert!(field.cells[c2].is_mine());

        // Round 2 (Inference-breaking Case A): Change (1,0) from Count(2) to Count(1).
        // Neither (0,0) nor (2,0) can be inferred anymore; both must reset to InferredState::None!
        field.update(".1");
        assert!(!field.cells[c0].is_inferred());
        assert!(!field.cells[c2].is_inferred());
        assert_eq!(field.cell_neighbors[c1].unknown(), W | E);
        assert!(field.cell_neighbors[c1].mines().is_empty());
        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
        }

        // Round 3: Change (1,0) to Count(0) -> both (0,0) and (2,0) are inferred as Safe.
        field.update(".0");
        assert!(field.cells[c0].is_safe());
        assert!(field.cells[c2].is_safe());

        // Round 4 (Inference-breaking Case B): Directly violate inference on (0,0) by setting it to Mine,
        // and simultaneously change (1,0) to Count(1).
        // (2,0) must remain/be recomputed as Safe, and all neighbor invariants must hold.
        field.update("*1");
        assert!(!field.cells[c0].is_inferred());
        assert!(field.cells[c2].is_safe());
        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
        }

        // Round 5 (Inference-breaking Case C): Dissolve the partition completely by resetting
        // (0,0) and (1,0) to Unknown. Verify (2,0)'s Safe inference is cleared even though no
        // new partition is created!
        field.update("__");
        assert!(field.partitions.is_empty());
        for (&cell, &neighbors) in field.cells.iter().zip(field.cell_neighbors.iter()) {
            assert!(!cell.is_inferred());
            assert_neighbor_invariants(neighbors);
        }
    }

    #[test]
    fn test_inference_breaking_update_preserves_untouched_partitions() {
        // 5x1 strip: [(0,0)=Count(1), (1,0)=Unknown, (2,0)=Unknown, (3,0)=Unknown, (4,0)=Count(1)]
        // Two independent partitions:
        // - Left partition {(0,0), (1,0)} infers (1,0) is Mine.
        // - Right partition {(3,0), (4,0)} infers (3,0) is Mine.
        let mut field = MineField::create("1...1");
        let c1 = field.cell_idx(Point::new(1, 0));
        let c3 = field.cell_idx(Point::new(3, 0));
        let c4 = field.cell_idx(Point::new(4, 0));
        assert!(field.cells[c1].is_mine());
        assert!(field.cells[c3].is_mine());
        let right_pid = field.partition_ids[c4];

        // Trigger an inference-breaking update on the LEFT partition only:
        // change (0,0) from Count(1) to Count(0).
        // Left cell (1,0) must change from InferredValue::Mine to InferredValue::Safe,
        // while the untouched RIGHT partition (and cell (3,0)'s Mine inference) remains intact!
        field.update("0");
        assert!(field.cells[c1].is_safe());
        assert!(field.cells[c3].is_mine());
        assert_eq!(field.partition_ids[c4], right_pid);
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
        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
        }
    }

    #[test]
    fn test_invalid_resolved_by_editing_other_cell_or_invalid_cell_itself() {
        // 3x1 strip: [(0,0)=Count(0), (1,0)=Unknown, (2,0)=Count(1)]
        // (0,0) is visited first (mine_count = 0), inferring (1,0) is Safe.
        // (2,0) is visited last (mine_count = 1), and index 2 is added to `field.invalid`.
        let mut field = MineField::create("0.1");
        let c1 = field.cell_idx(Point::new(1, 0));
        let invalid: Vec<_> = field.recent_invalid().collect();
        // solver is free to mark the center as either  a mine or not
        if field.cells[c1].is_mine() {
            assert_eq!(invalid, vec![(0, 0).into()]);
        } else {
            assert_eq!(invalid, vec![(2, 0).into()]);
        }

        // Case A: Resolve the inconsistency by editing (0,0) (index 0, NOT index 2!)
        // from Count(0) to Count(1). Index 2 must be removed from `field.invalid`,
        // and (1,0) must now be inferred as Mine.
        field.update("1");
        assert!(field.recent_invalid().next().is_none());
        assert!(field.cells[c1].is_mine());
        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
        }

        // Re-introduce the inconsistency sequentially: set (0,0) back to Count(0).
        // Index 2 (highest mine_count) is again recorded in `field.invalid`.
        field.update("0");
        let invalid: Vec<_> = field.recent_invalid().collect();
        if field.cells[c1].is_mine() {
            assert_eq!(invalid, vec![(0, 0).into()]);
        } else {
            assert_eq!(invalid, vec![(2, 0).into()]);
            assert!(field.cells[c1].is_safe());
        }

        // Case B: Resolve the inconsistency by editing (2,0) (index 2 itself)
        // from Count(1) to Count(0).
        field.update("..0");
        assert!(field.recent_invalid().next().is_none());
        assert!(field.cells[c1].is_safe());
        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
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
        let c1 = field.cell_idx(Point::new(1, 0));
        let c4 = field.cell_idx(Point::new(4, 0));
        let c6 = field.cell_idx(Point::new(6, 0));
        assert!(field.cells[c1].is_mine());
        // one of id: 4 or id: 6 is invalid
        assert_ne!(
            field.cells[c4].is_invalid(field.cell_neighbors[c4]),
            field.cells[c6].is_invalid(field.cell_neighbors[c6])
        );

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
        assert_ne!(
            field.cells[c4].is_invalid(field.cell_neighbors[c4]),
            field.cells[c6].is_invalid(field.cell_neighbors[c6])
        );
        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
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
        let c20 = field.cell_idx(Point::new(2, 0));
        assert_eq!(field.partition_ids[c20].0, 0);
        for x in 1..=3 {
            let idx = field.cell_idx(Point { x, y: 1 });
            assert!(field.cells[idx].is_mine());
            assert_eq!(field.partition_ids[idx].0, 0);
        }

        let c00 = field.cell_idx(Point::new(0, 0));
        let c40 = field.cell_idx(Point::new(4, 0));
        let left_pid = field.partition_ids[c00];
        let right_pid = field.partition_ids[c40];
        assert_ne!(left_pid.0, 0);
        assert_ne!(right_pid.0, 0);
        assert_ne!(left_pid, right_pid);
        assert_eq!(
            field
                .partition_ids
                .iter()
                .filter(|&&p| p == left_pid)
                .count(),
            3
        );
        assert_eq!(
            field
                .partition_ids
                .iter()
                .filter(|&&p| p == right_pid)
                .count(),
            3
        );
        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
        }
    }

    fn assert_partition_invariants(field: &MineField) {
        let mut partition_sizes: HashMap<PartitionId, u32> = HashMap::new();
        for &pid in &field.partition_ids {
            if pid.0 != 0 {
                *partition_sizes.entry(pid).or_insert(0) += 1;
            }
        }
        for (&pid, &Partition { len, .. }) in &field.partitions {
            assert_ne!(pid.0, 0);
            assert_eq!(partition_sizes.get(&pid), Some(&len));
        }
    }

    #[test]
    fn test_index_pending_updates_masks_and_logs_neighbors() {
        let mut field = MineField::new(3, 1).unwrap();
        let c0 = field.cell_idx(Point::new(0, 0));
        let c1 = field.cell_idx(Point::new(1, 0));
        let c2 = field.cell_idx(Point::new(2, 0));
        field.pending.push(IndexedCell {
            idx: c1,
            value: CellValue::for_mine_count(1),
        });

        field.index_pending();

        // pending is drained and the neighbors' masks reflect the new known cell
        assert!(field.pending.is_empty());
        assert_eq!(field.cells[c1], CellValue::for_mine_count(1));
        assert_eq!(field.cell_neighbors[c0].opened(), !!E);
        assert_eq!(field.cell_neighbors[c2].opened(), !!W);
        assert_eq!(field.cell_neighbors[c1].unknown(), W | E);

        // only the changed cell will be queued for repartitioning
        let queued: Vec<CellIdx> = field.solved_queue.iter().collect();
        assert_eq!(queued, vec![c1]);
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
        assert_eq!(field.partitions.len(), 1);
        let c1 = field.cell_idx(Point::new(1, 0));
        assert!(field.partition_ids[c1] < field.next_partition_id);
        assert_partition_invariants(&field);
    }

    #[test]
    fn test_partition_dissolution_leaves_no_stale_partition_ids() {
        let mut field = MineField::create(".1.");
        assert_eq!(field.partitions.len(), 1);

        field.update("._");
        assert!(field.partitions.is_empty());
        for &pid in field.partition_ids.iter() {
            assert!(pid < field.next_partition_id);
        }
    }

    #[test]
    fn test_unchanged_update_does_not_rebuild_partitions() {
        let mut field = MineField::create(
            "1...1
             .....",
        );
        let before = field.partition_ids.to_vec();

        field.update("1...1");
        let after = field.partition_ids.to_vec();
        assert_eq!(before, after);
    }

    #[test]
    fn test_solve_queue_with_duplicates() {
        let field = MineField::create(
            "2.13
             2.1.",
        );
        assert!(field.recent_invalid().next().is_some());
    }

    #[test]
    fn test_partition_create_connected_bipartite_component() {
        // 3x1 board: [Count(1), Unknown, Count(2)]
        let mut field = MineField::new(3, 1).unwrap();
        let c0 = field.cell_idx(Point::new(0, 0));
        let c1 = field.cell_idx(Point::new(1, 0));
        let c2 = field.cell_idx(Point::new(2, 0));
        field.cells[c0] = CellValue::for_mine_count(1);

        field.cells[c1] = CellValue::UNKNOWN;
        field.cell_neighbors[c1].set_opened(W | E);
        field.cell_neighbors[c1].set_unknown(!!E - E);

        field.cells[c2] = CellValue::for_mine_count(2);

        let mut members = field.members(c0, 3);
        members.sort();
        assert_eq!(members, vec![c0, c1, c2]);

        field
            .pending_partition_queue
            .push(&mut field.cell_flags, c0);
        field.partition();
        let id = field.partition_ids[c0];
        assert_ne!(id.0, 0);
        assert_eq!(field.partition_ids[c1], id);
        assert_eq!(field.partition_ids[c2], id);
    }

    #[test]
    fn test_compact_partition_ids() {
        let spec = "1...1
                    .....
                    1...1";
        let mut field = MineField::create(spec);
        assert!(!field.partitions.is_empty());
        let orig_partition_count = field.partitions.len();

        // Artificially elevate next_partition_id to u8::MAX
        field.next_partition_id = PartitionId::MAX;
        // Inject an obsolete cell partition_id that is NOT in field.partitions
        field.partition_ids[9].set(PartitionId::COMPACT_THRESHOLD.0 + 155);

        field.compact_partition();

        assert_eq!(field.partitions.len(), orig_partition_count);
        assert_eq!(field.next_partition_id.0, orig_partition_count as u32 + 1);
        // Dangling partition ID on cell 7 was reset
        assert_eq!(field.partition_ids[9].0, 0);

        // All active partition members have IDs in <= orig_partition_count
        for &pid in field.partitions.keys() {
            assert!(pid.0 <= orig_partition_count as u32);
        }

        // Solving further when next_partition_id >= 2^31 automatically triggers compaction
        field.next_partition_id = PartitionId(PartitionId::COMPACT_THRESHOLD.0 + 911);
        field
            .set_cell(UserCell {
                point: Point { x: 0, y: 1 },
                value: UserCellValue::Opened(2),
            })
            .unwrap();

        assert_eq!(field.next_partition_id.0, orig_partition_count as u32 + 2);
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
            assert_eq!(c1, c2);
        }
    }

    #[test]
    fn test_queue_open_cells_invariants() {
        // Create a board with counts, unknowns, and mines
        let spec = "1.1\n...\n1.1";
        let mut field = MineField::create(spec);
        for idx in field.pending_partition_queue.iter() {
            assert!(
                field.cells[idx].is_opened(),
                "pending_partition_queue must only contain cells with is_open"
            );
        }
        for idx in field.solved_queue.iter() {
            assert!(
                field.cells[idx].is_opened(),
                "solved_queue must only contain cells with is_open"
            );
        }

        // Apply a breaking update changing an open cell into an unknown and another into a mine
        field.update("._*\n...\n1.1");
        for idx in field.pending_partition_queue.iter() {
            assert!(
                field.cells[idx].is_opened(),
                "pending_partition_queue must strictly maintain is_open after updates"
            );
        }
        for idx in field.solved_queue.iter() {
            assert!(
                field.cells[idx].is_opened(),
                "solved_queue must strictly maintain is_open after updates"
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
        assert!(field.next_partition_id.0 <= pid_before.0 + 4);
    }

    #[test]
    fn test_non_count_on_solved_queue() {
        let mut field = MineField::create(".11.");
        // Reset both `1` cells to unknown. Processing the first cell puts both cells in the
        // DESOLVED queue, and cell #3 in the SOLVED queue. Processing the second changes id #3 from
        // a `COUNT` cell to a `UNKNOWN` cell, leaving a non-count cell in the solved queue. Test
        // the following to make sure this passes as expected.
        field.update(".__");

        let mut field = MineField::create(".111.\n.....");
        field.update(".__")
    }

    #[test]
    fn test_adjacent_walls_solve_isolation_and_reversion() {
        let mut field = MineField::new(3, 1).unwrap();
        let c0 = field.cell_idx((0, 0).into());
        let c1 = field.cell_idx((1, 0).into());
        field.change_cell_value(c0, CellValue::WALL);
        field.change_cell_value(c1, CellValue::WALL);
        field.change_cell_value(c0, CellValue::UNKNOWN);

        // Solving an open cell adjacent to a wall must not overwrite the wall
        field.change_cell_value(c0, CellValue::for_mine_count(0));
        field.solve();
        assert_eq!(
            field.cells[c1],
            CellValue::WALL,
            "Wall at (1, 0) was overwritten by solver!"
        );

        // Reverting the second wall restores bidirectional connections between playable cells
        field.change_cell_value(c1, CellValue::UNKNOWN);
        assert!(
            !(field.cell_neighbors[c0].unknown() & crate::solver::grid::E).is_empty(),
            "Cell (0, 0) should see (1, 0) as Unknown (E)"
        );
        assert!(
            !(field.cell_neighbors[c1].opened() & crate::solver::grid::W).is_empty(),
            "Cell (1, 0) should see (0, 0) as Opened (W)"
        );
        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
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
        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
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
        for &neighbors in field.cell_neighbors.iter() {
            assert_neighbor_invariants(neighbors);
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
        assert_eq!(field.cells[c1], CellValue::FLAGGED);
        assert_eq!(field.cells[c3], CellValue::CLEARED);

        // Confirming Flagged -> Mine and Cleared -> Opened(1) are both NonBreaking.
        // Because (3,0)=1 now has 1 unknown neighbor (4,0) and 0 mines, (4,0) is inferred as Mine.
        assert_eq!(
            field.change_cell_value(c1, CellValue::MINE),
            ChangeStatus::NonBreaking
        );
        assert_eq!(
            field.change_cell_value(c3, CellValue::for_mine_count(1)),
            ChangeStatus::NonBreaking
        );
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
        assert_ne!(field.partition_ids[c00], field.partition_ids[c40]);

        // Restoring (2,0) as Opened(2) bridges the two partitions back into one
        field.update("..2");
        assert_eq!(field.partitions.len(), 1);
        assert_eq!(field.partition_ids[c00], field.partition_ids[c40]);
    }
}

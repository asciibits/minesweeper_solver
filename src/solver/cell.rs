//! Cell data structures representing both the external, user-facing board state
//! ([`UserCell`], [`UserCellValue`], [`UserCellWithInference`]) and internal cell representations.

use std::{
    cmp::Ordering,
    collections::VecDeque,
    fmt::{self, Display, Formatter},
    hash::{BuildHasher, Hasher},
    ops::{Index, IndexMut},
};

use crate::solver::{
    Metrics,
    grid::{DirectionSet, DirectionSetIterator, E, N, NE, NW, Neighbor, Point, S, SE, SW, W},
    stats,
};

/// Provide the x/y coordinates of a Cell with the given cell_id in a fixed
/// width field. This is the corollary to `cell_id`.
pub(super) fn coordinates(cell_idx: CellIdx, width: u32) -> Point {
    Point::from_index(cell_idx, width)
}

/// Get a unique ID for the cell in a given-width field. Useful for indexing in
/// a Vector or Array. This is the corollary to `coordinates`.
pub(super) fn cell_idx(p: Point, width: u32) -> CellIdx {
    p.to_index(width)
}

/// A single cell on the Minesweeper board, pairing its [`Point`] coordinates with its [`UserCellValue`].
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct UserCell {
    /// Grid coordinates of the cell.
    pub point: Point,
    /// Current state/value of the cell.
    pub value: UserCellValue,
}

impl UserCell {
    /// Creates a new [`UserCell`] at `point` initialized to [`UserCellValue::Unknown`].
    pub fn new(point: Point) -> UserCell {
        UserCell {
            point,
            value: UserCellValue::Unknown,
        }
    }
}

impl Display for UserCell {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.point, self.value)
    }
}

/// The state of a single cell on the Minesweeper board.
#[derive(Debug, Default, Hash, Clone, Copy, PartialEq, Eq)]
pub enum UserCellValue {
    /// The cell has not been revealed or flagged.
    #[default]
    Unknown,
    /// The cell is an interior wall
    Wall,
    /// The cell contains a mine.
    Mine,
    /// The cell is revealed and has the given number of adjacent mines (`0..=8`).
    Safe(u8),
}

impl Display for UserCellValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            UserCellValue::Unknown => write!(f, "Unknown"),
            UserCellValue::Wall => write!(f, "Wall"),
            UserCellValue::Mine => write!(f, "MINE"),
            UserCellValue::Safe(c) => write!(f, "{c}"),
        }
    }
}

/// What the solver has deduced about an [`UserCellValue::Unknown`] cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferredValue {
    /// Nothing has been deduced.
    None,
    /// The cell is certainly not a mine.
    Safe,
    /// The cell is certainly a mine.
    Mine,
}

/// A user-visible cell value paired with whatever the solver has inferred about it.
///
/// `inferred` is only ever non-[`InferredValue::None`] when `value` is
/// [`UserCellValue::Unknown`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserValueWithInference {
    /// The value supplied by the user.
    pub value: UserCellValue,
    /// The solver's deduction for this cell.
    pub inferred: InferredValue,
}

/// A user-visible cell paired with whatever the solver has inferred about it.
///
/// `inferred` is only ever non-[`InferredValue::None`] when `value` is
/// [`UserCellValue::Unknown`].
pub struct UserCellWithInference {
    /// The value supplied by the user.
    pub cell: UserCell,
    /// The solver's deduction for this cell.
    pub inferred: InferredValue,
}

/// Internal solver representation of a board cell, caching neighbor masks and partition membership.
///
/// A cell does not store its own index; callers identify a cell by its row-major position in the
/// `cells` slice (see [`coordinates`] / [`cell_idx`]).
// Note: Kept at 8 bytes so 8 cells fit in a single 64-byte cache line (enforced below).
#[derive(Debug, Clone, Copy)]
pub(super) struct Cell {
    /// Inline the contents of Cell
    value: CellValue,

    /// All neighbors to this cell. For inner cells this is DirectionSet::ALL, for border cells it
    /// is the set of valid neighbors.
    neighbors: DirectionSet,

    /// For [`CellType::Unknown`] cells, this points to all [`CellType::Count`] neighbors.
    /// Conversely, for [`CellType::Count`] and [`CellType::InferredSafe`] cells, this points to
    /// neighbors of type [`CellType::Unknown`]. Unspecified for mine type cells.
    alt_neighbors: DirectionSet,

    /// All [`CellType::Mine`] and [`CellType::InferredMine`] neighbors. Maintained for all
    /// non-mine cells; unspecified for mine cells ([`CellType::Mine`] and [`CellType::InferredMine`]).
    mine_neighbors: DirectionSet,

    /// For [`CellType::Unknown`] cells, this points to all [`CellType::InferredSafe`] neighbors.
    /// For all other cell types, this is unspecified.
    safe_neighbors: DirectionSet,

    /// A unique id for all members of a partition. Zero values are always un-partitioned; non-zero
    /// values may or may not be partitioned.
    partition_id: u8,

    /// A set of bit flags used when executing a solve pass.
    flags: CellFlags,
}

const _: () = assert!(std::mem::size_of::<Cell>() <= 8);

impl Cell {
    pub(super) fn grid(width: u32, height: u32) -> Box<[Cell]> {
        let width = width as usize;
        let height = height as usize;
        let len = width * height;
        let inner_cell = Cell {
            value: CellValue::UNKNOWN,
            neighbors: DirectionSet::ALL,
            alt_neighbors: DirectionSet::EMPTY,
            mine_neighbors: DirectionSet::EMPTY,
            safe_neighbors: DirectionSet::EMPTY,
            partition_id: 0,
            flags: CellFlags::new(),
        };
        let mut grid: Vec<Cell> = vec![inner_cell; len];
        // fix up the edge neighbors
        for cell in grid.iter_mut().take(width) {
            // top row
            cell.neighbors -= DirectionSet::N_DIRECTIONS;
        }
        for cell in grid.iter_mut().skip(len - width) {
            // bottom row
            cell.neighbors -= DirectionSet::S_DIRECTIONS;
        }
        for cell in grid.iter_mut().step_by(width) {
            // left col
            cell.neighbors -= DirectionSet::W_DIRECTIONS;
        }
        for cell in grid.iter_mut().skip(width - 1).step_by(width) {
            // right col
            cell.neighbors -= DirectionSet::E_DIRECTIONS;
        }

        grid.into_boxed_slice()
    }

    /// Returns the cell's current [`CellValue`].
    #[inline]
    pub(super) fn value(&self) -> CellValue {
        self.value
    }

    /// Sets the cell's [`CellValue`].
    #[inline]
    pub(super) fn set_value(&mut self, value: CellValue) {
        self.value = value;
    }

    /// Returns the ID of the partition this cell belongs to (`0` if unpartitioned).
    #[inline]
    pub(super) fn partition_id(&self) -> u8 {
        self.partition_id
    }

    /// Sets the ID of the partition this cell belongs to (`0` if unpartitioned).
    #[inline]
    pub(super) fn set_partition_id(&mut self, partition_id: u8) {
        self.partition_id = partition_id;
    }

    /// Returns all valid board neighbors for this cell.
    #[inline]
    pub(super) fn neighbors(&self) -> DirectionSet {
        self.neighbors
    }

    /// Returns  alternate neighbors:
    /// - For [`CellType::Unknown`] cells: points to adjacent [`CellType::Count`] cells.
    /// - For [`CellType::Count`] and [`CellType::InferredSafe`] cells: points to adjacent [`CellType::Unknown`] cells.
    ///
    /// Not valid for mine cells.
    #[inline]
    pub(super) fn alt_neighbors(&self) -> DirectionSet {
        // not valid for mine cells
        debug_assert!(!self.value.is_mine());
        self.alt_neighbors
    }

    /// Returns neighbors known or inferred to be mines.
    ///
    /// Valid for all non-mine cells; unspecified for mine cells.
    #[inline]
    pub(super) fn mine_neighbors(&self) -> DirectionSet {
        // not valid for mine cells
        debug_assert!(!self.value.is_mine());
        self.mine_neighbors
    }

    /// Returns inferred safe neighbors.
    ///
    /// Only valid for [`CellType::Unknown`] cells.
    #[inline]
    pub(super) fn safe_neighbors(&self) -> DirectionSet {
        // only valid for unknown cells
        debug_assert!(self.value.is_unknown());
        self.safe_neighbors
    }

    /// Returns a mutable reference to the cell's alternate neighbor set.
    #[inline]
    pub(super) fn alt_neighbors_mut(&mut self) -> &mut DirectionSet {
        // not valid for mine cells
        debug_assert!(!self.value.is_mine());
        &mut self.alt_neighbors
    }

    /// Returns a mutable reference to the cell's mine neighbor set.
    #[inline]
    pub(super) fn mine_neighbors_mut(&mut self) -> &mut DirectionSet {
        // not valid for mine cells
        debug_assert!(!self.value.is_mine());
        &mut self.mine_neighbors
    }

    /// Returns a mutable reference to the cell's safe neighbor set.
    #[inline]
    pub(super) fn safe_neighbors_mut(&mut self) -> &mut DirectionSet {
        // only valid for unknown cells
        debug_assert!(self.value.is_unknown());
        &mut self.safe_neighbors
    }

    /// Returns mutable reference to the cell's status flags.
    #[inline]
    pub(super) fn flags(&mut self) -> &mut CellFlags {
        &mut self.flags
    }

    pub(super) fn remaining_mines(&self) -> i8 {
        // only valid for count cells
        debug_assert!(self.value.has_count());
        self.value.mine_count() as i8 - self.mine_neighbors().len() as i8
    }

    /// Returns `true` if this cell's count cannot be satisfied by its neighbors, i.e. it needs
    /// more mines than it has unknown neighbors left to hold, or it has more
    /// mines than remaining spaces
    pub(super) fn is_invalid(&self) -> bool {
        // only valid for count cells
        debug_assert!(self.value.has_count());
        let mine_neighbors = self.mine_neighbors().len() as i8;
        let unknown_neighbors = self.alt_neighbors().len() as i8;
        self.value().safe_mine_count().is_some_and(|mine_count| {
            let mines_remaining = mine_count as i8 - mine_neighbors;
            mines_remaining < 0 || mines_remaining > unknown_neighbors
        })
    }

    // A fully solved cell is one with no neighbors. Note, this is distinct from a "solvable" cell,
    // which has all of its mines accounted for, but still has unknown neighbors to be set.
    pub(super) fn is_solved(&self) -> bool {
        // only valid for count cells
        debug_assert!(self.value.has_count());
        self.alt_neighbors().is_empty()
    }

    // If the cell is a count cell that is not yet solved, and the neighbors can be inferred as
    // either a Mine or Safe...
    pub(super) fn infer_neighbors(&self) -> Option<CellValue> {
        // only valid for count cells
        debug_assert!(self.value.has_count());
        let unknown_count = self.alt_neighbors().len() as i8;

        self.value().safe_mine_count().and_then(|mine_count| {
            let remaining = mine_count as i8 - self.mine_neighbors().len() as i8;
            if remaining <= 0 {
                Some(CellValue::INFERRED_SAFE)
            } else if remaining >= unknown_count {
                Some(CellValue::INFERRED_MINE)
            } else {
                None
            }
        })
    }

    pub(super) fn apply_change(
        cells: &mut [Cell],
        solved_queue: &mut GuardedQueue,
        desolved_queue: &mut GuardedQueue,
        metrics: &mut Metrics,
        idx: CellIdx,
        new_value: CellValue,
        width: u32,
    ) -> ChangeStatus {
        let prev_value = cells[idx].value();

        let change_function = match prev_value.cell_type() {
            CellType::Unknown => match new_value.cell_type() {
                CellType::Unknown => Self::no_change,
                CellType::InferredMine | CellType::Mine => Self::unknown_to_mine,
                CellType::InferredSafe => Self::unknown_to_inferred_safe,
                CellType::Count => Self::unknown_to_count,
                CellType::Wall => Self::inference_breaking_change,
            },
            CellType::InferredMine | CellType::Mine => match new_value.cell_type() {
                CellType::InferredMine | CellType::Mine => Self::to_mine,
                _ => Self::inference_breaking_change,
            },
            CellType::InferredSafe => match new_value.cell_type() {
                CellType::InferredSafe => Self::no_change,
                CellType::Count => Self::inferred_safe_to_count,
                _ => Self::inference_breaking_change,
            },
            CellType::Count => match new_value.cell_type() {
                CellType::Count => {
                    if prev_value == new_value {
                        Self::no_change
                    } else {
                        // count -> count (diff counts) - it could be de-solved, or solved
                        Self::count_to_count
                    }
                }
                _ => Self::inference_breaking_change,
            },
            CellType::Wall => Self::inference_breaking_change,
        };

        change_function(
            cells,
            solved_queue,
            desolved_queue,
            metrics,
            idx,
            new_value,
            width,
        )
    }

    fn no_change(
        _cells: &mut [Cell],
        _solved_queue: &mut GuardedQueue,
        _desolved_queue: &mut GuardedQueue,
        _metrics: &mut Metrics,
        _idx: CellIdx,
        _new_value: CellValue,
        _width: u32,
    ) -> ChangeStatus {
        ChangeStatus::NoChange
    }

    fn to_mine(
        cells: &mut [Cell],
        _solved_queue: &mut GuardedQueue,
        _desolved_queue: &mut GuardedQueue,
        _metrics: &mut Metrics,
        idx: CellIdx,
        new_value: CellValue,
        _width: u32,
    ) -> ChangeStatus {
        cells[idx].set_value(new_value);
        ChangeStatus::NoChange
    }

    fn count_to_count(
        cells: &mut [Cell],
        solved_queue: &mut GuardedQueue,
        desolved_queue: &mut GuardedQueue,
        _metrics: &mut Metrics,
        idx: CellIdx,
        new_value: CellValue,
        _width: u32,
    ) -> ChangeStatus {
        cells[idx].set_value(new_value);
        solved_queue.push(cells, idx);
        desolved_queue.push(cells, idx);
        ChangeStatus::Breaking
    }

    fn unknown_to_mine(
        cells: &mut [Cell],
        solved_queue: &mut GuardedQueue,
        _desolved_queue: &mut GuardedQueue,
        metrics: &mut Metrics,
        idx: CellIdx,
        new_value: CellValue,
        width: u32,
    ) -> ChangeStatus {
        // alt (count):  +mine, -alt, +solve
        // safe: +mine, -alt
        // mine: n/a
        // unknown: +mine
        // self: n/a
        let alt_neighbors = cells[idx].alt_neighbors();
        metrics.inc_neighbor_visits(alt_neighbors);
        for neighbor in Self::neighbor_iter(idx, width, alt_neighbors) {
            let cell = &mut cells[neighbor.idx];
            *cell.mine_neighbors_mut() |= neighbor.from;
            *cell.alt_neighbors_mut() -= neighbor.from;
            solved_queue.push(cells, neighbor.idx);
        }
        let safe_neighbors = cells[idx].safe_neighbors();
        metrics.inc_neighbor_visits(safe_neighbors);
        for neighbor in Self::neighbor_iter(idx, width, safe_neighbors) {
            let cell = &mut cells[neighbor.idx];
            *cell.mine_neighbors_mut() |= neighbor.from;
            *cell.alt_neighbors_mut() -= neighbor.from;
        }
        let cell = cells[idx];
        let unknown_neighbors = cell.neighbors()
            - (cell.mine_neighbors() | cell.alt_neighbors() | cell.safe_neighbors());
        metrics.inc_neighbor_visits(unknown_neighbors);
        for neighbor in Self::neighbor_iter(idx, width, unknown_neighbors) {
            let cell = &mut cells[neighbor.idx];
            *cell.mine_neighbors_mut() |= neighbor.from;
        }
        cells[idx].set_value(new_value);
        ChangeStatus::NonBreaking
    }

    fn unknown_to_inferred_safe(
        cells: &mut [Cell],
        solved_queue: &mut GuardedQueue,
        _desolved_queue: &mut GuardedQueue,
        metrics: &mut Metrics,
        idx: CellIdx,
        new_value: CellValue,
        width: u32,
    ) -> ChangeStatus {
        // alt (count):  -alt, +solve
        // safe: -alt
        // mine: n/a
        // unknown: +safe
        // self: alt = unknown
        let alt_neighbors = cells[idx].alt_neighbors();
        metrics.inc_neighbor_visits(alt_neighbors);
        for neighbor in Self::neighbor_iter(idx, width, alt_neighbors) {
            let cell = &mut cells[neighbor.idx];
            *cell.alt_neighbors_mut() -= neighbor.from;
            solved_queue.push(cells, neighbor.idx);
        }
        let safe_neighbors = cells[idx].safe_neighbors();
        metrics.inc_neighbor_visits(safe_neighbors);
        for neighbor in Self::neighbor_iter(idx, width, safe_neighbors) {
            let cell = &mut cells[neighbor.idx];
            *cell.alt_neighbors_mut() -= neighbor.from;
        }
        let cell = cells[idx];
        let unknown_neighbors = cell.neighbors()
            - (cell.mine_neighbors() | cell.alt_neighbors() | cell.safe_neighbors());
        metrics.inc_neighbor_visits(unknown_neighbors);
        for neighbor in Self::neighbor_iter(idx, width, unknown_neighbors) {
            let cell = &mut cells[neighbor.idx];
            *cell.safe_neighbors_mut() |= neighbor.from;
        }
        *cells[idx].alt_neighbors_mut() = unknown_neighbors;
        cells[idx].set_value(new_value);

        ChangeStatus::NonBreaking
    }

    fn unknown_to_count(
        cells: &mut [Cell],
        solved_queue: &mut GuardedQueue,
        _desolved_queue: &mut GuardedQueue,
        metrics: &mut Metrics,
        idx: CellIdx,
        new_value: CellValue,
        width: u32,
    ) -> ChangeStatus {
        // alt (count):  -alt, +solve
        // safe: -alt
        // mine: n/a
        // unknown: +alt
        // self: alt = unknown, +solve
        let alt_neighbors = cells[idx].alt_neighbors();
        metrics.inc_neighbor_visits(alt_neighbors);
        for neighbor in Self::neighbor_iter(idx, width, alt_neighbors) {
            let cell = &mut cells[neighbor.idx];
            *cell.alt_neighbors_mut() -= neighbor.from;
            solved_queue.push(cells, neighbor.idx);
        }
        let safe_neighbors = cells[idx].safe_neighbors();
        metrics.inc_neighbor_visits(safe_neighbors);
        for neighbor in Self::neighbor_iter(idx, width, safe_neighbors) {
            let cell = &mut cells[neighbor.idx];
            *cell.alt_neighbors_mut() -= neighbor.from;
        }
        let cell = &cells[idx];
        let unknown_neighbors = cell.neighbors()
            - (cell.mine_neighbors() | cell.alt_neighbors() | cell.safe_neighbors());
        metrics.inc_neighbor_visits(unknown_neighbors);
        for neighbor in Self::neighbor_iter(idx, width, unknown_neighbors) {
            let cell = &mut cells[neighbor.idx];
            *cell.alt_neighbors_mut() |= neighbor.from;
        }
        *cells[idx].alt_neighbors_mut() = unknown_neighbors;
        cells[idx].set_value(new_value);
        solved_queue.push(cells, idx);

        ChangeStatus::NonBreaking
    }

    fn inferred_safe_to_count(
        cells: &mut [Cell],
        solved_queue: &mut GuardedQueue,
        _desolved_queue: &mut GuardedQueue,
        metrics: &mut Metrics,
        idx: CellIdx,
        new_value: CellValue,
        width: u32,
    ) -> ChangeStatus {
        // alt (unknown):  -safe, +alt
        // safe: n/a
        // mine: n/a
        // count: n/a
        // self: +solve
        let alt_neighbors = cells[idx].alt_neighbors();
        metrics.inc_neighbor_visits(alt_neighbors);
        for neighbor in Self::neighbor_iter(idx, width, alt_neighbors) {
            let cell = &mut cells[neighbor.idx];
            *cell.safe_neighbors_mut() -= neighbor.from;
            *cell.alt_neighbors_mut() |= neighbor.from;
        }
        cells[idx].set_value(new_value);
        solved_queue.push(cells, idx);

        ChangeStatus::NonBreaking
    }

    // The slower, not-unrolled general purpose function for the rest of the values. Always involves
    // a loop over all neighbors with a decent amount of branching
    fn inference_breaking_change(
        cells: &mut [Cell],
        solved_queue: &mut GuardedQueue,
        desolved_queue: &mut GuardedQueue,
        metrics: &mut Metrics,
        idx: CellIdx,
        new_value: CellValue,
        width: u32,
    ) -> ChangeStatus {
        let mut mines = DirectionSet::new();
        let mut count = DirectionSet::new();
        let mut safe = DirectionSet::new();
        let mut unknown = DirectionSet::new();
        let prev_value = cells[idx].value();
        let neighbors = cells[idx].neighbors();
        metrics.inc_neighbor_visits(neighbors);
        for neighbor in Self::neighbor_iter(idx, width, neighbors) {
            let dir = neighbor.from.reverse();
            let n_value = cells[neighbor.idx].value();
            if n_value.is_mine() {
                mines |= dir;
            } else if n_value.has_count() {
                count |= dir;
                desolved_queue.push(cells, neighbor.idx);
                solved_queue.push(cells, neighbor.idx);
            } else if n_value.is_safe() {
                safe |= dir;
            } else if n_value.is_unknown() {
                unknown |= dir;
            } else {
                debug_assert!(n_value.is_wall());
            }
            match prev_value.cell_type() {
                CellType::Unknown => {
                    if n_value.is_safe() {
                        *cells[neighbor.idx].alt_neighbors_mut() -= neighbor.from;
                    }
                }
                CellType::Mine | CellType::InferredMine => {
                    if !n_value.is_mine() {
                        *cells[neighbor.idx].mine_neighbors_mut() -= neighbor.from;
                    }
                }
                CellType::Count => {
                    if n_value.is_unknown() {
                        *cells[neighbor.idx].alt_neighbors_mut() -= neighbor.from;
                    }
                    desolved_queue.push(cells, neighbor.idx);
                }
                CellType::InferredSafe => {
                    if n_value.is_unknown() {
                        *cells[neighbor.idx].safe_neighbors_mut() -= neighbor.from;
                    }
                }
                CellType::Wall => {
                    // nothing to subtract for WALL
                }
            }
            match new_value.cell_type() {
                CellType::Unknown => {
                    if n_value.is_safe() {
                        *cells[neighbor.idx].alt_neighbors_mut() |= neighbor.from;
                    }
                }
                CellType::Mine | CellType::InferredMine => {
                    if !n_value.is_mine() {
                        *cells[neighbor.idx].mine_neighbors_mut() |= neighbor.from;
                    }
                }
                CellType::Count => {
                    if n_value.is_unknown() {
                        *cells[neighbor.idx].alt_neighbors_mut() |= neighbor.from;
                    }
                }
                CellType::InferredSafe => {
                    if n_value.is_unknown() {
                        *cells[neighbor.idx].safe_neighbors_mut() |= neighbor.from;
                    }
                }
                CellType::Wall => {
                    // nothing to add for WALL
                }
            }
        }
        let cell = &mut cells[idx];
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
            desolved_queue.push(cells, idx);
        }
        if new_value.has_count() {
            solved_queue.push(cells, idx);
        }

        ChangeStatus::Breaking
    }

    pub(super) fn neighbor_iter(
        idx: CellIdx,
        width: u32,
        directions: DirectionSet,
    ) -> impl Iterator<Item = Neighbor> {
        DirectionSetIterator::new(directions).map(move |d| match d.index() {
            0 => Neighbor {
                idx: CellIdx(idx.0 - width - 1),
                from: SE,
            },
            1 => Neighbor {
                idx: CellIdx(idx.0 - width),
                from: S,
            },
            2 => Neighbor {
                idx: CellIdx(idx.0 - width + 1),
                from: SW,
            },
            3 => Neighbor {
                idx: CellIdx(idx.0 - 1),
                from: E,
            },
            4 => Neighbor {
                idx: CellIdx(idx.0 + 1),
                from: W,
            },
            5 => Neighbor {
                idx: CellIdx(idx.0 + width - 1),
                from: NE,
            },
            6 => Neighbor {
                idx: CellIdx(idx.0 + width),
                from: N,
            },
            7 => Neighbor {
                idx: CellIdx(idx.0 + width + 1),
                from: NW,
            },
            _ => unreachable!(),
        })
    }

    #[cfg(test)]
    pub(super) fn reset(&mut self) {
        self.mine_neighbors = DirectionSet::EMPTY;
        self.safe_neighbors = DirectionSet::EMPTY;
        self.alt_neighbors = DirectionSet::EMPTY;
        self.value = CellValue::UNKNOWN;
    }
}

impl PartialEq for Cell {
    fn eq(&self, other: &Self) -> bool {
        if self.value != other.value
            || self.neighbors != other.neighbors
            || self.flags != other.flags
        {
            return false;
        }

        if !self.value.is_mine() {
            if self.alt_neighbors != other.alt_neighbors
                || self.mine_neighbors != other.mine_neighbors
            {
                return false;
            }
            if self.value.is_unknown() && self.safe_neighbors != other.safe_neighbors {
                return false;
            }
        }

        true
    }
}

impl Display for Cell {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let spacer = if f.alternate() { "\n    " } else { " " };
        write!(
            f,
            "Cell{{{spacer}value: {},{spacer}neighbors: {},",
            self.value, self.neighbors,
        )?;
        match self.value.cell_type() {
            CellType::Unknown => {
                write!(
                    f,
                    "{spacer}count_neighbors: {},{spacer}mine_neighbors: {},{spacer}safe_neighbors: {},",
                    self.alt_neighbors(),
                    self.mine_neighbors(),
                    self.safe_neighbors()
                )?;
            }
            CellType::InferredSafe | CellType::Count => {
                write!(
                    f,
                    "{spacer}alt_neighbors (unknown): {},{spacer}mine_neighbors: {},",
                    self.alt_neighbors(),
                    self.mine_neighbors(),
                )?;
            }
            _ => (),
        };
        write!(
            f,
            "{spacer}partition_id: {},{spacer}flags: {}",
            self.partition_id, self.flags
        )?;
        if f.alternate() {
            writeln!(f)?;
        }
        write!(f, " }}")
    }
}

impl Index<CellIdx> for [Cell] {
    type Output = Cell;

    #[inline]
    fn index(&self, index: CellIdx) -> &Self::Output {
        &self[index.0 as usize]
    }
}

impl IndexMut<CellIdx> for [Cell] {
    #[inline]
    fn index_mut(&mut self, index: CellIdx) -> &mut Self::Output {
        &mut self[index.0 as usize]
    }
}

#[derive(Debug, Copy, Hash, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct CellIdx(pub(super) u32);

impl From<u32> for CellIdx {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<CellIdx> for u32 {
    fn from(value: CellIdx) -> Self {
        value.0
    }
}

impl From<CellIdx> for usize {
    fn from(value: CellIdx) -> Self {
        value.0 as usize
    }
}

impl Display for CellIdx {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Compact (4-bit) internal cell value that packs the user value and any solver inference.
///
/// # Encoding
///
/// | Value | Binary | Meaning | `is_safe()` | `has_count()` | `is_mine()` | `is_inferred()` | `cell_type()` |
/// | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
/// | `0..=8` | `0000`..`1000` | User Count (`0` to `8`) | `true` | `true` | `false` | `false` | [`CellType::Count`] |
/// | `9` | `1001` | Inferred Safe | `true` | `false` | `false` | `true` | [`CellType::InferredSafe`] |
/// | `10` | `1010` | Unknown | `false` | `false` | `false` | `false` | [`CellType::Unknown`] |
/// | `12` | `1100` | User Mine | `false` | `false` | `true` | `false` | [`CellType::Mine`] |
/// | `13` | `1101` | Inferred Mine | `false` | `false` | `true` | `true` | [`CellType::InferredMine`] |
///
/// # Invariants & Predicates
///
/// - [`Self::is_safe`]: value $\le 9$ (all count cells `0..=8` and inferred safe `9`).
/// - [`Self::has_count`]: value $< 9$ (`0..=8`). Note that [`Self::is_safe`] returns `true` for inferred values too;
///   use [`Self::safe_mine_count`] or [`Self::has_count`] to identify true count cells.
/// - [`Self::is_mine`]: value $\ge 12$ (user mine `12` and inferred mine `13`).
/// - [`Self::is_inferred`]: `value & 0b1001 == 0b1001` (matches only `9` and `13`).
/// - [`Self::cell_type`]: calculated via `value.saturating_sub(8)` mapped to [`CellType`].
#[derive(Debug, Default, Hash, Clone, Copy, PartialEq, Eq)]
pub(super) struct CellValue(u8);

impl CellValue {
    /// The solver deduced the cell is safe.
    pub(super) const INFERRED_SAFE: Self = Self(0b1001);
    /// Nothing is known about the cell.
    pub(super) const UNKNOWN: Self = Self(0b1010);
    /// Cell is an interior wall.
    pub(super) const WALL: Self = Self(0b1100);
    /// The cell is provided as a mine.
    pub(super) const MINE: Self = Self(0b1110);
    /// The solver deduced the cell is a mine.
    pub(super) const INFERRED_MINE: Self = Self(0b1111);

    /// Creates the value of a revealed, user-supplied count cell.
    pub(super) fn for_mine_count(mine_count: u8) -> CellValue {
        debug_assert!(mine_count <= 8);
        CellValue(mine_count)
    }

    #[inline]
    pub(super) fn is_safe(&self) -> bool {
        self.0 <= Self::INFERRED_SAFE.0
    }

    #[inline]
    pub(super) fn has_count(&self) -> bool {
        self.0 < Self::INFERRED_SAFE.0
    }

    #[inline]
    pub(super) fn is_mine(&self) -> bool {
        self.0 >= Self::MINE.0
    }

    #[inline]
    #[cfg(test)]
    pub(super) fn is_inferred(&self) -> bool {
        self.0 & Self::INFERRED_SAFE.0 == Self::INFERRED_SAFE.0
    }

    #[inline]
    pub(super) fn is_unknown(&self) -> bool {
        *self == Self::UNKNOWN
    }

    #[inline]
    pub(super) fn is_wall(&self) -> bool {
        *self == Self::WALL
    }

    /// For valid, non-inferred Safe cells, this returns the mine count.
    /// For all other types, returns 0.
    #[inline]
    pub(super) fn mine_count(&self) -> u8 {
        debug_assert!(self.has_count());
        self.0
    }

    /// For valid, non-inferred Safe cells, this returns the mine count.
    /// For all other types, returns None.
    pub(super) fn safe_mine_count(&self) -> Option<u8> {
        // NOTE: this is not the same as `.is_safe()`. This will not return
        // `true` for inferred values
        self.has_count().then_some(self.0)
    }

    /// TODO: Document
    pub(super) fn cell_type(&self) -> CellType {
        match self.0.saturating_sub(8) {
            0b000 => CellType::Count,
            0b001 => CellType::InferredSafe,
            0b010 => CellType::Unknown,
            0b100 => CellType::Wall,
            0b110 => CellType::Mine,
            0b111 => CellType::InferredMine,
            _ => unreachable!(),
        }
    }
}

impl From<UserCellValue> for CellValue {
    fn from(value: UserCellValue) -> Self {
        match value {
            UserCellValue::Unknown => Self::UNKNOWN,
            UserCellValue::Wall => Self::WALL,
            UserCellValue::Mine => Self::MINE,
            UserCellValue::Safe(c) => Self(c),
        }
    }
}

impl From<UserValueWithInference> for CellValue {
    fn from(value: UserValueWithInference) -> Self {
        match value {
            UserValueWithInference {
                value: UserCellValue::Mine,
                inferred: InferredValue::None,
            } => CellValue::MINE,
            UserValueWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::Mine,
            } => CellValue::INFERRED_MINE,
            UserValueWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::Safe,
            } => CellValue::INFERRED_SAFE,
            UserValueWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::None,
            } => CellValue::UNKNOWN,
            UserValueWithInference {
                value: UserCellValue::Safe(c),
                ..
            } => CellValue::for_mine_count(c),
            _ => unreachable!(),
        }
    }
}

impl From<CellValue> for UserValueWithInference {
    fn from(value: CellValue) -> Self {
        match value {
            CellValue::MINE => UserValueWithInference {
                value: UserCellValue::Mine,
                inferred: InferredValue::None,
            },
            CellValue::WALL => UserValueWithInference {
                value: UserCellValue::Wall,
                inferred: InferredValue::None,
            },
            CellValue::INFERRED_MINE => UserValueWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::Mine,
            },
            CellValue::INFERRED_SAFE => UserValueWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::Safe,
            },
            CellValue::UNKNOWN => UserValueWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::None,
            },
            c => UserValueWithInference {
                value: UserCellValue::Safe(c.0 & 0xf),
                inferred: InferredValue::None,
            },
        }
    }
}

impl Display for CellValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UNKNOWN => write!(f, "??"),
            Self::MINE => write!(f, "**"),
            Self::WALL => write!(f, "@@"),
            Self::INFERRED_MINE => write!(f, "!*"),
            Self::INFERRED_SAFE => write!(f, "!#"),
            CellValue(c) => write!(f, "{}#", c & 0xf),
        }
    }
}

/// High-level classification of a cell's value used to simplify traversal and neighbor updates.
///
/// Note: [`CellType::Count`] represents a user-provided count cell that has an adjacent mine count.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum CellType {
    /// An interior wall
    Wall = 0b000,
    /// User-provided safe cell with an adjacent mine count (`has_count() == true`).
    Count = Self::COUNT_MASK,
    /// Cell deduced by the solver to be safe.
    InferredSafe = Self::INFERRED_MASK,
    /// Cell whose state is unknown.
    #[default]
    Unknown = Self::UNKNOWN_MASK,
    /// Cell deduced by the solver to be a mine.
    InferredMine = Self::MINE_MASK | Self::INFERRED_MASK,
    /// User-provided mine.
    Mine = Self::MINE_MASK,
}

impl CellType {
    const COUNT_MASK: u8 = 0b1000;
    const UNKNOWN_MASK: u8 = 0b0100;
    const MINE_MASK: u8 = 0b0010;
    const INFERRED_MASK: u8 = 0b0001;
}

#[derive(Debug, Copy, Clone)]
pub(super) struct IdxAndCell {
    pub(super) idx: CellIdx,
    pub(super) value: CellValue,
}

impl PartialEq for IdxAndCell {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx
    }
}

impl Eq for IdxAndCell {}

impl PartialOrd for IdxAndCell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IdxAndCell {
    fn cmp(&self, other: &Self) -> Ordering {
        self.idx.cmp(&other.idx)
    }
}

/// Status returned by cell value updates indicating the impact on current inferences.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(super) enum ChangeStatus {
    /// The cell value did not change.
    NoChange,
    /// The value changed along a fast path where existing inferences remain valid.
    NonBreaking,
    /// The value changed in a way that may invalidate existing inferences.
    Breaking,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(super) struct CellFlags(u8);

impl CellFlags {
    pub(super) fn new() -> CellFlags {
        CellFlags(0)
    }

    #[inline]
    pub(super) fn is_marked(&self, mask: FlagMask) -> bool {
        self.0 & mask.0 != 0
    }

    #[inline]
    pub(super) fn mark(&mut self, mask: FlagMask) -> bool {
        if !self.is_marked(mask) {
            self.0 |= mask.0;
            true
        } else {
            false
        }
    }

    #[inline]
    pub(super) fn clear(&mut self, mask: FlagMask) {
        self.0 &= !mask.0;
    }
}

impl Display for CellFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        if self.is_marked(FlagMask::PENDING_PARTITION) {
            write!(f, "PENDING_PARTITION,")?;
        }
        if self.is_marked(FlagMask::SOLVED_QUEUE) {
            write!(f, "SOLVED,")?;
        }
        if self.is_marked(FlagMask::DESOLVED_QUEUE) {
            write!(f, "DESOLVED,")?;
        }
        write!(f, "]")
    }
}

#[derive(Debug, Copy, Clone)]
pub(super) struct FlagMask(u8);

impl FlagMask {
    pub(super) const PENDING_PARTITION: FlagMask = FlagMask(0b1);
    pub(super) const SOLVED_QUEUE: FlagMask = FlagMask(0b10);
    pub(super) const DESOLVED_QUEUE: FlagMask = FlagMask(0b100);
}

/// A queue of [`CellIdx`] instances backed by a [`VecDeque`] and guarded by a [`FlagMask`] in each cell's [`CellFlags`].
///
/// Ensures elements are added at most once without linear duplicate scans. When an element is pushed,
/// its corresponding flag bit is marked; pushing an already-marked cell is a no-op. When popped, the flag
/// bit is automatically cleared.
#[derive(Debug, Clone)]
pub(super) struct GuardedQueue {
    queue: VecDeque<CellIdx>,
    mask: FlagMask,
}

impl GuardedQueue {
    /// Creates a new [`GuardedQueue`] with the specified initial capacity and guarding [`FlagMask`].
    pub(super) fn with_capacity(capacity: usize, mask: FlagMask) -> GuardedQueue {
        GuardedQueue {
            queue: VecDeque::with_capacity(capacity),
            mask,
        }
    }

    /// Pushes `idx` to the back of the queue if not already marked with this queue's mask.
    pub(super) fn push(&mut self, cells: &mut [Cell], idx: CellIdx) {
        cells[idx]
            .flags()
            .mark(self.mask)
            .then(|| self.queue.push_back(idx));
    }

    /// Pops the front index from the queue and clears its corresponding flag mask.
    pub(super) fn pop(&mut self, cells: &mut [Cell]) -> Option<CellIdx> {
        self.queue
            .pop_front()
            .inspect(|&i| cells[i].flags().clear(self.mask))
    }

    /// Removes `idx` from the queue if present and clears its flag mask.
    pub(super) fn remove(&mut self, cells: &mut [Cell], idx: CellIdx) {
        if cells[idx].flags().is_marked(self.mask) {
            cells[idx].flags().clear(self.mask);
            if let Some(i) = self.queue.iter().position(|&i| i == idx) {
                self.queue.remove(i);
            }
        }
    }

    #[cfg(test)]
    pub(super) fn iter(&self) -> impl Iterator<Item = CellIdx> {
        self.queue.iter().copied()
    }

    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    #[cfg(test)]
    pub(super) fn clear(&mut self) {
        self.queue.clear();
    }
}

#[derive(Debug)]
pub(super) struct FastNumberHasher(u32);

impl Hasher for FastNumberHasher {
    // This is called sequentially for chunks of bytes in the key
    fn write(&mut self, bytes: &[u8]) {
        debug_assert!(
            false,
            "FastNumberHasher should only be used on types with a single primitive"
        );
        let mut padded = [0u8; 4];
        padded[..bytes.len().min(4)]
            .copy_from_slice(bytes[..bytes.len().min(4)].try_into().unwrap());
        self.0 = u32::from_le_bytes(padded);
    }

    fn write_u16(&mut self, i: u16) {
        self.0 = i as u32;
    }

    fn write_u32(&mut self, i: u32) {
        self.0 = i;
    }

    // Return the final computed hash value
    fn finish(&self) -> u64 {
        self.0 as u64
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub(super) struct FastNumberHasherBuilder;

impl BuildHasher for FastNumberHasherBuilder {
    type Hasher = FastNumberHasher;

    fn build_hasher(&self) -> Self::Hasher {
        FastNumberHasher(0)
    }
}

/// A known count cell annotated with its local combinatorial complexity (`solution_count`),
/// ordered from most constrained (fewest valid mine placements) to least constrained.
///
/// TODO: Implement the brute-force solution counting
#[derive(Debug, Copy, Hash, Clone, PartialEq, Eq)]
pub(super) struct CellOrdering {
    /// Linear index of the cell on the board.
    idx: CellIdx,
    /// Original count count on the cell.
    mine_count: u8,
    /// Number of mines still unaccounted for among the cell's unknown neighbors.
    remaining_mines: i8,
    /// Number of adjacent unknown cells available to hold `remaining_mines`.
    available_cells: u8,
    /// Number of ways to choose `remaining_mines` from `available_cells` ($\binom{\text{available}}{\text{remaining}}$).
    solution_count: u8,
}

#[allow(dead_code)]
impl CellOrdering {
    /// Constructs a [`CellOrdering`] if `cell` holds a known count (`Safe(n)`), or `None` otherwise.
    pub(super) fn from_cell(idx: CellIdx, cell: Cell) -> Option<CellOrdering> {
        if let Some(mine_count) = cell.value().safe_mine_count() {
            let available_cells = cell.alt_neighbors().len();
            let remaining_mines = cell.remaining_mines();
            let solution_count = stats::choose(available_cells, remaining_mines) as u8;
            Some(CellOrdering {
                idx,
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
    #[allow(dead_code)]
    pub(super) fn idx(&self) -> CellIdx {
        self.idx
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
        //
        // Order "easier" solves as greater-than "harder" - this makes use with
        // the BinaryHeap natural without needing to reverse it.
        other
            .solution_count
            .max(1)
            .cmp(&self.solution_count.max(1))
            .then(other.mine_count.cmp(&self.mine_count))
            .then(other.remaining_mines.cmp(&self.remaining_mines))
            .then(other.available_cells.cmp(&self.available_cells))
            .then(other.idx.cmp(&self.idx))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::solver::grid::{E, N, NW, S, SE, W};

    #[test]
    fn test_coordinates_and_cell_idx_round_trip() {
        let width = 7u32;
        for idx in 0..35u32 {
            let p = coordinates(CellIdx(idx), width);
            assert_eq!(cell_idx(p, width).0, idx);
        }
    }

    #[test]
    fn test_cell_value_is_known_and_display() {
        assert_eq!(UserCellValue::default(), UserCellValue::Unknown);
        assert_eq!(UserCellValue::Unknown.to_string(), "Unknown");
        assert_eq!(UserCellValue::Mine.to_string(), "MINE");
        assert_eq!(UserCellValue::Safe(4).to_string(), "4");
    }

    #[test]
    fn test_processed_cell_initialization_and_display() {
        let grid = Cell::grid(3, 3);
        let mut pc = grid[0];
        assert_eq!(pc.value, CellValue::UNKNOWN);
        assert_eq!(pc.neighbors(), E | S | SE);
        assert_eq!(pc.alt_neighbors(), DirectionSet::EMPTY);
        assert_eq!(pc.mine_neighbors(), DirectionSet::EMPTY);

        pc.value = CellValue::for_mine_count(2);

        let compact = format!("{pc}");
        let alternate = format!("{pc:#}");
        assert!(compact.contains("Cell{ value: 2"));
        assert!(alternate.contains("Cell{\n    value: 2"));
    }

    #[test]
    fn test_cell_value_for_mine_count_bounds() {
        assert_eq!(CellValue::for_mine_count(0).safe_mine_count(), Some(0));
        assert_eq!(CellValue::for_mine_count(8).safe_mine_count(), Some(8));
        // `Safe(0)` must be distinguishable from `Unknown`
        assert_ne!(CellValue::for_mine_count(0), CellValue::UNKNOWN);
        assert!(CellValue::for_mine_count(0).is_safe());
        assert!(!CellValue::for_mine_count(0).is_unknown());
        #[cfg(debug_assertions)]
        {
            assert!(std::panic::catch_unwind(|| CellValue::for_mine_count(9)).is_err());
        }
    }

    #[test]
    fn test_cell_value_predicates() {
        assert!(CellValue::UNKNOWN.is_unknown());
        assert!(!CellValue::UNKNOWN.is_safe() && !CellValue::UNKNOWN.is_mine());

        assert!(CellValue::MINE.is_mine());
        assert!(!CellValue::MINE.is_inferred());
        assert_eq!(CellValue::MINE.safe_mine_count(), None);

        assert!(CellValue::INFERRED_MINE.is_mine() && CellValue::INFERRED_MINE.is_inferred());
        assert!(CellValue::INFERRED_SAFE.is_safe() && CellValue::INFERRED_SAFE.is_inferred());
        // inferred-safe cells carry no count, so they must not look like a Count cell
        assert_eq!(CellValue::INFERRED_SAFE.safe_mine_count(), None);

        let three = CellValue::for_mine_count(3);
        assert!(three.is_safe() && !three.is_inferred() && !three.is_mine());
        assert_eq!(three.safe_mine_count(), Some(3));
        assert_eq!(three.mine_count(), 3);
    }

    #[test]
    fn test_cell_value_conversions_round_trip() {
        for user_value in [
            UserCellValue::Unknown,
            UserCellValue::Mine,
            UserCellValue::Safe(0),
            UserCellValue::Safe(4),
            UserCellValue::Safe(8),
        ] {
            let internal: CellValue = user_value.into();
            let with_inference: UserValueWithInference = internal.into();
            assert_eq!(with_inference.value, user_value);
            assert_eq!(with_inference.inferred, InferredValue::None);
            assert_eq!(CellValue::from(with_inference), internal);
        }

        let inferred_mine: UserValueWithInference = CellValue::INFERRED_MINE.into();
        assert_eq!(inferred_mine.value, UserCellValue::Unknown);
        assert_eq!(inferred_mine.inferred, InferredValue::Mine);
        assert_eq!(CellValue::from(inferred_mine), CellValue::INFERRED_MINE);

        let inferred_safe: UserValueWithInference = CellValue::INFERRED_SAFE.into();
        assert_eq!(inferred_safe.value, UserCellValue::Unknown);
        assert_eq!(inferred_safe.inferred, InferredValue::Safe);
        assert_eq!(CellValue::from(inferred_safe), CellValue::INFERRED_SAFE);
    }

    #[test]
    fn test_cell_is_invalid_detects_unsatisfiable_counts() {
        // 2x1 strip [Count(2), Unknown]: needs 2 mines but only 1 unknown neighbor -> invalid
        let mut cells = Cell::grid(2, 1);
        cells[0].set_value(CellValue::for_mine_count(2));
        assert!(
            cells[0].is_invalid(),
            "needs 2 mines but only 1 unknown neighbor"
        );

        let mut cells = Cell::grid(2, 1);
        cells[0].set_value(CellValue::for_mine_count(1));
        cells[0].alt_neighbors = !!E;
        assert!(!cells[0].is_invalid());
    }

    #[test]
    fn test_cell_is_invalid_handles_excess_mine_neighbors() {
        // a Count(0) next to a known mine is over-satisfied
        let mut cells = Cell::grid(2, 1);
        cells[0].set_value(CellValue::for_mine_count(0));
        cells[0].mine_neighbors |= E;
        assert!(cells[0].is_invalid());
    }

    #[test]
    fn test_ordered_cell_from_cell_and_ordering() {
        let grid = Cell::grid(3, 3);
        let mut unknown_cell = grid[0];
        assert!(CellOrdering::from_cell(CellIdx(0), unknown_cell).is_none());

        unknown_cell.set_value(CellValue::MINE);
        assert!(CellOrdering::from_cell(CellIdx(0), unknown_cell).is_none());

        // Trivial cell A: 2 available, 2 remaining -> choose(2, 2) = 1 solution
        let mut cell_a = grid[0];
        cell_a.set_value(CellValue::for_mine_count(2));
        cell_a.alt_neighbors = E | S;

        // Trivial cell B: 5 available, 0 remaining (1 mine already found) -> choose(5, 0) = 1 solution, lower mine_count
        let mut cell_b = grid[1];
        cell_b.set_value(CellValue::for_mine_count(1));
        cell_b.alt_neighbors = cell_b.neighbors - S;
        cell_b.mine_neighbors = !!S;

        // Non-trivial cell C: 3 available, 1 remaining -> choose(3, 1) = 3 solutions
        let mut cell_c = grid[2];
        cell_c.set_value(CellValue::for_mine_count(1));
        *cell_c.alt_neighbors_mut() = W | SW | S;

        let ord_a = CellOrdering::from_cell(CellIdx(0), cell_a).unwrap();
        let ord_b = CellOrdering::from_cell(CellIdx(1), cell_b).unwrap();
        let ord_c = CellOrdering::from_cell(CellIdx(2), cell_c).unwrap();

        assert_eq!(ord_a.remaining_mines(), 2);
        assert_eq!(ord_b.remaining_mines(), 0);
        assert_eq!(ord_c.remaining_mines(), 1);

        assert_eq!(ord_a.solution_count(), 1);
        assert_eq!(ord_b.solution_count(), 1);
        assert_eq!(ord_c.solution_count(), 3);

        // ord_b < ord_a (both have solution_count <= 1, but ord_b has smaller mine_count)
        assert!(ord_b > ord_a);
        // ord_a < ord_c (1 solution < 3 solutions)
        assert!(ord_a > ord_c);
    }

    #[test]
    fn test_neighbor_iter() {
        // use a 3x3 grid, indices going from 0..=8, center 4
        let neighbors: Vec<_> = Cell::neighbor_iter(CellIdx(4), 3, E | S | SE).collect();
        assert_eq!(
            neighbors.iter().map(|n| n.idx.0).collect::<Vec<_>>(),
            vec![5, 7, 8]
        );
        assert_eq!(
            neighbors.iter().map(|n| n.from).collect::<DirectionSet>(),
            W | N | NW
        );
    }

    #[test]
    fn test_cell_flags_and_display() {
        let mut flags = CellFlags::new();
        assert!(!flags.is_marked(FlagMask::PENDING_PARTITION));
        assert!(!flags.is_marked(FlagMask::SOLVED_QUEUE));
        assert!(!flags.is_marked(FlagMask::DESOLVED_QUEUE));
        assert_eq!(format!("{flags}"), "[]");

        flags.mark(FlagMask::PENDING_PARTITION);
        assert!(flags.is_marked(FlagMask::PENDING_PARTITION));
        assert_eq!(format!("{flags}"), "[PENDING_PARTITION,]");

        flags.mark(FlagMask::SOLVED_QUEUE);
        assert!(flags.is_marked(FlagMask::SOLVED_QUEUE));
        assert_eq!(format!("{flags}"), "[PENDING_PARTITION,SOLVED,]");

        flags.mark(FlagMask::DESOLVED_QUEUE);
        assert!(flags.is_marked(FlagMask::DESOLVED_QUEUE));
        assert_eq!(format!("{flags}"), "[PENDING_PARTITION,SOLVED,DESOLVED,]");

        flags.clear(FlagMask::PENDING_PARTITION);
        assert!(!flags.is_marked(FlagMask::PENDING_PARTITION));
        assert!(flags.is_marked(FlagMask::SOLVED_QUEUE));
        assert!(flags.is_marked(FlagMask::DESOLVED_QUEUE));

        flags.clear(FlagMask::SOLVED_QUEUE);
        assert!(!flags.is_marked(FlagMask::SOLVED_QUEUE));

        flags.clear(FlagMask::DESOLVED_QUEUE);
        assert!(!flags.is_marked(FlagMask::DESOLVED_QUEUE));
        assert_eq!(format!("{flags}"), "[]");
    }

    #[test]
    fn test_fast_number_hasher_and_builder() {
        let builder = FastNumberHasherBuilder;
        let mut hasher = builder.build_hasher();
        hasher.write_u16(42);
        assert_eq!(hasher.finish(), 42);

        hasher.write_u32(100_000);
        assert_eq!(hasher.finish(), 100_000);

        let mut map: HashMap<CellIdx, usize, FastNumberHasherBuilder> =
            HashMap::with_hasher(FastNumberHasherBuilder);
        map.insert(CellIdx(10), 100);
        map.insert(CellIdx(20), 200);

        assert_eq!(map.get(&CellIdx(10)), Some(&100));
        assert_eq!(map.get(&CellIdx(20)), Some(&200));
        assert_eq!(map.get(&CellIdx(30)), None);
    }

    #[test]
    fn test_cell_type_mapping() {
        assert_eq!(CellValue::UNKNOWN.cell_type(), CellType::Unknown);
        assert_eq!(CellValue::MINE.cell_type(), CellType::Mine);
        assert_eq!(CellValue::INFERRED_MINE.cell_type(), CellType::InferredMine);
        assert_eq!(CellValue::INFERRED_SAFE.cell_type(), CellType::InferredSafe);

        for count in 0..=8 {
            assert_eq!(
                CellValue::for_mine_count(count).cell_type(),
                CellType::Count,
                "Count {count} must map to CellType::Count"
            );
        }
    }

    #[test]
    fn test_cell_value_4bit_encoding() {
        for count in 0..=8 {
            let val = CellValue::for_mine_count(count);
            assert!(val.has_count());
            assert!(val.is_safe());
            assert!(!val.is_mine());
            assert!(!val.is_inferred());
            assert!(!val.is_unknown());
            assert_eq!(val.cell_type(), CellType::Count);
            assert_eq!(val.safe_mine_count(), Some(count));
            assert_eq!(val.mine_count(), count);
        }

        let inf_safe = CellValue::INFERRED_SAFE;
        assert!(!inf_safe.has_count());
        assert!(inf_safe.is_safe());
        assert!(!inf_safe.is_mine());
        assert!(inf_safe.is_inferred());
        assert!(!inf_safe.is_unknown());
        assert_eq!(inf_safe.cell_type(), CellType::InferredSafe);
        assert_eq!(inf_safe.safe_mine_count(), None);

        let unk = CellValue::UNKNOWN;
        assert!(!unk.has_count());
        assert!(!unk.is_safe());
        assert!(!unk.is_mine());
        assert!(!unk.is_inferred());
        assert!(unk.is_unknown());
        assert_eq!(unk.cell_type(), CellType::Unknown);
        assert_eq!(unk.safe_mine_count(), None);

        let mine = CellValue::MINE;
        assert!(!mine.has_count());
        assert!(!mine.is_safe());
        assert!(mine.is_mine());
        assert!(!mine.is_inferred());
        assert!(!mine.is_unknown());
        assert_eq!(mine.cell_type(), CellType::Mine);
        assert_eq!(mine.safe_mine_count(), None);

        let inf_mine = CellValue::INFERRED_MINE;
        assert!(!inf_mine.has_count());
        assert!(!inf_mine.is_safe());
        assert!(inf_mine.is_mine());
        assert!(inf_mine.is_inferred());
        assert!(!inf_mine.is_unknown());
        assert_eq!(inf_mine.cell_type(), CellType::InferredMine);
        assert_eq!(inf_mine.safe_mine_count(), None);
    }

    #[test]
    fn test_cell_neighbor_direction_sets_semantics() {
        let mut cells = Cell::grid(3, 3);
        let center = &mut cells[4];
        assert_eq!(center.neighbors(), DirectionSet::ALL);
        assert_eq!(center.alt_neighbors(), DirectionSet::EMPTY);
        assert_eq!(center.safe_neighbors(), DirectionSet::EMPTY);
        assert_eq!(center.mine_neighbors(), DirectionSet::EMPTY);

        *center.alt_neighbors_mut() = N | S;
        *center.safe_neighbors_mut() = E | W;
        *center.mine_neighbors_mut() = NE | NW | SE | SW;

        assert_eq!(center.alt_neighbors(), N | S);
        assert_eq!(center.safe_neighbors(), E | W);
        assert_eq!(center.mine_neighbors(), NE | NW | SE | SW);

        // Subsets and pairwise disjoint
        assert_eq!(
            center.alt_neighbors() | center.safe_neighbors() | center.mine_neighbors(),
            center.neighbors()
        );
        assert_eq!(
            center.alt_neighbors() & center.mine_neighbors(),
            DirectionSet::EMPTY
        );
        assert_eq!(
            center.safe_neighbors() & center.mine_neighbors(),
            DirectionSet::EMPTY
        );
        assert_eq!(
            center.alt_neighbors() & center.safe_neighbors(),
            DirectionSet::EMPTY
        );
    }

    #[test]
    fn test_guarded_queue_push_pop_remove_dedup() {
        let mut cells = Cell::grid(4, 1);
        let mut queue = GuardedQueue::with_capacity(4, FlagMask::PENDING_PARTITION);
        assert!(queue.is_empty());

        // First push succeeds
        queue.push(&mut cells, CellIdx(1));
        assert!(!queue.is_empty());
        assert!(cells[1].flags().is_marked(FlagMask::PENDING_PARTITION));

        // Duplicate push of the same index is ignored
        queue.push(&mut cells, CellIdx(1));
        let items: Vec<_> = queue.iter().collect();
        assert_eq!(items, vec![CellIdx(1)]);

        // Push another index
        queue.push(&mut cells, CellIdx(3));
        assert_eq!(
            queue.iter().collect::<Vec<_>>(),
            vec![CellIdx(1), CellIdx(3)]
        );

        // Remove CellIdx(1)
        queue.remove(&mut cells, CellIdx(1));
        assert!(!cells[1].flags().is_marked(FlagMask::PENDING_PARTITION));
        assert_eq!(queue.iter().collect::<Vec<_>>(), vec![CellIdx(3)]);

        // Removing non-existent or already removed element is safe no-op
        queue.remove(&mut cells, CellIdx(1));
        queue.remove(&mut cells, CellIdx(0));

        // Pop remaining item
        assert_eq!(queue.pop(&mut cells), Some(CellIdx(3)));
        assert!(!cells[3].flags().is_marked(FlagMask::PENDING_PARTITION));
        assert!(queue.is_empty());
        assert_eq!(queue.pop(&mut cells), None);
    }
}

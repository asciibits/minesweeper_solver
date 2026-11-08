//! Cell data structures representing both the external, user-facing board state
//! ([`UserCell`], [`UserCellValue`], [`UserCellWithInference`]) and internal cell representations.

use std::{
    cmp::Ordering,
    collections::{HashMap, VecDeque},
    fmt::{self, Display, Formatter},
    hash::{BuildHasher, Hasher},
    ops::{Index, IndexMut},
};

use crate::{
    MineField,
    solver::{
        Metrics,
        grid::{
            Direction, DirectionSet, DirectionSetIterator, E, N, NE, NW, Neighbor, Point, S, SE,
            SW, W,
        },
        stats,
    },
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
    /// The cell contains a mine.
    Mine,
    /// The cell is revealed and has the given number of adjacent mines (`0..=8`).
    Safe(u8),
}

impl Display for UserCellValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            UserCellValue::Unknown => write!(f, "Unknown"),
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

/// Compact (1 byte) internal cell value that packs the user value *and* any solver inference.
///
/// Bit layout: `S M I _ NNNN`
/// - `S` (`SAFE_MASK`): the cell is known (or inferred) to be safe.
/// - `M` (`MINE_MASK`): the cell is known (or inferred) to be a mine.
/// - `I` (`INFERRED_MASK`): the `S`/`M` bit was deduced by the solver rather than supplied.
/// - `NNNN`: for user-supplied safe cells, the count (adjacent mine count, `0..=8`).
///
/// An all-zero value is [`CellValue::UNKNOWN`]. Note that [`CellValue::is_safe`] and
/// [`CellValue::is_mine`] are `true` for inferred values too; use
/// [`CellValue::safe_mine_count`] to identify true count cells.
#[derive(Debug, Default, Hash, Clone, Copy, PartialEq, Eq)]
pub(super) struct CellValue(u8);

impl CellValue {
    const SAFE_MASK: u8 = 0b1000_0000;
    const MINE_MASK: u8 = 0b0100_0000;
    const INFERRED_MASK: u8 = 0b0010_0000;
    const KNOWN_MASK: u8 = Self::SAFE_MASK | Self::MINE_MASK;
    const META_MASK: u8 = Self::SAFE_MASK | Self::MINE_MASK | Self::INFERRED_MASK;

    /// Nothing is known about the cell.
    pub(super) const UNKNOWN: Self = Self(0);
    /// The cell is provided as a mine.
    pub(super) const MINE: Self = Self(Self::MINE_MASK);
    /// The solver deduced the cell is a mine.
    pub(super) const INFERRED_MINE: Self = Self(Self::MINE_MASK | Self::INFERRED_MASK);
    /// The solver deduced the cell is safe.
    pub(super) const INFERRED_SAFE: Self = Self(Self::SAFE_MASK | Self::INFERRED_MASK);

    /// Creates the value of a revealed, user-supplied count cell.
    pub(super) fn for_mine_count(mine_count: u8) -> CellValue {
        debug_assert!(mine_count <= 8);
        CellValue(Self::SAFE_MASK | mine_count)
    }

    /// Overwrites `self` with `value`, classifying the transition so callers know how much of
    /// the existing solver state must be discarded:
    /// - [`CellValueChangeType::NoChange`]: `value` equals the current value.
    /// - [`CellValueChangeType::NonBreaking`]: previous inferences remain valid.
    /// - [`CellValueChangeType::Breaking`]: previous inferences may be invalid, so affected
    ///   partitions must be reset and re-solved.
    fn set_value(&mut self, value: CellValue) -> CellValueChangeType {
        if *self == value {
            return CellValueChangeType::NoChange;
        }
        let prev = *self;
        *self = value;
        // Returns NonBreaking iff:
        // 1. Previous value was UNKNOWN
        // 2. Previous value was inferred, and new value is the same type, and
        //    not inferred.
        if prev.is_unknown()
            || (prev.0 & Self::META_MASK).wrapping_sub(value.0 & Self::META_MASK)
                == Self::INFERRED_MASK
        {
            CellValueChangeType::NonBreaking
        } else {
            CellValueChangeType::Breaking
        }
    }

    #[inline]
    #[cfg(test)]
    pub(super) fn is_safe(&self) -> bool {
        self.0 & Self::SAFE_MASK != 0
    }

    #[inline]
    pub(super) fn has_count(&self) -> bool {
        self.0 & Self::META_MASK == Self::SAFE_MASK
    }

    #[inline]
    pub(super) fn is_mine(&self) -> bool {
        self.0 & Self::MINE_MASK != 0
    }

    #[inline]
    pub(super) fn is_inferred(&self) -> bool {
        self.0 & Self::INFERRED_MASK != 0
    }

    #[inline]
    pub(super) fn is_unknown(&self) -> bool {
        self.0 == 0
    }

    /// Returns `true` if `other` is the same "type" is this. This means that both will return the
    /// same result for [`CellValue::is_safe`], [`CellValue::is_mine`], and
    /// [`CellValue::is_unknown`]. Note: This wil return `false` if one is an inferred mine and the
    /// other is a known mine - those are not considered "different". Same fore inferred/knows safe
    /// cells. Also Note: this wil return false if both cells are safe, even if they have different
    /// mine count.
    #[inline]
    pub(super) fn types_match(&self, other: CellValue) -> bool {
        (self.0 ^ other.0) & Self::KNOWN_MASK == 0
    }

    /// For valid, non-inferred Safe cells, this returns the mine count.
    /// For all other types, returns 0.
    #[inline]
    pub(super) fn mine_count(&self) -> u8 {
        self.0 & 0xf
    }

    /// For valid, non-inferred Safe cells, this returns the mine count.
    /// For all other types, returns None.
    pub(super) fn safe_mine_count(&self) -> Option<u8> {
        // NOTE: this is not the same as `.is_safe()`. This will not return
        // `true` for inferred values
        (self.0 & Self::META_MASK == Self::SAFE_MASK).then_some(self.mine_count())
    }
}

impl From<UserCellValue> for CellValue {
    fn from(value: UserCellValue) -> Self {
        match value {
            UserCellValue::Unknown => Self::UNKNOWN,
            UserCellValue::Mine => Self::MINE,
            UserCellValue::Safe(c) => Self(Self::SAFE_MASK | c),
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
            Self::INFERRED_MINE => write!(f, "!*"),
            Self::INFERRED_SAFE => write!(f, "!#"),
            CellValue(c) => write!(f, "{}#", c & 0xf),
        }
    }
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

/// How a [`CellValue::set_value`] call changed a cell.
#[derive(Debug, Default, Hash, Clone, Copy, PartialEq, Eq)]
pub(super) enum CellValueChangeType {
    /// The new value equals the old value.
    #[default]
    NoChange,
    /// The value changed but existing inferences are still valid.
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

/// A queue of CellIdx instances that ensures elements are only added once
#[derive(Debug, Clone)]
pub(super) struct GuardedQueue {
    queue: VecDeque<CellIdx>,
    mask: FlagMask,
}

impl GuardedQueue {
    pub(super) fn with_capacity(capacity: usize, mask: FlagMask) -> GuardedQueue {
        GuardedQueue {
            queue: VecDeque::with_capacity(capacity),
            mask,
        }
    }

    pub(super) fn push(&mut self, cells: &mut [Cell], idx: CellIdx) {
        cells[idx]
            .flags()
            .mark(self.mask)
            .then(|| self.queue.push_back(idx));
    }

    pub(super) fn pop(&mut self, cells: &mut [Cell]) -> Option<CellIdx> {
        self.queue
            .pop_front()
            .inspect(|&i| cells[i].flags().clear(self.mask))
    }

    // Super slow mechanism to remove elements from the associated Vec - try not to use this
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

/// Internal solver representation of a board cell, caching neighbor masks and partition membership.
///
/// A cell does not store its own index; callers identify a cell by its row-major position in the
/// `cells` slice (see [`coordinates`] / [`cell_idx`]).
// Note: Kept at 8 bytes so 8 cells fit in a single 64-byte cache line (enforced below).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Cell {
    /// Inline the contents of Cell
    value: CellValue,

    neighbors: DirectionSet,

    unknown_neighbors: DirectionSet,

    mine_neighbors: DirectionSet,

    partition_id: u16,

    flags: CellFlags,
}

const _: () = assert!(std::mem::size_of::<Cell>() <= 8);

impl Cell {
    pub(super) fn grid(width: u32, height: u32) -> Box<[Cell]> {
        let len = width as usize * height as usize;
        let mut grid: Vec<Cell> = Vec::with_capacity(len);

        for y in 0..height {
            for x in 0..width {
                let point = Point { x, y };
                let neighbors = DirectionSet::all_neighbors(point, width, height);
                grid.push(Cell {
                    value: CellValue::UNKNOWN,
                    neighbors,
                    unknown_neighbors: neighbors,
                    mine_neighbors: DirectionSet::new(),
                    partition_id: 0,
                    flags: CellFlags::new(),
                });
            }
        }

        grid.into_boxed_slice()
    }

    /// Returns the cell's current [`CellValue`].
    #[inline]
    pub(super) fn value(&self) -> CellValue {
        self.value
    }

    /// Returns the cell's current [`CellValue`] as a mutable reference.
    #[inline]
    pub(super) fn value_mut(&mut self) -> &mut CellValue {
        &mut self.value
    }

    /// Sets the cell's [`CellValue`].
    #[inline]
    #[cfg(test)]
    pub(super) fn set_value(&mut self, value: CellValue) {
        self.value = value;
    }

    /// Returns the ID of the partition this cell belongs to (`0` if unpartitioned).
    #[inline]
    pub(super) fn partition_id(&self) -> u16 {
        self.partition_id
    }

    /// Sets the ID of the partition this cell belongs to (`0` if unpartitioned).
    #[inline]
    pub(super) fn set_partition_id(&mut self, partition_id: u16) {
        self.partition_id = partition_id;
    }

    /// Returns the mask of all valid board neighbors for this cell.
    #[inline]
    pub(super) fn neighbors(&self) -> DirectionSet {
        self.neighbors
    }

    /// Returns the mask of neighbors still unknown after applying solver inferences.
    #[inline]
    pub(super) fn unknown_neighbors(&self) -> DirectionSet {
        self.unknown_neighbors
    }

    /// Returns the mask of neighbors known or inferred to be safe.
    #[inline]
    pub(super) fn safe_neighbors(&self) -> DirectionSet {
        self.neighbors - (self.mine_neighbors | self.unknown_neighbors)
    }

    /// Returns the mask of neighbors known or inferred to be mines.
    #[inline]
    pub(super) fn mine_neighbors(&self) -> DirectionSet {
        self.mine_neighbors
    }

    /// Returns mutable reference to the cell's status flags.
    #[inline]
    pub(super) fn flags(&mut self) -> &mut CellFlags {
        &mut self.flags
    }

    pub(super) fn remaining_mines(&self) -> i8 {
        self.value.mine_count() as i8 - self.mine_neighbors().len() as i8
    }

    /// Returns `true` if this cell's count cannot be satisfied by its neighbors, i.e. it needs
    /// more mines than it has unknown neighbors left to hold, or it has more
    /// mines than remaining spaces
    pub(super) fn is_invalid(&self) -> bool {
        let mine_neighbors = self.mine_neighbors().len() as i8;
        let unknown_neighbors = self.unknown_neighbors().len() as i8;
        self.value().safe_mine_count().is_some_and(|mine_count| {
            let mines_remaining = mine_count as i8 - mine_neighbors;
            mines_remaining < 0 || mines_remaining > unknown_neighbors
        })
    }

    // A fully solved cell is one with no neighbors. Note, this is distinct from a "solvable" cell,
    // which has all of its mines accounted for, but still has unknown neighbors to be set.
    pub(super) fn is_solved(&self) -> bool {
        self.unknown_neighbors().is_empty()
    }

    // Checks if this cell is solvable. This yields the same result as `infer_neighbors().is_some()`
    pub(super) fn is_solvaable(&self) -> bool {
        let unknown_count = self.unknown_neighbors().len() as i8;
        if let Some(mine_count) = self.value().safe_mine_count() {
            let remaining = mine_count as i8 - self.mine_neighbors().len() as i8;
            remaining <= 0 || remaining >= unknown_count
        } else {
            false
        }
    }

    // If the cell is a count cell that is not yet solved, and the neighbors can be inferred as
    // either a Mine or Safe...
    pub(super) fn infer_neighbors(&self) -> Option<CellValue> {
        let unknown_count = self.unknown_neighbors().len() as i8;

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

    pub(super) fn set_unknown(&mut self, direction: Direction) {
        self.unknown_neighbors |= direction;
        self.mine_neighbors -= direction;
    }

    pub(super) fn set_safe(&mut self, direction: Direction) {
        self.unknown_neighbors -= direction;
        self.mine_neighbors -= direction;
    }

    pub(super) fn set_mine(&mut self, direction: Direction) {
        self.unknown_neighbors -= direction;
        self.mine_neighbors |= direction;
    }

    /// Sets `cells[idx]` to `value` and updates the `unknown`/`mine` neighbor masks of every
    /// adjacent cell to match, then calls `visit_function` with each neighbor's index and its
    /// already-updated state. `visit_function` is not called if the value did not change.
    ///
    /// Returns how the cell's value changed. The caller is responsible for deciding which
    /// partitions need rebuilding based on the returned [`CellValueChangeType`].
    pub(super) fn change_value(
        idx: CellIdx,
        value: CellValue,
        field_args: &mut FieldArgs,
    ) -> CellValueChangeType {
        let cell = &mut field_args.cells[idx];

        let prev_val = cell.value();
        let change_type = cell.value_mut().set_value(value);

        let inference_breaking = match change_type {
            CellValueChangeType::NoChange => return CellValueChangeType::NoChange,
            CellValueChangeType::NonBreaking => false,
            CellValueChangeType::Breaking => true,
        };

        field_args
            .pending_partition_queue
            .push(field_args.cells, idx);

        Self::check_cell_solve(idx, inference_breaking, field_args);

        if !prev_val.types_match(value) {
            let mark_neighbor = if value.is_unknown() {
                Self::mark_neighbor_unknown
            } else if value.is_mine() {
                Self::mark_neighbor_mine
            } else {
                Self::mark_neighbor_safe
            };

            let neighbors = field_args.cells[idx].neighbors();
            field_args.metrics.inc_neighbor_visits(neighbors);

            for Neighbor {
                neighbor: neighbor_idx,
                from,
            } in Cell::neighbor_iter(idx, field_args.width, neighbors)
            {
                mark_neighbor(neighbor_idx, from, field_args);
            }
        }
        change_type
    }

    fn mark_neighbor_unknown(n_idx: CellIdx, from: Direction, field_args: &mut FieldArgs) {
        // if the cell starts out as solved, the transition of a neighbor to unknown changes its
        // solved state (unless it was invalid and over-mined)
        let neighbor = &mut field_args.cells[n_idx];
        let was_solved = neighbor.is_solved();
        neighbor.set_unknown(from);
        Self::check_cell_solve(n_idx, was_solved, field_args);

        // When a count neighbor reverts to unknown, any other count cells sharing this unknown
        // cell may have had their partition split and need to be repartitioned.
        if field_args.cells[n_idx].value().is_unknown() {
            let safe_neighbors = field_args.cells[n_idx].safe_neighbors();
            for Neighbor {
                neighbor: count_idx,
                ..
            } in Cell::neighbor_iter(n_idx, field_args.width, safe_neighbors)
            {
                let count_cell = &mut field_args.cells[count_idx];
                if count_cell.value().has_count() {
                    field_args
                        .pending_partition_queue
                        .push(field_args.cells, count_idx);
                }
            }
        }
    }

    fn mark_neighbor_mine(n_idx: CellIdx, from: Direction, field_args: &mut FieldArgs) {
        // if the cell starts out as a solved given cell, the transition of a neighbor
        // to unknown changes its solved state
        let neighbor = &mut field_args.cells[n_idx];
        let was_known = !neighbor.unknown_neighbors().contains(from);
        neighbor.set_mine(from);
        Self::check_cell_solve(n_idx, was_known, field_args);
    }

    fn mark_neighbor_safe(n_idx: CellIdx, from: Direction, field_args: &mut FieldArgs) {
        // if the cell starts out as a solved given cell, the transition of a neighbor
        // to unknown changes its solved state
        let neighbor = &mut field_args.cells[n_idx];
        let was_known = !neighbor.unknown_neighbors().contains(from);
        neighbor.set_safe(from);
        Self::check_cell_solve(n_idx, was_known, field_args);
    }

    fn check_cell_solve(idx: CellIdx, inference_breaking: bool, field_args: &mut FieldArgs) {
        let cell = field_args.cells[idx];
        if inference_breaking {
            // Inferences attached to this cell's partition may be invalidated. Reset inferred.
            field_args.desolved_queue.push(field_args.cells, idx);
        }
        if cell.value().has_count() {
            field_args
                .pending_partition_queue
                .push(field_args.cells, idx);
            if cell.is_solvaable() {
                field_args.solved_queue.push(field_args.cells, idx);
            }
        }
    }

    pub(super) fn neighbor_iter(
        idx: CellIdx,
        width: u32,
        directions: DirectionSet,
    ) -> impl Iterator<Item = Neighbor<CellIdx>> {
        DirectionSetIterator::new(directions).map(move |d| match d.index() {
            0 => Neighbor {
                neighbor: CellIdx(idx.0 - width - 1),
                from: SE,
            },
            1 => Neighbor {
                neighbor: CellIdx(idx.0 - width),
                from: S,
            },
            2 => Neighbor {
                neighbor: CellIdx(idx.0 - width + 1),
                from: SW,
            },
            3 => Neighbor {
                neighbor: CellIdx(idx.0 - 1),
                from: E,
            },
            4 => Neighbor {
                neighbor: CellIdx(idx.0 + 1),
                from: W,
            },
            5 => Neighbor {
                neighbor: CellIdx(idx.0 + width - 1),
                from: NE,
            },
            6 => Neighbor {
                neighbor: CellIdx(idx.0 + width),
                from: N,
            },
            7 => Neighbor {
                neighbor: CellIdx(idx.0 + width + 1),
                from: NW,
            },
            _ => unreachable!(),
        })
    }

    #[cfg(test)]
    pub(super) fn reset(&mut self) {
        self.mine_neighbors = DirectionSet::EMPTY;
        self.unknown_neighbors = self.neighbors;
        self.value = CellValue::UNKNOWN;
    }
}

impl Display for Cell {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let spacer = if f.alternate() { "\n    " } else { " " };
        write!(
            f,
            "Cell{{{spacer}value: {},{spacer}neighbors: {},{spacer}unknown_neighbors: {},{spacer}mine_neighbors: {},{spacer}safe_neighbors: {},{spacer}partition_id: {},{spacer}flags: {}",
            self.value,
            self.neighbors,
            self.unknown_neighbors(),
            self.mine_neighbors(),
            self.safe_neighbors(),
            self.partition_id,
            self.flags
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

#[derive(Debug)]
pub(super) struct FieldArgs<'field> {
    pub(super) cells: &'field mut [Cell],
    pub(super) width: u32,
    pub(super) next_partition_id: &'field mut u16,
    pub(super) desolved_queue: &'field mut GuardedQueue,
    pub(super) solved_queue: &'field mut GuardedQueue,
    pub(super) pending_partition_queue: &'field mut GuardedQueue,
    pub(super) metrics: &'field mut Metrics,
    pub(super) partitions: &'field mut HashMap<CellIdx, usize, FastNumberHasherBuilder>,
}

impl<'field> FieldArgs<'field> {
    pub(super) fn new(field: &'field mut MineField) -> Self {
        FieldArgs {
            cells: &mut field.cells,
            width: field.width,
            next_partition_id: &mut field.next_partition_id,
            desolved_queue: &mut field.desolved_queue,
            solved_queue: &mut field.solved_queue,
            pending_partition_queue: &mut field.pending_partition_queue,
            metrics: &mut field.metrics,
            partitions: &mut field.partitions,
        }
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
    pub(super) fn from_cell(idx: CellIdx, cell: &Cell) -> Option<CellOrdering> {
        if let Some(mine_count) = cell.value().safe_mine_count() {
            let available_cells = cell.unknown_neighbors().len();
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
        assert_eq!(pc.neighbors, E | S | SE);
        assert_eq!(pc.unknown_neighbors(), E | S | SE);
        assert_eq!(pc.mine_neighbors(), DirectionSet::new());

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
        assert!(std::panic::catch_unwind(|| CellValue::for_mine_count(9)).is_err());
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
    fn test_cell_value_set_value_change_types() {
        let one = CellValue::for_mine_count(1);

        // identical values are NoChange
        let mut v = one;
        assert_eq!(v.set_value(one), CellValueChangeType::NoChange);

        // confirming an inference (same known-ness) is non-breaking
        let mut v = CellValue::INFERRED_SAFE;
        assert_eq!(v.set_value(one), CellValueChangeType::NonBreaking);
        assert_eq!(v, one);
        let mut v = CellValue::INFERRED_MINE;
        assert_eq!(
            v.set_value(CellValue::MINE),
            CellValueChangeType::NonBreaking
        );

        // contradicting an inference is breaking
        let mut v = CellValue::INFERRED_SAFE;
        assert_eq!(v.set_value(CellValue::MINE), CellValueChangeType::Breaking);
        let mut v = CellValue::INFERRED_MINE;
        assert_eq!(v.set_value(one), CellValueChangeType::Breaking);
    }

    #[test]
    fn test_cell_value_set_value_change_types_intended_semantics() {
        let one = CellValue::for_mine_count(1);
        let two = CellValue::for_mine_count(2);

        // nothing was inferred about an Unknown cell, so revealing it cannot break anything
        let mut v = CellValue::UNKNOWN;
        assert_eq!(v.set_value(one), CellValueChangeType::NonBreaking);
        let mut v = CellValue::UNKNOWN;
        assert_eq!(
            v.set_value(CellValue::MINE),
            CellValueChangeType::NonBreaking
        );

        // changing a count invalidates every inference derived from it
        let mut v = one;
        assert_eq!(v.set_value(two), CellValueChangeType::Breaking);

        // removing information invalidates inferences derived from it
        let mut v = one;
        assert_eq!(
            v.set_value(CellValue::UNKNOWN),
            CellValueChangeType::Breaking
        );
        let mut v = CellValue::MINE;
        assert_eq!(
            v.set_value(CellValue::UNKNOWN),
            CellValueChangeType::Breaking
        );
    }

    #[test]
    fn test_cell_change_value_updates_neighbor_masks() {
        // 3x1 strip, change the middle cell and watch both neighbors
        let mut field = MineField::new(3, 1).unwrap();

        Cell::change_value(CellIdx(1), CellValue::MINE, &mut FieldArgs::new(&mut field));
        assert_eq!(field.cells[1].value(), CellValue::MINE);
        assert_eq!(field.cells[0].mine_neighbors(), !!E);
        assert!(field.cells[0].unknown_neighbors().is_empty());
        assert_eq!(field.cells[2].mine_neighbors(), !!W);
        assert!(field.cells[2].unknown_neighbors().is_empty());

        // Mine -> Safe(1): neighbors drop the mine but stay known (not unknown)
        Cell::change_value(
            CellIdx(1),
            CellValue::for_mine_count(1),
            &mut FieldArgs::new(&mut field),
        );
        assert!(field.cells[0].mine_neighbors().is_empty());
        assert!(field.cells[0].unknown_neighbors().is_empty());
        assert_eq!(field.cells[0].safe_neighbors(), !!E);

        // Safe(1) -> Unknown: neighbors regain an unknown neighbor
        Cell::change_value(
            CellIdx(1),
            CellValue::UNKNOWN,
            &mut FieldArgs::new(&mut field),
        );
        assert_eq!(field.cells[0].unknown_neighbors(), !!E);
        assert_eq!(field.cells[2].unknown_neighbors(), !!W);
        assert!(field.cells[0].mine_neighbors().is_empty());
    }

    #[test]
    fn test_cell_is_invalid_detects_unsatisfiable_counts() {
        // 2x1 strip [Count(2), Unknown]: needs 2 mines but only 1 unknown neighbor -> invalid
        let mut cells = Cell::grid(2, 1);
        cells[0].value_mut().set_value(CellValue::for_mine_count(2));
        assert!(
            cells[0].is_invalid(),
            "needs 2 mines but only 1 unknown neighbor"
        );

        let mut cells = Cell::grid(2, 1);
        cells[0].value_mut().set_value(CellValue::for_mine_count(1));
        assert!(!cells[0].is_invalid());
    }

    #[test]
    fn test_cell_is_invalid_handles_excess_mine_neighbors() {
        // Unknown cell next to a mine is perfectly valid
        let mut cells = Cell::grid(2, 1);
        cells[1].set_mine(W);
        assert!(!cells[1].is_invalid());

        // a Count(0) next to a known mine is over-satisfied
        let mut cells = Cell::grid(2, 1);
        cells[0].value_mut().set_value(CellValue::for_mine_count(0));
        cells[0].set_mine(E);
        assert!(cells[0].is_invalid());
    }

    #[test]
    fn test_ordered_cell_from_cell_and_ordering() {
        let grid = Cell::grid(3, 3);
        let mut unknown_cell = grid[0];
        assert!(CellOrdering::from_cell(CellIdx(0), &unknown_cell).is_none());

        unknown_cell.value_mut().set_value(CellValue::MINE);
        assert!(CellOrdering::from_cell(CellIdx(0), &unknown_cell).is_none());

        // Trivial cell A: 2 available, 2 remaining -> choose(2, 2) = 1 solution
        let mut cell_a = grid[0];
        cell_a.value_mut().set_value(CellValue::for_mine_count(2));
        cell_a.set_safe(SE);

        // Trivial cell B: 2 available, 0 remaining (1 mine already found) -> choose(2, 0) = 1 solution, lower mine_count
        let mut cell_b = grid[1];
        cell_b.value_mut().set_value(CellValue::for_mine_count(1));
        cell_b.set_mine(S);

        // Non-trivial cell C: 3 available, 1 remaining -> choose(3, 1) = 3 solutions
        let mut cell_c = grid[2];
        cell_c.value_mut().set_value(CellValue::for_mine_count(1));

        let ord_a = CellOrdering::from_cell(CellIdx(0), &cell_a).unwrap();
        let ord_b = CellOrdering::from_cell(CellIdx(1), &cell_b).unwrap();
        let ord_c = CellOrdering::from_cell(CellIdx(2), &cell_c).unwrap();

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
            neighbors.iter().map(|n| n.neighbor.0).collect::<Vec<_>>(),
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
}

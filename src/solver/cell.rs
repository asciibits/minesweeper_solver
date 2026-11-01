//! Cell data structures representing both the external, user-facing board state
//! ([`UserCell`], [`UserCellValue`], [`UserCellWithInference`]) and internal cell representations.

use std::{
    cmp::Ordering,
    fmt::{self, Display, Formatter},
    ops::{Index, IndexMut},
};

use crate::solver::{
    IndexedOffsets, Metrics,
    grid::{Direction, DirectionSet, Neighbor, Point},
    indexed_neighbor_iter,
    partition::PartitionId,
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
pub struct UserCellWithInference {
    /// The value supplied by the user.
    pub value: UserCellValue,
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
/// - `NNNN`: for user-supplied safe cells, the clue (adjacent mine count, `0..=8`).
///
/// An all-zero value is [`CellValue::UNKNOWN`]. Note that [`CellValue::is_safe`] and
/// [`CellValue::is_mine`] are `true` for inferred values too; use
/// [`CellValue::safe_mine_count`] to identify true clue cells.
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

    /// Creates the value of a revealed, user-supplied clue cell.
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
    pub(super) fn set_value(&mut self, value: CellValue) -> CellValueChangeType {
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
    /// other is a known mine. Same fore inferred/knows safe cells. Also Note: this wil return false
    /// if both cells are safe, even if they have different mine count.
    #[inline]
    pub(super) fn is_different_type(&self, other: CellValue) -> bool {
        (self.0 ^ other.0) & Self::KNOWN_MASK != 0
    }

    /// Returns if this value and `other` are alternating types as occur during
    /// partitioning. This means one is safe (inferred or not), while the other
    /// is unknown (i.e. not inferred and not known).
    pub(super) fn is_alt(&self, other: Self) -> bool {
        (self.0 ^ other.0) & Self::KNOWN_MASK == Self::SAFE_MASK
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

impl From<UserCellWithInference> for CellValue {
    fn from(value: UserCellWithInference) -> Self {
        match value {
            UserCellWithInference {
                value: UserCellValue::Mine,
                inferred: InferredValue::None,
            } => CellValue::MINE,
            UserCellWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::Mine,
            } => CellValue::INFERRED_MINE,
            UserCellWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::Safe,
            } => CellValue::INFERRED_SAFE,
            UserCellWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::None,
            } => CellValue::UNKNOWN,
            UserCellWithInference {
                value: UserCellValue::Safe(c),
                ..
            } => CellValue::for_mine_count(c),
            _ => unreachable!(),
        }
    }
}

impl From<CellValue> for UserCellWithInference {
    fn from(value: CellValue) -> Self {
        match value {
            CellValue::MINE => UserCellWithInference {
                value: UserCellValue::Mine,
                inferred: InferredValue::None,
            },
            CellValue::INFERRED_MINE => UserCellWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::Mine,
            },
            CellValue::INFERRED_SAFE => UserCellWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::Safe,
            },
            CellValue::UNKNOWN => UserCellWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::None,
            },
            c => UserCellWithInference {
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
pub(super) struct CellFlags(bool);

impl CellFlags {
    pub(super) fn new() -> CellFlags {
        CellFlags(false)
    }

    #[inline]
    pub(super) fn is_set(&self) -> bool {
        self.0
    }

    #[inline]
    pub(super) fn set(&mut self) {
        self.0 = true
    }

    #[inline]
    pub(super) fn clear(&mut self) {
        self.0 = false
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

    /// The partition this cell belongs to (`0` if unpartitioned).
    partition_id: PartitionId,

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
                    partition_id: PartitionId::new(),
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

    /// Updates the cell's [`CellValue`].
    #[inline]
    #[cfg(test)]
    pub(super) fn set_value(&mut self, value: CellValue) {
        self.value = value;
    }

    /// Returns the mask of all valid board neighbors for this cell.
    #[inline]
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub(super) fn mine_neighbors(&self) -> DirectionSet {
        self.mine_neighbors
    }

    /// Returns the ID of the partition this cell belongs to (`0` if unpartitioned).
    #[inline]
    pub(super) fn partition_id(&self) -> PartitionId {
        self.partition_id
    }

    /// Returns the ID of the partition this cell belongs to (`0` if unpartitioned).
    #[inline]
    pub(super) fn flags(&mut self) -> &mut CellFlags {
        &mut self.flags
    }

    /// Updates the ID of the partition this cell belongs to.
    #[inline]
    pub(super) fn set_partition_id(&mut self, partition_id: PartitionId) {
        self.partition_id = partition_id;
    }

    pub(super) fn alt_neighbors(&self) -> DirectionSet {
        if self.value.is_unknown() {
            self.safe_neighbors()
        } else if self.value.is_safe() {
            self.unknown_neighbors
        } else {
            DirectionSet::EMPTY
        }
    }

    pub(super) fn remaining_mines(&self) -> i8 {
        self.value.mine_count() as i8 - self.mine_neighbors.len() as i8
    }

    /// Returns `true` if this cell's clue cannot be satisfied by its neighbors, i.e. it needs
    /// more mines than it has unknown neighbors left to hold, or it has more
    /// mines than remaining spaces
    pub(super) fn is_invalid(&self) -> bool {
        if !self.value.has_count() {
            // only safe cells with a count can be invalid
            return false;
        }
        let mines_remaining = self.remaining_mines();
        mines_remaining < 0 || mines_remaining > self.unknown_neighbors.len() as i8
    }

    pub(super) fn set_unknown(&mut self, direction: Direction) -> bool {
        if self.unknown_neighbors.contains(direction) {
            false
        } else {
            self.unknown_neighbors |= direction;
            self.mine_neighbors -= direction;
            true
        }
    }

    pub(super) fn set_safe(&mut self, direction: Direction) -> bool {
        if (self.unknown_neighbors | self.mine_neighbors).contains(direction) {
            self.unknown_neighbors -= direction;
            self.mine_neighbors -= direction;
            true
        } else {
            false
        }
    }

    pub(super) fn set_mine(&mut self, direction: Direction) -> bool {
        if self.mine_neighbors.contains(direction) {
            false
        } else {
            self.unknown_neighbors -= direction;
            self.mine_neighbors |= direction;
            true
        }
    }

    /// Sets `cells[idx]` to `value` and updates the `unknown`/`mine` neighbor masks of every
    /// adjacent cell to match, then calls `visit_function` with each neighbor's index and its
    /// already-updated state. `visit_function` is not called if the value did not change.
    ///
    /// Returns how the cell's value changed. The caller is responsible for deciding which
    /// partitions need rebuilding based on the returned [`CellValueChangeType`].
    pub(super) fn change_value<F>(
        cells: &mut [Cell],
        idx: CellIdx,
        value: CellValue,
        offsets: IndexedOffsets,
        metrics: &mut Metrics,
        mut visit_function: F,
    ) -> CellValueChangeType
    where
        F: FnMut(CellIdx, &mut Cell, &mut Metrics),
    {
        let cell = &mut cells[idx];
        let prev_val = cell.value();
        let change_type = cell.value_mut().set_value(value);

        if !matches!(change_type, CellValueChangeType::NoChange)
            && prev_val.is_different_type(value)
        {
            let updater = if value.is_unknown() {
                |neighbor: &mut Cell, from: Direction| neighbor.set_unknown(from)
            } else if value.is_mine() {
                |neighbor: &mut Cell, from: Direction| neighbor.set_mine(from)
            } else {
                |neighbor: &mut Cell, from: Direction| neighbor.set_safe(from)
            };

            metrics.inc_neighbor_visits(cell.neighbors);

            for Neighbor {
                neighbor: neighbor_idx,
                from,
            } in indexed_neighbor_iter(idx, cell.neighbors, offsets)
            {
                let neighbor = &mut cells[neighbor_idx];
                if updater(neighbor, from) {
                    visit_function(neighbor_idx, neighbor, metrics);
                }
            }
        }

        change_type
    }

    pub(super) fn reset(&mut self) {
        self.mine_neighbors = DirectionSet::EMPTY;
        self.unknown_neighbors = self.neighbors;
        self.partition_id = PartitionId::new();
        self.value = CellValue::UNKNOWN;
    }
}

impl Display for Cell {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let spacer = if f.alternate() { "\n    " } else { " " };
        write!(
            f,
            "Cell{{{spacer}value: {},{spacer}neighbors: {},{spacer}unknown_neighbors: {},{spacer}mine_neighbors: {},{spacer}partition_id: {}",
            self.value,
            self.neighbors,
            self.unknown_neighbors,
            self.mine_neighbors,
            self.partition_id
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

/// A known clue cell annotated with its local combinatorial complexity (`solution_count`),
/// ordered from most constrained (fewest valid mine placements) to least constrained.
#[derive(Debug, Copy, Hash, Clone, PartialEq, Eq)]
pub(super) struct CellOrdering {
    /// Linear index of the cell on the board.
    idx: CellIdx,
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
    /// Constructs a [`CellOrdering`] if `cell` holds a known clue (`Safe(n)`), or `None` otherwise.
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
    use crate::solver::grid::{E, N, NE, NW, S, SE, W};

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
        assert_eq!(pc.unknown_neighbors, E | S | SE);
        assert_eq!(pc.mine_neighbors, DirectionSet::new());
        assert_eq!(pc.partition_id, PartitionId::new());

        pc.value = CellValue::for_mine_count(2);

        let compact = format!("{pc}");
        let alternate = format!("{pc:#}");
        assert!(compact.contains("Cell{ value: 2"));
        assert!(alternate.contains("Cell{\n    value: 2"));
    }

    #[test]
    fn test_processed_cell_alt_neighbors_and_partition_state_reset() {
        let grid = Cell::grid(3, 3);
        let mut pc = grid[0];
        pc.neighbors -= NW | N | NE | W;
        pc.set_safe(S);
        pc.set_mine(SE);

        // Unknown cells return safe_neighbors from alt_neighbors()
        pc.value = CellValue::UNKNOWN;
        assert_eq!(pc.alt_neighbors(), !!S);

        // Count cells return unknown_neighbors from alt_neighbors()
        pc.value = CellValue::for_mine_count(1);
        assert_eq!(pc.alt_neighbors(), !!E);

        // Mine cells return an empty set from alt_neighbors()
        pc.value = CellValue::MINE;
        assert_eq!(pc.alt_neighbors(), DirectionSet::new());
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
        // inferred-safe cells carry no clue, so they must not look like a Count cell
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
            let with_inference: UserCellWithInference = internal.into();
            assert_eq!(with_inference.value, user_value);
            assert_eq!(with_inference.inferred, InferredValue::None);
            assert_eq!(CellValue::from(with_inference), internal);
        }

        let inferred_mine: UserCellWithInference = CellValue::INFERRED_MINE.into();
        assert_eq!(inferred_mine.value, UserCellValue::Unknown);
        assert_eq!(inferred_mine.inferred, InferredValue::Mine);
        assert_eq!(CellValue::from(inferred_mine), CellValue::INFERRED_MINE);

        let inferred_safe: UserCellWithInference = CellValue::INFERRED_SAFE.into();
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

        // changing a clue invalidates every inference derived from it
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
        let mut cells = Cell::grid(3, 1);
        let offsets = IndexedOffsets::new(3);

        let mut visited = Vec::new();
        let change = Cell::change_value(
            &mut cells,
            CellIdx(1),
            CellValue::MINE,
            offsets,
            &mut Metrics::new(),
            |i, _, _| visited.push(i.0),
        );
        assert_ne!(change, CellValueChangeType::NoChange);
        visited.sort();
        assert_eq!(visited, vec![0, 2]);
        assert_eq!(cells[1].value(), CellValue::MINE);
        assert_eq!(cells[0].mine_neighbors(), !!E);
        assert!(cells[0].unknown_neighbors().is_empty());
        assert_eq!(cells[2].mine_neighbors(), !!W);
        assert!(cells[2].unknown_neighbors().is_empty());

        // Mine -> Safe(1): neighbors drop the mine but stay known (not unknown)
        Cell::change_value(
            &mut cells,
            CellIdx(1),
            CellValue::for_mine_count(1),
            offsets,
            &mut Metrics::new(),
            |_, _, _| {},
        );
        assert!(cells[0].mine_neighbors().is_empty());
        assert!(cells[0].unknown_neighbors().is_empty());
        assert_eq!(cells[0].safe_neighbors(), !!E);

        // Safe(1) -> Unknown: neighbors regain an unknown neighbor
        Cell::change_value(
            &mut cells,
            CellIdx(1),
            CellValue::UNKNOWN,
            offsets,
            &mut Metrics::new(),
            |_, _, _| {},
        );
        assert_eq!(cells[0].unknown_neighbors(), !!E);
        assert_eq!(cells[2].unknown_neighbors(), !!W);
        assert!(cells[0].mine_neighbors().is_empty());
    }

    #[test]
    fn test_cell_is_invalid_detects_unsatisfiable_clues() {
        // 2x1 strip [Count(2), Unknown]: needs 2 mines but only 1 unknown neighbor -> invalid
        let mut cells = Cell::grid(2, 1);
        cells[0].set_value(CellValue::for_mine_count(2));
        assert!(
            cells[0].is_invalid(),
            "needs 2 mines but only 1 unknown neighbor"
        );

        let mut cells = Cell::grid(2, 1);
        cells[0].set_value(CellValue::for_mine_count(1));
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
        cells[0].set_value(CellValue::for_mine_count(0));
        cells[0].set_mine(E);
        assert!(cells[0].is_invalid());
    }

    #[test]
    fn test_ordered_cell_from_cell_and_ordering() {
        let grid = Cell::grid(3, 3);
        let mut unknown_cell = grid[0];
        assert!(CellOrdering::from_cell(CellIdx(0), &unknown_cell).is_none());

        unknown_cell.set_value(CellValue::MINE);
        assert!(CellOrdering::from_cell(CellIdx(0), &unknown_cell).is_none());

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
}

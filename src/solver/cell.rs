//! Cell data structures representing both the external, user-facing board state
//! ([`UserCell`], [`UserCellValue`], [`UserCellWithInference`]) and internal cell representations.

use std::{
    cmp::Ordering,
    collections::VecDeque,
    fmt::{self, Display, Formatter},
    hash::{BuildHasher, Hasher},
    ops::{Index, IndexMut},
};

use crate::{
    MineField,
    solver::{
        grid::{ALL_DIRECTIONS, Direction, DirectionSet, Point},
        stats,
    },
};

/// Provide the x/y coordinates of a Cell with the given cell_id in a fixed
/// width field. This is the corollary to `cell_id`.
pub(super) fn coordinates(idx: u32, width: u32) -> Point {
    Point::from_index(idx, width)
}

/// Get a unique ID for the cell in a given-width field. Useful for indexing in
/// a Vector or Array. This is the corollary to `coordinates`.
pub(super) fn cell_idx(p: Point, width: u32) -> u32 {
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
    Opened(u8),
}

impl Display for UserCellValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            UserCellValue::Unknown => write!(f, "Unknown"),
            UserCellValue::Wall => write!(f, "Wall"),
            UserCellValue::Mine => write!(f, "MINE"),
            UserCellValue::Opened(c) => write!(f, "{c}"),
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

impl CellIdx {
    pub(super) fn neighbor_iter(&self, width: u32, directions: DirectionSet) -> NeighborIterator {
        let idx = self.0;
        let idx_u = idx - width;
        let idx_d = idx + width;

        NeighborIterator {
            mask: directions.mask(),
            neighbor_indices: [
                CellIdx(idx_u - 1),
                CellIdx(idx_u),
                CellIdx(idx_u + 1),
                CellIdx(idx - 1),
                CellIdx(idx + 1),
                CellIdx(idx_d - 1),
                CellIdx(idx_d),
                CellIdx(idx_d + 1),
            ],
        }
    }
}

pub(super) struct NeighborIterator {
    mask: u8,
    neighbor_indices: [CellIdx; 8],
}

impl Iterator for NeighborIterator {
    type Item = Neighbor;
    fn next(&mut self) -> Option<Self::Item> {
        if self.mask == 0 {
            return None;
        }
        let t = self.mask;
        self.mask &= t - 1;
        let i = t.trailing_zeros() as usize;
        Some(Neighbor {
            idx: self.neighbor_indices[i],
            from: ALL_DIRECTIONS[i].reverse(),
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.mask.count_ones() as usize;
        (len, Some(len))
    }
}
impl ExactSizeIterator for NeighborIterator {}

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
/// | Value | Binary | Meaning | `is_open()` | `is_clear()` | `is_mine()` | `is_inferred()` | `cell_type()` |
/// | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
/// | `0..=8` | `0000`..`1000` | Open with adjacent mine count (`0` to `8`) | `true` | `true` | `false` | `false` | [`CellType::Open`] |
/// | `9` | `1001` | Inferred Clear | `false` | `true` | `false` | `true` | [`CellType::Clear`] |
/// | `10` | `1010` | Unknown | `false` | `false` | `false` | `false` | [`CellType::Unknown`] |
/// | `12` | `1100` | Interior Wall | `false` | `false` | `false` | `false` | [`CellType::Wall`] |
/// | `14` | `1110` | User Mine | `false` | `false` | `true` | `false` | [`CellType::Mine`] |
/// | `15` | `1111` | Inferred Mine (Flag) | `false` | `false` | `true` | `true` | [`CellType::Flag`] |
///
/// # Invariants & Predicates
///
/// - [`Self::is_clear`]: value $\le 9$ (all open cells `0..=8` and inferred clear `9`).
/// - [`Self::is_open`]: value $< 9$ (`0..=8`). Note that [`Self::is_clear`] returns `true` for inferred clear cells too;
///   use [`Self::safe_mine_count`] or [`Self::is_open`] to identify true opened cells.
/// - [`Self::is_mine`]: value $\ge 14$ (user mine `14` and inferred flag `15`).
/// - [`Self::is_wall`]: value $== 12$ ([`Self::WALL`]).
/// - [`Self::is_inferred`]: `value & 0b1001 == 0b1001` (matches only `9` and `15`).
/// - [`Self::cell_type`]: calculated via `value.saturating_sub(8)` mapped to [`CellType`].
#[derive(Debug, Default, Hash, Clone, Copy, PartialEq, Eq)]
pub(super) struct CellValue(u8);

impl CellValue {
    const INFERRED_MASK: u8 = 0b1001;

    /// The solver deduced the cell is safe.
    pub(super) const CLEARED: Self = Self(0b1001);
    /// Nothing is known about the cell.
    pub(super) const UNKNOWN: Self = Self(0b1010);
    /// Cell is an interior wall.
    pub(super) const WALL: Self = Self(0b1100);
    /// The cell is provided as a mine.
    pub(super) const MINE: Self = Self(0b1110);
    /// The solver deduced the cell is a mine.
    pub(super) const FLAGGED: Self = Self(0b1111);

    /// Creates the value of a revealed, user-supplied count cell.
    pub(super) fn for_mine_count(mine_count: u8) -> CellValue {
        debug_assert!(mine_count <= 8);
        CellValue(mine_count)
    }

    #[inline]
    pub(super) fn is_cleared(&self) -> bool {
        *self == Self::CLEARED
    }

    #[inline]
    #[cfg(test)]
    pub(super) fn is_safe(&self) -> bool {
        self.0 <= Self::CLEARED.0
    }

    #[inline]
    pub(super) fn is_opened(&self) -> bool {
        self.0 < Self::CLEARED.0
    }

    #[inline]
    pub(super) fn is_mine(&self) -> bool {
        self.0 >= Self::MINE.0
    }

    #[inline]
    pub(super) fn is_inferred(&self) -> bool {
        self.0 & Self::INFERRED_MASK == Self::INFERRED_MASK
    }

    #[inline]
    pub(super) fn is_unknown(&self) -> bool {
        *self == Self::UNKNOWN
    }

    #[inline]
    pub(super) fn is_wall(&self) -> bool {
        *self == Self::WALL
    }

    #[inline]
    pub(super) fn is_wall_or_mine(&self) -> bool {
        self.0 >= Self::WALL.0
    }

    /// For open cells, this returns the mine count.
    #[inline]
    pub(super) fn mine_count(&self) -> u8 {
        debug_assert!(self.is_opened());
        self.0
    }

    /// For open cells, this returns the mine count as `Some(count)`.
    /// For all other cell types, returns `None`.
    pub(super) fn safe_mine_count(&self) -> Option<u8> {
        self.is_opened().then_some(self.0)
    }

    /// Returns the high-level [`CellType`] classification for this cell value.
    pub(super) fn cell_type(&self) -> CellType {
        match self.0.saturating_sub(8) {
            0b000 => CellType::Opened,
            0b001 => CellType::Cleared,
            0b010 => CellType::Unknown,
            0b100 => CellType::Wall,
            0b110 => CellType::Mine,
            0b111 => CellType::Flag,
            _ => unreachable!(),
        }
    }

    pub(super) fn remaining_mines(&self, neighbors: CellNeighbors) -> i8 {
        // only valid for count cells
        debug_assert!(self.is_opened());
        self.mine_count() as i8 - neighbors.mines.len() as i8
    }

    /// Returns `true` if this cell's count cannot be satisfied by its neighbors, i.e. it needs
    /// more mines than it has unknown neighbors left to hold, or it has more
    /// mines than remaining spaces
    pub(super) fn is_invalid(&self, neighbors: CellNeighbors) -> bool {
        // only valid for count cells
        debug_assert!(self.is_opened());
        let mine_neighbors = neighbors.mines.len() as i8;
        let unknown_neighbors = neighbors.unknown.len() as i8;
        self.safe_mine_count().is_some_and(|mine_count| {
            let mines_remaining = mine_count as i8 - mine_neighbors;
            mines_remaining < 0 || mines_remaining > unknown_neighbors
        })
    }

    // If the cell is a count cell that is not yet solved, and the neighbors can be inferred as
    // either a Mine or Safe...
    pub(super) fn infer_neighbors(&self, neighbors: CellNeighbors) -> Option<CellValue> {
        // only valid for count cells
        debug_assert!(self.is_opened());
        let unknown_count = neighbors.unknown.len() as i8;

        self.safe_mine_count().and_then(|mine_count| {
            let remaining = mine_count as i8 - neighbors.mines.len() as i8;
            if remaining <= 0 {
                Some(CellValue::CLEARED)
            } else if remaining >= unknown_count {
                Some(CellValue::FLAGGED)
            } else {
                None
            }
        })
    }

    pub(super) fn apply_change(
        field: &mut MineField,
        idx: CellIdx,
        new_value: CellValue,
    ) -> ChangeStatus {
        let prev_value = field.cells[idx];
        let change_function = match prev_value.cell_type() {
            CellType::Unknown => match new_value.cell_type() {
                CellType::Unknown => Self::no_change,
                CellType::Mine => Self::unknown_to_mine,
                CellType::Flag => Self::unknown_to_flagged,
                CellType::Cleared => Self::unknown_to_clear,
                CellType::Opened => Self::unknown_to_open,
                CellType::Wall => Self::general_change_value,
            },
            CellType::Mine => match new_value.cell_type() {
                CellType::Mine => Self::no_change,
                CellType::Flag => Self::mine_to_flag,
                _ => Self::general_change_value,
            },
            CellType::Flag => match new_value.cell_type() {
                CellType::Mine => Self::no_change,
                CellType::Flag => Self::flag_to_mine,
                _ => Self::general_change_value,
            },
            CellType::Cleared => match new_value.cell_type() {
                CellType::Cleared => Self::no_change,
                CellType::Opened => Self::clear_to_open,
                _ => Self::general_change_value,
            },
            CellType::Opened => match new_value.cell_type() {
                CellType::Opened => {
                    if prev_value == new_value {
                        Self::no_change
                    } else {
                        Self::open_to_open
                    }
                }
                _ => Self::general_change_value,
            },
            CellType::Wall => Self::general_change_value,
        };

        change_function(field, idx, new_value)
    }

    fn no_change(_field: &mut MineField, _idx: CellIdx, _new_value: CellValue) -> ChangeStatus {
        ChangeStatus::NoChange
    }

    fn flag_to_mine(field: &mut MineField, idx: CellIdx, new_value: CellValue) -> ChangeStatus {
        let unknown_neighbors = field.cell_neighbors[idx].all();
        field.metrics.inc_neighbor_visits(unknown_neighbors);
        for neighbor in idx.neighbor_iter(field.width_with_border, unknown_neighbors) {
            let neighbors = &mut field.cell_neighbors[neighbor.idx];
            neighbors.inferred -= neighbor.from;
        }
        field.cells[idx] = new_value;
        ChangeStatus::NoChange
    }

    fn mine_to_flag(field: &mut MineField, idx: CellIdx, new_value: CellValue) -> ChangeStatus {
        let unknown_neighbors = field.cell_neighbors[idx].all();
        field.metrics.inc_neighbor_visits(unknown_neighbors);
        for neighbor in idx.neighbor_iter(field.width_with_border, unknown_neighbors) {
            let neighbors = &mut field.cell_neighbors[neighbor.idx];
            neighbors.inferred |= neighbor.from;
        }
        field.cells[idx] = new_value;
        ChangeStatus::NoChange
    }

    fn open_to_open(field: &mut MineField, idx: CellIdx, new_value: CellValue) -> ChangeStatus {
        // count -> count (diff counts) - it could be de-solved, or solved
        field.cells[idx] = new_value;
        field.solved_queue.push(&mut field.cell_flags, idx);
        field.desolved_queue.push(&mut field.cell_flags, idx);
        ChangeStatus::Breaking
    }

    fn unknown_to_mine(field: &mut MineField, idx: CellIdx, new_value: CellValue) -> ChangeStatus {
        let open_neighbors = field.cell_neighbors[idx].opened;
        field.metrics.inc_neighbor_visits(open_neighbors);
        for neighbor in idx.neighbor_iter(field.width_with_border, open_neighbors) {
            let neighbors = &mut field.cell_neighbors[neighbor.idx];
            neighbors.mines |= neighbor.from;
            neighbors.unknown -= neighbor.from;
            field.solved_queue.push(&mut field.cell_flags, neighbor.idx);
        }
        let remaining = field.cell_neighbors[idx].mines
            | field.cell_neighbors[idx].unknown
            | field.cell_neighbors[idx].inferred;
        field.metrics.inc_neighbor_visits(remaining);
        for neighbor in idx.neighbor_iter(field.width_with_border, remaining) {
            let neighbors = &mut field.cell_neighbors[neighbor.idx];
            neighbors.mines |= neighbor.from;
            neighbors.unknown -= neighbor.from;
        }
        field.cells[idx] = new_value;
        ChangeStatus::NonBreaking
    }

    fn unknown_to_flagged(
        field: &mut MineField,
        idx: CellIdx,
        new_value: CellValue,
    ) -> ChangeStatus {
        let open_neighbors = field.cell_neighbors[idx].opened;
        field.metrics.inc_neighbor_visits(open_neighbors);
        for neighbor in idx.neighbor_iter(field.width_with_border, open_neighbors) {
            let neighbors = &mut field.cell_neighbors[neighbor.idx];
            neighbors.mines |= neighbor.from;
            neighbors.inferred |= neighbor.from;
            neighbors.unknown -= neighbor.from;
            field.solved_queue.push(&mut field.cell_flags, neighbor.idx);
        }
        let remaining = field.cell_neighbors[idx].mines
            | field.cell_neighbors[idx].unknown
            | field.cell_neighbors[idx].inferred;
        field.metrics.inc_neighbor_visits(remaining);
        for neighbor in idx.neighbor_iter(field.width_with_border, remaining) {
            let neighbors = &mut field.cell_neighbors[neighbor.idx];
            neighbors.mines |= neighbor.from;
            neighbors.inferred |= neighbor.from;
            neighbors.unknown -= neighbor.from;
        }
        field.cells[idx] = new_value;
        ChangeStatus::NonBreaking
    }

    fn unknown_to_clear(field: &mut MineField, idx: CellIdx, new_value: CellValue) -> ChangeStatus {
        let open_neighbors = field.cell_neighbors[idx].opened;
        field.metrics.inc_neighbor_visits(open_neighbors);
        for neighbor in idx.neighbor_iter(field.width_with_border, open_neighbors) {
            let neighbors = &mut field.cell_neighbors[neighbor.idx];
            neighbors.unknown -= neighbor.from;
            neighbors.inferred |= neighbor.from;
            field.solved_queue.push(&mut field.cell_flags, neighbor.idx);
        }
        let remaining = field.cell_neighbors[idx].mines
            | field.cell_neighbors[idx].unknown
            | field.cell_neighbors[idx].inferred;
        field.metrics.inc_neighbor_visits(remaining);
        for neighbor in idx.neighbor_iter(field.width_with_border, remaining) {
            let neighbors = &mut field.cell_neighbors[neighbor.idx];
            neighbors.unknown -= neighbor.from;
            neighbors.inferred |= neighbor.from;
        }
        field.cells[idx] = new_value;

        ChangeStatus::NonBreaking
    }

    fn unknown_to_open(field: &mut MineField, idx: CellIdx, new_value: CellValue) -> ChangeStatus {
        // opened:  -unknown, +solve
        // cleared: -unknown
        // unknown: -unknown,+opened
        // self: alt = unknown, +solve
        let opened_neighbors = field.cell_neighbors[idx].opened;
        field.metrics.inc_neighbor_visits(opened_neighbors);
        for neighbor in idx.neighbor_iter(field.width_with_border, opened_neighbors) {
            let neighbors = &mut field.cell_neighbors[neighbor.idx];
            neighbors.unknown -= neighbor.from;
            neighbors.opened |= neighbor.from;
            field.solved_queue.push(&mut field.cell_flags, neighbor.idx);
        }
        let remaining = field.cell_neighbors[idx].mines
            | field.cell_neighbors[idx].unknown
            | field.cell_neighbors[idx].inferred;
        field.metrics.inc_neighbor_visits(remaining);
        for neighbor in idx.neighbor_iter(field.width_with_border, remaining) {
            let neighbors = &mut field.cell_neighbors[neighbor.idx];
            neighbors.unknown -= neighbor.from;
            neighbors.opened |= neighbor.from;
        }
        field.cells[idx] = new_value;
        field.solved_queue.push(&mut field.cell_flags, idx);

        ChangeStatus::NonBreaking
    }

    fn clear_to_open(field: &mut MineField, idx: CellIdx, new_value: CellValue) -> ChangeStatus {
        // opened: n/a
        // cleared: n/a
        // unknown:  -cleared, +opened
        // self: +solve
        let unknown_neighbors = field.cell_neighbors[idx].all();
        field.metrics.inc_neighbor_visits(unknown_neighbors);
        for neighbor in idx.neighbor_iter(field.width_with_border, unknown_neighbors) {
            let neighbors = &mut field.cell_neighbors[neighbor.idx];
            neighbors.inferred -= neighbor.from;
            neighbors.opened |= neighbor.from;
        }
        field.cells[idx] = new_value;
        field.solved_queue.push(&mut field.cell_flags, idx);

        ChangeStatus::NonBreaking
    }

    // The slower, not-unrolled general purpose function for the rest of the values. Always involves
    // a loop over all neighbors with a decent amount of branching
    fn general_change_value(
        field: &mut MineField,
        idx: CellIdx,
        new_value: CellValue,
    ) -> ChangeStatus {
        let prev_value = field.cells[idx];

        if prev_value == new_value {
            return ChangeStatus::NoChange;
        }
        let change_status = if prev_value.is_mine() && new_value.is_mine() {
            // going to/from flagged/mine is a no-change from all inference perspective
            ChangeStatus::NoChange
        } else if prev_value.is_unknown() && !new_value.is_wall()
            || prev_value.is_cleared() && new_value.is_opened()
        {
            ChangeStatus::NonBreaking
        } else {
            ChangeStatus::Breaking
        };

        let mut unknown = DirectionSet::new();
        let mut opened = DirectionSet::new();
        let mut mines = DirectionSet::new();
        let mut inferred = DirectionSet::new();

        let neighbors = field.cell_neighbors[idx].all();
        field.metrics.inc_neighbor_visits(neighbors);
        for neighbor in idx.neighbor_iter(field.width_with_border, neighbors) {
            let neighbor_value = field.cells[neighbor.idx];

            if change_status == ChangeStatus::Breaking {
                // We only track the minimal neighbor sets, so that "non-breaking" changes will
                // always have the accurate neighbor states upon update. Breaking changes, however
                // require the neighbor states to be updated.
                if !new_value.is_wall_or_mine() {
                    let dir = neighbor.from.reverse();
                    match neighbor_value.cell_type() {
                        CellType::Flag => {
                            mines |= dir;
                            inferred |= dir;
                        }
                        CellType::Mine => mines |= dir,
                        CellType::Cleared => inferred |= dir,
                        CellType::Unknown => unknown |= dir,
                        CellType::Opened => {
                            opened |= dir;
                            field
                                .desolved_queue
                                .push(&mut field.cell_flags, neighbor.idx);
                            field.solved_queue.push(&mut field.cell_flags, neighbor.idx);
                        }
                        CellType::Wall => (),
                    }
                } else if neighbor_value.is_opened() {
                    field
                        .desolved_queue
                        .push(&mut field.cell_flags, neighbor.idx);
                    field.solved_queue.push(&mut field.cell_flags, neighbor.idx);
                }
            } else if neighbor_value.is_opened() {
                field.solved_queue.push(&mut field.cell_flags, neighbor.idx);
            }

            let neighbor_neighbors = &mut field.cell_neighbors[neighbor.idx];
            if !neighbor_value.is_wall_or_mine() {
                let mut neighbor_missing = false;
                match prev_value.cell_type() {
                    CellType::Unknown => {
                        neighbor_missing = true;
                        neighbor_neighbors.unknown -= neighbor.from;
                    }
                    CellType::Mine | CellType::Flag => {
                        neighbor_missing = true;
                        neighbor_neighbors.mines -= neighbor.from;
                    }
                    CellType::Opened => {
                        field
                            .desolved_queue
                            .push(&mut field.cell_flags, neighbor.idx);
                    }
                    _ => (),
                }
                match new_value.cell_type() {
                    CellType::Unknown => {
                        neighbor_missing = false;
                        neighbor_neighbors.unknown |= neighbor.from;
                    }
                    CellType::Mine | CellType::Flag => {
                        neighbor_missing = false;
                        neighbor_neighbors.mines |= neighbor.from;
                    }
                    _ => (),
                }
                if neighbor_value.is_unknown() || neighbor_missing {
                    match prev_value.cell_type() {
                        CellType::Flag => {
                            neighbor_neighbors.inferred -= neighbor.from;
                        }
                        CellType::Opened => {
                            neighbor_neighbors.opened -= neighbor.from;
                        }
                        CellType::Cleared => {
                            neighbor_neighbors.inferred -= neighbor.from;
                        }
                        _ => (),
                    }
                    match new_value.cell_type() {
                        CellType::Flag => {
                            neighbor_neighbors.inferred |= neighbor.from;
                        }
                        CellType::Opened => {
                            neighbor_neighbors.opened |= neighbor.from;
                        }
                        CellType::Cleared => {
                            neighbor_neighbors.inferred |= neighbor.from;
                        }
                        _ => (),
                    }
                }
            }
        }
        if change_status == ChangeStatus::Breaking && !new_value.is_wall_or_mine() {
            // for breaking changes, update all neighbor references.
            let neighbors = &mut field.cell_neighbors[idx];
            neighbors.set_unknown(unknown);
            neighbors.set_mines(mines);
            neighbors.set_opened(opened);
            neighbors.set_inferred(inferred);
        }

        field.cells[idx] = new_value;
        if prev_value.is_opened() {
            field.desolved_queue.push(&mut field.cell_flags, idx);
        }
        if new_value.is_opened() {
            field.solved_queue.push(&mut field.cell_flags, idx);
        }

        change_status
    }
}

impl From<UserCellValue> for CellValue {
    fn from(value: UserCellValue) -> Self {
        match value {
            UserCellValue::Unknown => Self::UNKNOWN,
            UserCellValue::Wall => Self::WALL,
            UserCellValue::Mine => Self::MINE,
            UserCellValue::Opened(c) => Self(c),
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
                value: UserCellValue::Wall,
                inferred: InferredValue::None,
            } => CellValue::WALL,
            UserValueWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::Mine,
            } => CellValue::FLAGGED,
            UserValueWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::Safe,
            } => CellValue::CLEARED,
            UserValueWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::None,
            } => CellValue::UNKNOWN,
            UserValueWithInference {
                value: UserCellValue::Opened(c),
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
            CellValue::FLAGGED => UserValueWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::Mine,
            },
            CellValue::CLEARED => UserValueWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::Safe,
            },
            CellValue::UNKNOWN => UserValueWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::None,
            },
            c => UserValueWithInference {
                value: UserCellValue::Opened(c.0 & 0xf),
                inferred: InferredValue::None,
            },
        }
    }
}

impl Index<CellIdx> for [CellValue] {
    type Output = CellValue;

    #[inline]
    fn index(&self, index: CellIdx) -> &Self::Output {
        &self[index.0 as usize]
    }
}

impl IndexMut<CellIdx> for [CellValue] {
    #[inline]
    fn index_mut(&mut self, index: CellIdx) -> &mut Self::Output {
        &mut self[index.0 as usize]
    }
}

impl Display for CellValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UNKNOWN => write!(f, "??"),
            Self::MINE => write!(f, "**"),
            Self::WALL => write!(f, "@@"),
            Self::FLAGGED => write!(f, "!*"),
            Self::CLEARED => write!(f, "!#"),
            CellValue(c) => write!(f, "{}#", c & 0xf),
        }
    }
}

/// High-level classification of a cell's value used to simplify traversal and neighbor updates.
///
/// Note: [`CellType::Open`] represents a user-provided open cell that has an adjacent mine count.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum CellType {
    /// An interior wall
    Wall = 0b000,
    /// User-provided safe cell with an adjacent mine count (`is_open() == true`).
    Opened = Self::OPEN_MASK,
    /// Cell deduced by the solver to be safe.
    Cleared = Self::INFERRED_MASK,
    /// Cell whose state is unknown.
    #[default]
    Unknown = Self::UNKNOWN_MASK,
    /// Cell deduced by the solver to be a mine.
    Flag = Self::MINE_MASK | Self::INFERRED_MASK,
    /// User-provided mine.
    Mine = Self::MINE_MASK,
}

impl CellType {
    const UNKNOWN_MASK: u8 = 0b0001;
    const OPEN_MASK: u8 = 0b0010;
    const MINE_MASK: u8 = 0b0100;
    const INFERRED_MASK: u8 = 0b1000;
}

/// Associates a neighboring item `T` (such as a [`Point`] or cell reference) with the
/// [`Direction`] pointing from that neighbor back to the origin cell.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub(super) struct Neighbor {
    /// The neighboring value or coordinate.
    pub idx: CellIdx,
    /// The direction from `neighbor` back to the origin cell.
    pub from: Direction,
}

pub(super) fn create_grid_with_border(
    width: u32,
    height: u32,
) -> (Box<[CellValue]>, Box<[CellNeighbors]>) {
    let width = width as usize;
    let height = height as usize;
    let len = (width + 1) * (height + 2) + 1;
    let mut values = vec![CellValue::UNKNOWN; len];
    let mut neighbors = vec![CellNeighbors::all_unknown(); len];

    // Set the border
    values[0..width + 2].fill(CellValue::WALL);
    values[len - width - 2..len].fill(CellValue::WALL);
    for value in values.iter_mut().skip(width + 1).step_by(width + 1) {
        *value = CellValue::WALL;
    }

    // Set the edge neighbors
    for neighbors in neighbors.iter_mut().skip(width + 2).take(width) {
        // top border row
        neighbors.unknown -= DirectionSet::N_DIRECTIONS;
    }
    for neighbors in neighbors.iter_mut().skip(len - 2 * width - 2).take(width) {
        // bottom row
        neighbors.unknown -= DirectionSet::S_DIRECTIONS;
    }
    for neighbors in neighbors
        .iter_mut()
        .skip(width + 2)
        .step_by(width + 1)
        .take(height)
    {
        // left col
        neighbors.unknown -= DirectionSet::W_DIRECTIONS;
    }
    for neighbors in neighbors
        .iter_mut()
        .skip(2 * width + 1)
        .step_by(width + 1)
        .take(height)
    {
        // right col
        neighbors.unknown -= DirectionSet::E_DIRECTIONS;
    }

    (values.into_boxed_slice(), neighbors.into_boxed_slice())
}

#[derive(Debug, Copy, Clone, Default)]
pub(super) struct CellNeighbors {
    /// TODO
    unknown: DirectionSet,

    /// TODO
    opened: DirectionSet,

    /// TODO
    mines: DirectionSet,

    /// TODO
    inferred: DirectionSet,
}

impl CellNeighbors {
    pub(super) fn all_unknown() -> Self {
        Self {
            unknown: DirectionSet::ALL,
            opened: DirectionSet::EMPTY,
            mines: DirectionSet::EMPTY,
            inferred: DirectionSet::EMPTY,
        }
    }

    pub(super) fn unknown(&self) -> DirectionSet {
        self.unknown
    }

    pub(super) fn set_unknown(&mut self, unknown: DirectionSet) {
        self.unknown = unknown;
    }

    pub(super) fn opened(&self) -> DirectionSet {
        self.opened
    }

    pub(super) fn set_opened(&mut self, open: DirectionSet) {
        self.opened = open;
    }

    pub(super) fn inferred(&self) -> DirectionSet {
        self.inferred
    }

    pub(super) fn set_inferred(&mut self, inferred: DirectionSet) {
        self.inferred = inferred;
    }

    pub(super) fn mines(&self) -> DirectionSet {
        self.mines
    }

    pub(super) fn set_mines(&mut self, mines: DirectionSet) {
        self.mines = mines;
    }

    pub(super) fn all(&self) -> DirectionSet {
        self.unknown | self.opened | self.mines | self.inferred
    }

    #[cfg(test)]
    pub(super) fn reset(&mut self) {
        self.unknown = self.all();
        self.opened = DirectionSet::EMPTY;
        self.mines = DirectionSet::EMPTY;
        self.inferred = DirectionSet::EMPTY;
    }
}

impl Display for CellNeighbors {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let spacer = if f.alternate() { "\n    " } else { " " };
        write!(f, "CellNeighbors{{{spacer}unknown: {},", self.unknown,)?;
        write!(f, "CellNeighbors{{{spacer}open: {},", self.opened,)?;
        write!(f, "CellNeighbors{{{spacer}mine: {},", self.mines,)?;
        write!(f, "CellNeighbors{{{spacer}inferred: {},", self.inferred,)?;
        if f.alternate() {
            writeln!(f)?;
        }
        write!(f, " }}")
    }
}

impl Index<CellIdx> for [CellNeighbors] {
    type Output = CellNeighbors;

    #[inline]
    fn index(&self, index: CellIdx) -> &Self::Output {
        &self[index.0 as usize]
    }
}

impl IndexMut<CellIdx> for [CellNeighbors] {
    #[inline]
    fn index_mut(&mut self, index: CellIdx) -> &mut Self::Output {
        &mut self[index.0 as usize]
    }
}

#[cfg(test)]
pub(super) fn cells_equal(
    cell_a: CellValue,
    neighbors_a: CellNeighbors,
    cell_b: CellValue,
    neighbors_b: CellNeighbors,
) -> bool {
    if cell_a != cell_b {
        return false;
    }
    if !cell_a.is_wall_or_mine() {
        if neighbors_a.mines != neighbors_b.mines || neighbors_a.unknown != neighbors_b.unknown {
            return false;
        }
        if cell_a.is_unknown()
            && (neighbors_a.opened != neighbors_b.opened
                || neighbors_a.inferred != neighbors_b.inferred)
        {
            return false;
        }
    }
    true
}

#[derive(Debug, Copy, Clone)]
pub(super) struct IndexedCell {
    pub(super) idx: CellIdx,
    pub(super) value: CellValue,
}

impl PartialEq for IndexedCell {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx
    }
}

impl Eq for IndexedCell {}

impl PartialOrd for IndexedCell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IndexedCell {
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

impl Index<CellIdx> for [CellFlags] {
    type Output = CellFlags;

    #[inline]
    fn index(&self, index: CellIdx) -> &Self::Output {
        &self[index.0 as usize]
    }
}

impl IndexMut<CellIdx> for [CellFlags] {
    #[inline]
    fn index_mut(&mut self, index: CellIdx) -> &mut Self::Output {
        &mut self[index.0 as usize]
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
    pub(super) fn push(&mut self, flags: &mut [CellFlags], idx: CellIdx) {
        flags[idx]
            .mark(self.mask)
            .then(|| self.queue.push_back(idx));
    }

    /// Pops the front index from the queue and clears its corresponding flag mask.
    pub(super) fn pop(&mut self, flags: &mut [CellFlags]) -> Option<CellIdx> {
        self.queue
            .pop_front()
            .inspect(|&i| flags[i].clear(self.mask))
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
    pub(super) fn from_cell(
        idx: CellIdx,
        cell: CellValue,
        neighbors: CellNeighbors,
    ) -> Option<CellOrdering> {
        if let Some(mine_count) = cell.safe_mine_count() {
            let available_cells = neighbors.unknown.len();
            let remaining_mines = cell.remaining_mines(neighbors);
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
    use crate::solver::grid::{E, N, NW, S, SE, SW, W};

    #[test]
    fn test_cell_value_is_known_and_display() {
        assert_eq!(UserCellValue::default(), UserCellValue::Unknown);
        assert_eq!(UserCellValue::Unknown.to_string(), "Unknown");
        assert_eq!(UserCellValue::Wall.to_string(), "Wall");
        assert_eq!(UserCellValue::Mine.to_string(), "MINE");
        assert_eq!(UserCellValue::Opened(4).to_string(), "4");

        assert_eq!(CellValue::UNKNOWN.to_string(), "??");
        assert_eq!(CellValue::WALL.to_string(), "@@");
        assert_eq!(CellValue::MINE.to_string(), "**");
        assert_eq!(CellValue::FLAGGED.to_string(), "!*");
        assert_eq!(CellValue::CLEARED.to_string(), "!#");
        assert_eq!(CellValue::for_mine_count(3).to_string(), "3#");
    }

    #[test]
    fn test_processed_cell_initialization_and_display() {
        let width = 3;
        let (_, grid) = create_grid_with_border(width, 3);
        let idx = Point { x: 1, y: 1 }.to_index(width + 1) as usize;
        let pc = grid[idx];
        assert_eq!(pc.unknown, E | S | SE);
        assert_eq!(pc.opened, DirectionSet::EMPTY);
        assert_eq!(pc.mines, DirectionSet::EMPTY);

        let compact = format!("{pc}");
        let alternate = format!("{pc:#}");
        assert!(compact.contains("CellNeighbors{ unknown: "));
        assert!(alternate.contains("CellNeighbors{\n    unknown: "));
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

        assert!(CellValue::FLAGGED.is_mine() && CellValue::FLAGGED.is_inferred());
        assert!(CellValue::CLEARED.is_safe() && CellValue::CLEARED.is_inferred());
        // inferred-safe cells carry no count, so they must not look like a Count cell
        assert_eq!(CellValue::CLEARED.safe_mine_count(), None);

        let three = CellValue::for_mine_count(3);
        assert!(three.is_safe() && !three.is_inferred() && !three.is_mine());
        assert_eq!(three.safe_mine_count(), Some(3));
        assert_eq!(three.mine_count(), 3);
    }

    #[test]
    fn test_cell_value_conversions_round_trip() {
        for user_value in [
            UserCellValue::Unknown,
            UserCellValue::Wall,
            UserCellValue::Mine,
            UserCellValue::Opened(0),
            UserCellValue::Opened(4),
            UserCellValue::Opened(8),
        ] {
            let internal: CellValue = user_value.into();
            let with_inference: UserValueWithInference = internal.into();
            assert_eq!(with_inference.value, user_value);
            assert_eq!(with_inference.inferred, InferredValue::None);
            assert_eq!(CellValue::from(with_inference), internal);
        }

        let inferred_mine: UserValueWithInference = CellValue::FLAGGED.into();
        assert_eq!(inferred_mine.value, UserCellValue::Unknown);
        assert_eq!(inferred_mine.inferred, InferredValue::Mine);
        assert_eq!(CellValue::from(inferred_mine), CellValue::FLAGGED);

        let inferred_safe: UserValueWithInference = CellValue::CLEARED.into();
        assert_eq!(inferred_safe.value, UserCellValue::Unknown);
        assert_eq!(inferred_safe.inferred, InferredValue::Safe);
        assert_eq!(CellValue::from(inferred_safe), CellValue::CLEARED);
    }

    #[test]
    fn test_cell_is_invalid_detects_unsatisfiable_counts() {
        // 2x1 strip [Count(2), Unknown]: needs 2 mines but only 1 unknown neighbor -> invalid
        let (_, neighbors) = create_grid_with_border(2, 1);
        let idx = Point { x: 1, y: 1 }.to_index(3) as usize;
        let cell = CellValue::for_mine_count(2);
        assert!(
            cell.is_invalid(neighbors[idx]),
            "needs 2 mines but only 1 unknown neighbor"
        );

        let (_, mut neighbors) = create_grid_with_border(2, 1);
        neighbors[idx].unknown = !!E;
        let cell = CellValue::for_mine_count(1);
        assert!(!cell.is_invalid(neighbors[idx]));
    }

    #[test]
    fn test_cell_is_invalid_handles_excess_mine_neighbors() {
        // a Count(0) next to a known mine is over-satisfied
        let (_, mut neighbors) = create_grid_with_border(2, 1);
        let idx = Point { x: 1, y: 1 }.to_index(3) as usize;
        neighbors[idx].mines |= E;
        let cell = CellValue::for_mine_count(0);
        assert!(cell.is_invalid(neighbors[idx]));
    }

    #[test]
    fn test_ordered_cell_from_cell_and_ordering() {
        let (_, grid) = create_grid_with_border(3, 3);
        let neighbors = grid[0];
        assert!(CellOrdering::from_cell(CellIdx(0), CellValue::UNKNOWN, neighbors).is_none());

        assert!(CellOrdering::from_cell(CellIdx(0), CellValue::MINE, neighbors).is_none());

        // Trivial cell A: 2 available, 2 remaining -> choose(2, 2) = 1 solution
        let mut neighbors_a = grid[0];
        let cell_a = CellValue::for_mine_count(2);
        neighbors_a.unknown = E | S;

        // Trivial cell B: 5 available, 0 remaining (1 mine already found) -> choose(5, 0) = 1 solution, lower mine_count
        let mut neighbor_b = grid[1];
        let cell_b = CellValue::for_mine_count(1);
        neighbor_b.unknown = neighbor_b.all() - S;
        neighbor_b.mines = !!S;

        // Non-trivial cell C: 3 available, 1 remaining -> choose(3, 1) = 3 solutions
        let mut neighbors_c = grid[2];
        let cell_c = CellValue::for_mine_count(1);
        neighbors_c.unknown = W | SW | S;

        let ord_a = CellOrdering::from_cell(CellIdx(0), cell_a, neighbors_a).unwrap();
        let ord_b = CellOrdering::from_cell(CellIdx(1), cell_b, neighbor_b).unwrap();
        let ord_c = CellOrdering::from_cell(CellIdx(2), cell_c, neighbors_c).unwrap();

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
        let neighbors: Vec<_> = CellIdx(4).neighbor_iter(3, E | S | SE).collect();
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
        assert_eq!(CellValue::WALL.cell_type(), CellType::Wall);
        assert_eq!(CellValue::MINE.cell_type(), CellType::Mine);
        assert_eq!(CellValue::FLAGGED.cell_type(), CellType::Flag);
        assert_eq!(CellValue::CLEARED.cell_type(), CellType::Cleared);

        for count in 0..=8 {
            assert_eq!(
                CellValue::for_mine_count(count).cell_type(),
                CellType::Opened,
                "Count {count} must map to CellType::Open"
            );
        }
    }

    #[test]
    fn test_cell_value_4bit_encoding() {
        for count in 0..=8 {
            let val = CellValue::for_mine_count(count);
            assert!(val.is_opened());
            assert!(val.is_safe());
            assert!(!val.is_mine());
            assert!(!val.is_inferred());
            assert!(!val.is_unknown());
            assert_eq!(val.cell_type(), CellType::Opened);
            assert_eq!(val.safe_mine_count(), Some(count));
            assert_eq!(val.mine_count(), count);
        }

        let inf_safe = CellValue::CLEARED;
        assert!(!inf_safe.is_opened());
        assert!(inf_safe.is_safe());
        assert!(!inf_safe.is_mine());
        assert!(inf_safe.is_inferred());
        assert!(!inf_safe.is_unknown());
        assert_eq!(inf_safe.cell_type(), CellType::Cleared);
        assert_eq!(inf_safe.safe_mine_count(), None);

        let unk = CellValue::UNKNOWN;
        assert!(!unk.is_opened());
        assert!(!unk.is_safe());
        assert!(!unk.is_mine());
        assert!(!unk.is_inferred());
        assert!(unk.is_unknown());
        assert_eq!(unk.cell_type(), CellType::Unknown);
        assert_eq!(unk.safe_mine_count(), None);

        let wall = CellValue::WALL;
        assert!(!wall.is_opened());
        assert!(!wall.is_safe());
        assert!(!wall.is_mine());
        assert!(!wall.is_inferred());
        assert!(!wall.is_unknown());
        assert!(wall.is_wall());
        assert_eq!(wall.cell_type(), CellType::Wall);
        assert_eq!(wall.safe_mine_count(), None);

        let mine = CellValue::MINE;
        assert!(!mine.is_opened());
        assert!(!mine.is_safe());
        assert!(mine.is_mine());
        assert!(!mine.is_inferred());
        assert!(!mine.is_unknown());
        assert_eq!(mine.cell_type(), CellType::Mine);
        assert_eq!(mine.safe_mine_count(), None);

        let inf_mine = CellValue::FLAGGED;
        assert!(!inf_mine.is_opened());
        assert!(!inf_mine.is_safe());
        assert!(inf_mine.is_mine());
        assert!(inf_mine.is_inferred());
        assert!(!inf_mine.is_unknown());
        assert_eq!(inf_mine.cell_type(), CellType::Flag);
        assert_eq!(inf_mine.safe_mine_count(), None);
    }

    #[test]
    fn test_guarded_queue_push_pop_remove_dedup() {
        let mut flags = vec![CellFlags::new(); 4].into_boxed_slice();
        let mut queue = GuardedQueue::with_capacity(4, FlagMask::PENDING_PARTITION);
        assert!(queue.is_empty());

        // First push succeeds
        queue.push(&mut flags, CellIdx(1));
        assert!(!queue.is_empty());
        assert!(flags[1].is_marked(FlagMask::PENDING_PARTITION));

        // Duplicate push of the same index is ignored
        queue.push(&mut flags, CellIdx(1));
        let items: Vec<_> = queue.iter().collect();
        assert_eq!(items, vec![CellIdx(1)]);

        // Push another index
        queue.push(&mut flags, CellIdx(3));
        assert_eq!(
            queue.iter().collect::<Vec<_>>(),
            vec![CellIdx(1), CellIdx(3)]
        );
    }
}

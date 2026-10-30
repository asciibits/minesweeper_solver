//! Cell data structures representing both external board state ([`Cell`], [`CellValue`])
//! and internal solver state ([`ProcessedCell`]).

use std::fmt::{self, Display, Formatter};

use crate::solver::grid::{Direction, DirectionSet, Point};

/// Provide the x/y coordinates of a Cell with the given cell_id in a fixed
/// width field. This is the corollary to `cell_id`.
pub fn coordinates(cell_idx: usize, width: u32) -> Point {
    Point::from_index(cell_idx, width)
}

/// Get a unique ID for the cell in a given-width field. Useful for indexing in
/// a Vector or Array. This is the corollary to `coordinates`.
pub fn cell_idx(p: Point, width: u32) -> usize {
    p.to_index(width)
}

/// A single cell on the Minesweeper board, pairing its [`Point`] coordinates with its [`CellValue`].
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct UserCell {
    /// Grid coordinates of the cell.
    pub point: Point,
    /// Current state/value of the cell.
    pub value: UserCellValue,
}

impl UserCell {
    /// Creates a new [`Cell`] at `point` initialized to [`CellValue::Unknown`].
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
    Safe(i8),
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

#[derive(Debug, Default, Hash, Clone, Copy, PartialEq, Eq)]
pub(super) struct CellValue(u8);

impl CellValue {
    const SAFE_MASK: u8 = 0b1000_0000;
    const MINE_MASK: u8 = 0b0100_0000;
    const KNOWN_MASK: u8 = Self::SAFE_MASK | Self::MINE_MASK;
    const INFERRED_MASK: u8 = 0b0010_0000;

    pub(super) const UNKNOWN: Self = Self(0);
    pub(super) const MINE: Self = Self(Self::MINE_MASK);
    pub(super) const INFERRED_MINE: Self = Self(Self::MINE_MASK | Self::INFERRED_MASK);
    pub(super) const INFERRED_SAFE: Self = Self(Self::SAFE_MASK | Self::INFERRED_MASK);

    pub(super) fn for_mine_count(mine_count: i8) -> CellValue {
        assert!(mine_count > 0 && mine_count <= 8);
        CellValue(Self::SAFE_MASK | mine_count as u8)
    }

    pub(super) fn set_value(&mut self, other: CellValue) -> CellValueChangeType {
        if *self == other {
            return CellValueChangeType::NoChange;
        }
        *self = other;
        if self.is_unknown() || (self.0 ^ other.0) & Self::KNOWN_MASK == 0 {
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

    pub(super) fn mine_count(&self) -> Option<i8> {
        // NOTE: this is not the same as `.is_safe()`. This will not return
        // `true` for inferred values
        (self.0 & 0xf0 == Self::SAFE_MASK).then_some((self.0 & 0xf) as i8)
    }

    // Returns if this value and `other` are alternating types as occur during
    // partitioning. This means one is safe (inferred or not), while the other
    // is unknown (i.e. not inferred and not known).
    pub(super) fn is_alt(&self, other: Self) -> bool {
        (self.0 ^ other.0) & Self::KNOWN_MASK == Self::MINE_MASK
    }

    pub(super) fn to_user_value(self) -> UserCellValue {
        let user_value: (UserCellValue, InferredValue) = self.into();
        user_value.0
    }

    pub(super) fn to_inferred_value(self) -> InferredValue {
        let user_value: (UserCellValue, InferredValue) = self.into();
        user_value.1
    }
}

impl From<UserCellValue> for CellValue {
    fn from(value: UserCellValue) -> Self {
        match value {
            UserCellValue::Unknown => Self::UNKNOWN,
            UserCellValue::Mine => Self::MINE,
            UserCellValue::Safe(c) => Self(Self::SAFE_MASK | c as u8),
        }
    }
}

impl From<(UserCellValue, InferredValue)> for CellValue {
    fn from(value: (UserCellValue, InferredValue)) -> Self {
        match value {
            (UserCellValue::Mine, InferredValue::None) => CellValue::MINE,
            (UserCellValue::Unknown, InferredValue::Mine) => CellValue::INFERRED_MINE,
            (UserCellValue::Unknown, InferredValue::Safe) => CellValue::INFERRED_SAFE,
            (UserCellValue::Unknown, InferredValue::None) => CellValue::UNKNOWN,
            (UserCellValue::Safe(c), _) => CellValue::for_mine_count(c),
            _ => unreachable!(),
        }
    }
}

impl From<CellValue> for (UserCellValue, InferredValue) {
    fn from(value: CellValue) -> Self {
        match value {
            CellValue::MINE => (UserCellValue::Mine, InferredValue::None),
            CellValue::INFERRED_MINE => (UserCellValue::Unknown, InferredValue::Mine),
            CellValue::INFERRED_SAFE => (UserCellValue::Unknown, InferredValue::Safe),
            CellValue::UNKNOWN => (UserCellValue::Unknown, InferredValue::None),
            c => (UserCellValue::Safe((c.0 & 0xf) as i8), InferredValue::None),
        }
    }
}

impl Display for CellValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UNKNOWN => write!(f, "??"),
            Self::MINE => write!(f, "**"),
            Self::INFERRED_MINE => write!(f, ":*"),
            Self::INFERRED_SAFE => write!(f, ":#"),
            CellValue(c) => write!(f, "{}#", c & 0xf),
        }
    }
}

#[derive(Debug, Default, Hash, Clone, Copy, PartialEq, Eq)]
pub(super) enum CellValueChangeType {
    #[default]
    NoChange,
    NonBreaking,
    Breaking,
}

impl CellValueChangeType {
    pub(super) fn is_change(&self) -> bool {
        !matches!(self, Self::NoChange)
    }
}

/// Internal solver representation of a board cell, caching neighbor masks and partition membership.
// Note: Try to keep this under 32 bytes to fit in a single cache line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Cell {
    /// The id - calculated as `cell.x + cell.y * field.width`.
    idx: usize,

    /// Inline the contents of Cell - this removes some internal padding and
    /// keeps ProcessedCell within the 32 byte boundary
    value: CellValue,

    // ========== Cached neighbor data ==========
    // The following fields are derived from the cell's coordinates and the
    // board dimensions. They do not change on cell value changes.
    //
    //
    /// The "all neighbors" mask (taking board edges into account).
    neighbors: DirectionSet,

    // ========== Cached neighbor data (inferred from cell values) ==========
    // The following fields are derived from the cell's neighbors' values. They are updated
    // whenever a neighbor's value changes, and are used to determine the cell's
    // inferred state. Note: these fields are a disjoint set the collectively
    // cover all neighbors of the cell.
    //

    // ========== Solver state data ==========
    //
    //
    /// The value that is staged to be applied to the cell. Used prior
    /// to a `solve`.
    staged_value: Option<UserCellValue>,

    /// The inferred state of the cell.
    inferred_value: InferredValue,

    /// The remaining neighbors left after [`safe_neighbors`] and
    /// [`mine_neighbors`].
    unknown_neighbors: DirectionSet,

    /// [`safe_neighbors`] unioned with unknown neighbors that have their
    /// inferred_state set to Safe.
    safe_neighbors: DirectionSet,

    /// [`mine_neighbors`] unioned with unknown neighbors that have their
    /// inferred_state set to Mine.
    mine_neighbors: DirectionSet,

    // Which directions are inferred
    inferred_neighbors: DirectionSet,

    /// The partition this cell belongs to (`0` if unpartitioned).
    partition_id: u32,
}

impl Cell {
    pub(super) fn grid(width: u32, height: u32) -> Box<[Cell]> {
        let len = width as usize * height as usize;
        let mut grid: Vec<Cell> = Vec::with_capacity(len);

        let mut idx = 0;
        for y in 0..height {
            for x in 0..width {
                let point = Point { x, y };
                let neighbors = DirectionSet::all_neighbors(point, width, height);
                grid.push(Cell {
                    idx,
                    value: CellValue::UNKNOWN,
                    inferred_value: InferredValue::None,
                    staged_value: None,
                    neighbors,
                    unknown_neighbors: neighbors,
                    safe_neighbors: DirectionSet::new(),
                    mine_neighbors: DirectionSet::new(),
                    inferred_neighbors: DirectionSet::new(),
                    partition_id: 0,
                });

                idx += 1;
            }
        }

        grid.into_boxed_slice()
    }

    /// Returns the cell's grid coordinates.
    #[inline]
    pub(super) fn point(&self, width: u32) -> Point {
        coordinates(self.idx, width)
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
    pub(super) fn set_value(&mut self, value: CellValue) {
        self.value = value;
    }

    /// Returns the cell's row-major linear index.
    #[inline]
    pub(super) fn idx(&self) -> usize {
        self.idx
    }

    /// Constructs a [`Cell`] from this cell's coordinates and current value.
    #[inline]
    pub(super) fn cell(&self, width: u32) -> UserCell {
        UserCell {
            point: coordinates(self.idx, width),
            value: self.value.to_user_value(),
        }
    }

    /// Returns the mask of all valid board neighbors for this cell.
    #[inline]
    pub(super) fn neighbors(&self) -> DirectionSet {
        self.neighbors
    }

    /// Returns the mask of neighbors currently holding [`CellValue::Unknown`].
    #[inline]
    #[allow(dead_code)]
    pub(super) fn unknown_neighbors(&self) -> DirectionSet {
        self.unknown_neighbors + self.inferred_neighbors
    }

    /// Returns the mask of neighbors currently holding [`CellValue::Count`].
    #[inline]
    #[allow(dead_code)]
    pub(super) fn safe_neighbors(&self) -> DirectionSet {
        self.safe_neighbors - self.inferred_neighbors
    }

    /// Returns the mask of neighbors currently holding [`CellValue::Mine`].
    #[inline]
    #[allow(dead_code)]
    pub(super) fn mine_neighbors(&self) -> DirectionSet {
        self.mine_neighbors - self.inferred_neighbors
    }

    /// Returns the value staged to be applied on the next solver pass, if any.
    #[inline]
    pub(super) fn staged_value(&self) -> Option<UserCellValue> {
        self.staged_value
    }

    /// Sets or clears the staged value for the next solver pass.
    #[inline]
    pub(super) fn set_staged_value(&mut self, staged_value: Option<UserCellValue>) {
        self.staged_value = staged_value;
    }

    /// Returns the solver's current [`InferredState`] for this cell.
    #[inline]
    pub(super) fn inferred_value(&self) -> InferredValue {
        self.inferred_value
    }

    /// Updates the solver's [`InferredState`] for this cell.
    #[inline]
    pub(super) fn set_inferred_value(&mut self, inferred_state: InferredValue) {
        self.inferred_value = inferred_state;
    }

    /// Returns the mask of neighbors still unknown after applying solver inferences.
    #[inline]
    pub(super) fn inferred_unknown_neighbors(&self) -> DirectionSet {
        self.unknown_neighbors
    }

    /// Returns the mask of neighbors known or inferred to be safe.
    #[inline]
    #[allow(dead_code)]
    pub(super) fn inferred_safe_neighbors(&self) -> DirectionSet {
        self.safe_neighbors
    }

    /// Returns the mask of neighbors known or inferred to be mines.
    #[inline]
    pub(super) fn inferred_mine_neighbors(&self) -> DirectionSet {
        self.mine_neighbors
    }

    /// Returns the ID of the partition this cell belongs to (`0` if unpartitioned).
    #[inline]
    pub(super) fn partition_id(&self) -> u32 {
        self.partition_id
    }

    /// Updates the ID of the partition this cell belongs to.
    #[inline]
    pub(super) fn set_partition_id(&mut self, partition_id: u32) {
        self.partition_id = partition_id;
    }

    pub(super) fn alt_neighbors(&self, ignore_inferred: bool) -> DirectionSet {
        match self.value.0 >> 6 {
            0 => {
                // if ignore_inferred || self.inferred_value == InferredValue::None {
                //     self.safe_neighbors
                // } else {
                //     DirectionSet::new()
                // }

                // TODO : Why does the  above work? And the below fail?

                if ignore_inferred {
                    self.safe_neighbors()
                } else {
                    match self.inferred_value {
                        // InferredValue::Safe => self.inferred_unknown_neighbors,
                        InferredValue::Safe => DirectionSet::new(),
                        InferredValue::Mine => DirectionSet::new(),
                        InferredValue::None => self.safe_neighbors(),
                    }
                }
            }
            1 => DirectionSet::new(),
            2 => {
                if ignore_inferred {
                    self.unknown_neighbors()
                } else {
                    self.inferred_unknown_neighbors()
                }
            }
            _ => unreachable!(),
        }
    }

    pub(super) fn is_mine(&self, ignore_inferred: bool) -> bool {
        self.value.is_mine() && !(ignore_inferred && self.value.is_inferred())
    }

    pub(super) fn is_safe(&self, ignore_inferred: bool) -> bool {
        self.value.is_safe() && !(ignore_inferred && self.value.is_inferred())
    }

    pub(super) fn set_unknown(&mut self, direction: Direction) {
        self.unknown_neighbors += direction;
        self.safe_neighbors -= direction;
        self.mine_neighbors -= direction;
        // unknowns don't use the inferred flag
    }

    pub(super) fn set_safe(&mut self, direction: Direction, inferred: bool) {
        self.unknown_neighbors -= direction;
        self.safe_neighbors += direction;
        self.mine_neighbors -= direction;
        if inferred {
            self.inferred_neighbors += direction;
        } else {
            self.inferred_neighbors -= direction;
        }
    }

    pub(super) fn set_mine(&mut self, direction: Direction, inferred: bool) {
        self.unknown_neighbors -= direction;
        self.safe_neighbors -= direction;
        self.mine_neighbors += direction;
        if inferred {
            self.inferred_neighbors += direction;
        } else {
            self.inferred_neighbors -= direction;
        }
    }
}

impl Display for Cell {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let spacer = if f.alternate() { "\n  " } else { " " };
        write!(
            f,
            "Cell{{{spacer}idx:{},{spacer}value:{},{spacer}neighbors:{},{spacer}unknown_neighbors:{},{spacer}safe_neighbors:{},{spacer}mine_neighbors:{},{spacer}inferred:{},{spacer}staged_value:{:?},{spacer}inferred_state:{:?},{spacer}partition_id:{}{spacer}}}",
            self.idx,
            self.value,
            self.neighbors,
            self.unknown_neighbors,
            self.safe_neighbors,
            self.mine_neighbors,
            self.inferred_neighbors,
            self.staged_value,
            self.inferred_value,
            self.partition_id
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InferredValue {
    None,
    Safe,
    Mine,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::grid::{E, N, NE, NW, S, SE, W};

    #[test]
    fn test_coordinates_and_cell_idx_round_trip() {
        let width = 7u32;
        for idx in 0..35usize {
            let p = coordinates(idx, width);
            assert_eq!(cell_idx(p, width), idx);
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
        assert_eq!(pc.idx, 0);
        assert_eq!(pc.value, CellValue::UNKNOWN);
        assert_eq!(pc.neighbors, E | S | SE);
        assert_eq!(pc.safe_neighbors, DirectionSet::new());
        assert_eq!(pc.mine_neighbors, DirectionSet::new());
        assert_eq!(pc.unknown_neighbors, E | S | SE);
        assert_eq!(pc.staged_value, None);
        assert_eq!(pc.inferred_neighbors, DirectionSet::new());
        assert_eq!(pc.inferred_value, InferredValue::None);
        assert_eq!(pc.partition_id, 0);

        pc.value = CellValue::for_mine_count(2);

        let compact = dbg!(format!("{pc}"));
        let alternate = dbg!(format!("{pc:#}"));
        assert!(compact.contains("Cell{ idx:0, value:2"));
        assert!(alternate.contains("Cell{\n  idx:0,\n  value:2"));
    }

    #[test]
    fn test_processed_cell_alt_neighbors_and_partition_state_reset() {
        let grid = Cell::grid(3, 3);
        let mut pc = grid[0];
        pc.neighbors -= NW | N | NE | W;
        pc.set_safe(S, false);
        pc.set_mine(SE, false);

        // Unknown cells return safe_neighbors from alt_neighbors()
        pc.value = CellValue::UNKNOWN;
        assert_eq!(pc.alt_neighbors(false), !!S);

        // Count cells return unknown_neighbors from alt_neighbors()
        pc.value = CellValue::for_mine_count(1);
        assert_eq!(pc.alt_neighbors(false), !!E);

        // Mine cells return an empty set from alt_neighbors()
        pc.value = CellValue::MINE;
        assert_eq!(pc.alt_neighbors(false), DirectionSet::new());
    }
}

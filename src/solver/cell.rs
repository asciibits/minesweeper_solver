//! Cell data structures representing both external board state ([`Cell`], [`CellValue`])
//! and internal solver state ([`ProcessedCell`]).

use std::fmt::{self, Display, Formatter};

use crate::solver::grid::{Direction, DirectionSet, Point};

/// Provide the x/y coordinates of a Cell with the given cell_id in a fixed
/// width field. This is the corollary to `cell_id`.
pub fn coordinates(cell_idx: usize, width: usize) -> Point {
    Point::from_index(cell_idx, width as u32)
}

/// Get a unique ID for the cell in a given-width field. Useful for indexing in
/// a Vector or Array. This is the corollary to `coordinates`.
pub fn cell_idx(p: Point, width: u32) -> usize {
    p.to_index(width)
}

/// A single cell on the Minesweeper board, pairing its [`Point`] coordinates with its [`CellValue`].
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Cell {
    /// Grid coordinates of the cell.
    pub point: Point,
    /// Current state/value of the cell.
    pub value: CellValue,
}

impl Cell {
    /// Creates a new [`Cell`] at `point` initialized to [`CellValue::Unknown`].
    pub fn new(point: Point) -> Cell {
        Cell {
            point,
            value: CellValue::Unknown,
        }
    }

    /// Creates a new [`Cell`] from a row-major linear `cell_idx` in a board of the given `width`.
    pub fn from_index(cell_idx: usize, width: usize) -> Cell {
        Cell::new(Point::from_index(cell_idx, width as u32))
    }

    /// Returns the row-major linear index of this cell in a board of the given `width`.
    pub fn idx(&self, width: u32) -> usize {
        self.point.to_index(width)
    }
}

impl Display for Cell {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.point, self.value)
    }
}

/// The state of a single cell on the Minesweeper board.
#[derive(Debug, Default, Hash, Clone, Copy, PartialEq, Eq)]
pub enum CellValue {
    /// The cell has not been revealed or flagged.
    #[default]
    Unknown,
    /// The cell contains a mine.
    Mine,
    /// The cell is revealed and has the given number of adjacent mines (`0..=8`).
    Count(i8),
}

impl CellValue {
    /// Returns `true` if the cell is anything other than [`CellValue::Unknown`].
    pub fn is_known(&self) -> bool {
        *self != CellValue::Unknown
    }
}

impl Display for CellValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            CellValue::Unknown => write!(f, "Unknown"),
            CellValue::Mine => write!(f, "MINE"),
            CellValue::Count(c) => write!(f, "{c}"),
        }
    }
}

/// Internal solver representation of a board cell, caching neighbor masks and partition membership.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// Note: Try to keep this under 32 bytes to fit in a single cache line.
pub(super) struct ProcessedCell {
    /// Inline the contents of Cell - this removes some internal padding and
    /// keeps ProcessedCell within the 32 byte boundary
    pub(super) point: Point,
    pub(super) value: CellValue,

    // ========== Cached neighbor data ==========
    // The following fields are derived from the cell's coordinates and the
    // board dimensions. They do not change on cell value changes.
    //
    //
    /// The id - calculated as `cell.x + cell.y * field.width`.
    pub(super) idx: usize,

    /// The "all neighbors" mask (taking board edges into account).
    pub(super) neighbors: DirectionSet,

    // ========== Cached neighbor data (inferred from cell values) ==========
    // The following fields are derived from the cell's neighbors' values. They are updated
    // whenever a neighbor's value changes, and are used to determine the cell's
    // inferred state. Note: these fields are a disjoint set the collectively
    // cover all neighbors of the cell.
    //
    //
    /// The neighbors that are have a CellValue of type Unknown.
    pub(super) unknown_neighbors: DirectionSet,

    /// The neighbors that are have a CellValue of type Count.
    pub(super) open_neighbors: DirectionSet,

    /// The neighbors that are have a CellValue of type Mine.
    pub(super) mine_neighbors: DirectionSet,

    // ========== Solver state data ==========
    //
    //
    /// The value that is staged to be applied to the cell. Used prior
    /// to a `solve`.
    pub(super) staged_value: Option<CellValue>,

    /// The inferred state of the cell.
    pub(super) inferred_state: InferredState,

    /// The remaining neighbors left after [`inferred_open_neighbors`] and
    /// [`inferred_mine_neighbors`].
    pub(super) inferred_unknown_neighbors: DirectionSet,

    /// [`open_neighbors`] unioned with unknown neighbors that have their
    /// inferred_state set to Open.
    pub(super) inferred_open_neighbors: DirectionSet,

    /// [`mine_neighbors`] unioned with unknown neighbors that have their
    /// inferred_state set to Mine.
    pub(super) inferred_mine_neighbors: DirectionSet,

    /// The partition this cell belongs to (`0` if unpartitioned).
    pub(super) partition_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InferredState {
    None,
    Open,
    Mine,
}

impl ProcessedCell {
    /// Initializes a [`ProcessedCell`] at `p` for a board of dimensions `width` by `height`.
    pub(super) fn new(point: Point, width: u32, height: u32) -> ProcessedCell {
        let idx = point.to_index(width);
        let neighbors = DirectionSet::all_neighbors(point, width, height);
        ProcessedCell {
            point,
            value: CellValue::Unknown,
            idx,
            neighbors,
            unknown_neighbors: neighbors,
            open_neighbors: DirectionSet::new(),
            mine_neighbors: DirectionSet::new(),
            staged_value: None,
            inferred_state: InferredState::None,
            inferred_unknown_neighbors: neighbors,
            inferred_open_neighbors: DirectionSet::new(),
            inferred_mine_neighbors: DirectionSet::new(),
            partition_id: 0,
        }
    }

    /// Returns the cell's current [`CellValue`].
    pub(super) fn cell(&self) -> Cell {
        Cell {
            point: self.point,
            value: self.value,
        }
    }

    pub(super) fn alt_inferred_neighbors(&self) -> DirectionSet {
        match self.value {
            CellValue::Unknown => self.inferred_open_neighbors,
            CellValue::Count(_) => self.inferred_unknown_neighbors,
            CellValue::Mine => DirectionSet::new(),
        }
    }

    pub(super) fn clear_inferred(&mut self) {
        self.inferred_state = InferredState::None;
        self.inferred_unknown_neighbors = self.unknown_neighbors;
        self.inferred_open_neighbors = self.open_neighbors;
        self.inferred_mine_neighbors = self.mine_neighbors;
    }

    pub(super) fn is_mine(&self) -> bool {
        self.value == CellValue::Mine || self.inferred_state == InferredState::Mine
    }

    pub(super) fn is_open(&self) -> bool {
        matches!(self.value, CellValue::Count(_)) || self.inferred_state == InferredState::Open
    }

    pub(super) fn add_unknown_neighbor(&mut self, direction: Direction) {
        self.unknown_neighbors.insert(direction);
        self.inferred_unknown_neighbors.insert(direction);
    }

    pub(super) fn remove_unknown_neighbor(&mut self, direction: Direction) {
        self.unknown_neighbors.remove(direction);
        self.inferred_unknown_neighbors.remove(direction);
    }

    pub(super) fn add_open_neighbor(&mut self, direction: Direction) {
        self.open_neighbors.insert(direction);
        self.inferred_open_neighbors.insert(direction);
    }

    pub(super) fn remove_open_neighbor(&mut self, direction: Direction) {
        self.open_neighbors.remove(direction);
        self.inferred_open_neighbors.remove(direction);
    }

    pub(super) fn add_mine_neighbor(&mut self, direction: Direction) {
        self.mine_neighbors.insert(direction);
        self.inferred_mine_neighbors.insert(direction);
    }

    pub(super) fn remove_mine_neighbor(&mut self, direction: Direction) {
        self.mine_neighbors.remove(direction);
        self.inferred_mine_neighbors.remove(direction);
    }
}

impl Display for ProcessedCell {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let alt_formatting = if f.alternate() { "\n  " } else { "" };
        write!(
            f,
            "ProcessedCell{{{alt_formatting}cell: {}, {alt_formatting}idx: {}, {alt_formatting}neighbors: {}, {alt_formatting}open_neighbors: {}, {alt_formatting}mine_neighbors: {}, {alt_formatting}unknown_neighbors: {}, {alt_formatting}staged_value: {:?}, {alt_formatting}inferred_open_neighbors: {}, {alt_formatting}inferred_mine_neighbors: {}, {alt_formatting}inferred_unknown_neighbors: {}, {alt_formatting}inferred_state: {:?}, {alt_formatting}partition_id: {}{alt_formatting}}}",
            self.cell(),
            self.idx,
            self.neighbors,
            self.open_neighbors,
            self.mine_neighbors,
            self.unknown_neighbors,
            self.staged_value,
            self.inferred_open_neighbors,
            self.inferred_mine_neighbors,
            self.inferred_unknown_neighbors,
            self.inferred_state,
            self.partition_id
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::grid::{E, S, SE};

    #[test]
    fn test_coordinates_and_cell_idx_round_trip() {
        let width = 7u32;
        for idx in 0..35usize {
            let p = coordinates(idx, width as usize);
            assert_eq!(cell_idx(p, width), idx);
        }
    }

    #[test]
    fn test_cell_creation_and_display() {
        let c = Cell::from_index(12, 5);
        assert_eq!(c.point, Point { x: 2, y: 2 });
        assert_eq!(c.value, CellValue::Unknown);
        assert_eq!(c.idx(5), 12);
        assert_eq!(c.to_string(), "(2, 2): Unknown");
    }

    #[test]
    fn test_cell_value_is_known_and_display() {
        assert_eq!(CellValue::default(), CellValue::Unknown);
        assert!(!CellValue::Unknown.is_known());
        assert!(CellValue::Mine.is_known());
        assert!(CellValue::Count(0).is_known());
        assert!(CellValue::Count(3).is_known());

        assert_eq!(CellValue::Unknown.to_string(), "Unknown");
        assert_eq!(CellValue::Mine.to_string(), "MINE");
        assert_eq!(CellValue::Count(4).to_string(), "4");
    }

    #[test]
    fn test_processed_cell_initialization_and_display() {
        let mut pc = ProcessedCell::new(Point { x: 0, y: 0 }, 3, 3);
        assert_eq!(pc.idx, 0);
        assert_eq!(pc.value, CellValue::Unknown);
        assert_eq!(pc.neighbors, [E, S, SE].into_iter().collect());
        assert_eq!(pc.open_neighbors, DirectionSet::new());
        assert_eq!(pc.mine_neighbors, DirectionSet::new());
        assert_eq!(pc.unknown_neighbors, [E, S, SE].into_iter().collect());
        assert_eq!(pc.staged_value, None);
        assert_eq!(pc.inferred_open_neighbors, DirectionSet::new());
        assert_eq!(pc.inferred_mine_neighbors, DirectionSet::new());
        assert_eq!(
            pc.inferred_unknown_neighbors,
            [E, S, SE].into_iter().collect()
        );
        assert_eq!(pc.inferred_state, InferredState::None);
        assert_eq!(pc.partition_id, 0);

        pc.value = CellValue::Count(2);

        let compact = format!("{pc}");
        let alternate = format!("{pc:#}");
        assert!(compact.contains("ProcessedCell{cell: (0, 0): 2"));
        assert!(alternate.contains("ProcessedCell{\n  cell: (0, 0): 2"));
    }

    #[test]
    fn test_processed_cell_alt_neighbors_and_partition_state_reset() {
        let mut pc = ProcessedCell::new(Point { x: 0, y: 0 }, 3, 3);
        pc.add_open_neighbor(S);
        pc.remove_unknown_neighbor(S);
        pc.add_mine_neighbor(SE);
        pc.remove_unknown_neighbor(SE);

        // Unknown cells return open_neighbors from alt_neighbors()
        pc.value = CellValue::Unknown;
        assert_eq!(pc.alt_inferred_neighbors(), [S].into_iter().collect());

        // Count cells return unknown_neighbors from alt_neighbors()
        pc.value = CellValue::Count(1);
        assert_eq!(pc.alt_inferred_neighbors(), [E].into_iter().collect());

        // Mine cells return an empty set from alt_neighbors()
        pc.value = CellValue::Mine;
        assert_eq!(pc.alt_inferred_neighbors(), DirectionSet::new());
    }
}

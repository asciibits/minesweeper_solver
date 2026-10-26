//! Cell data structures representing both external board state ([`Cell`], [`CellValue`])
//! and internal solver state ([`ProcessedCell`]).

use std::fmt::{self, Display, Formatter};

use crate::solver::grid::{DirectionSet, Point};

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
pub(super) struct ProcessedCell {
    /// `cell` contains the provided raw data.
    pub(super) cell: Cell,

    /// The id - calculated as `cell.x + cell.y * field.width`.
    pub(super) idx: usize,

    /// The "all neighbors" mask (taking board edges into account).
    pub(super) neighbors: DirectionSet,

    /// The value that is staged to be applied to the cell. Used prior
    /// to a `solve`.
    pub(super) staged_value: Option<CellValue>,

    /// The neighbors of different type (unknown cells have known neighbors and vice versa).
    pub(super) alt_neighbors: DirectionSet,

    /// This is the set of all neighbors that are unknown and not inferred. It is
    /// `alt_neighbors` with the inferred opens and mines removed.
    pub(super) remaining_neighbors: DirectionSet,

    /// The number of mines needed to place into `remaining_neighbors`. For
    /// unknown cells, this is a negative value indicating how many mines are
    /// already placed in the neighbors. This allows simpler updates when a
    /// neighbor is revealed to be a mine or opened, as we can just
    /// increment/decrement this value.
    pub(super) remaining_mines: i8,

    /// The partition this cell belongs to (`0` if unpartitioned).
    pub(super) partition_id: usize,
}

impl ProcessedCell {
    /// Initializes a [`ProcessedCell`] at `p` for a board of dimensions `width` by `height`.
    pub(super) fn new(p: Point, width: u32, height: u32) -> ProcessedCell {
        let cell = Cell::new(p);
        let idx = p.to_index(width);
        let neighbors = DirectionSet::all_neighbors(p, width, height);
        ProcessedCell {
            cell,
            idx,
            neighbors,
            staged_value: None,
            alt_neighbors: DirectionSet::new(),
            remaining_neighbors: DirectionSet::new(),
            remaining_mines: 0,
            partition_id: 0,
        }
    }

    /// Returns the cell's current [`CellValue`].
    pub(super) fn value(&self) -> CellValue {
        self.cell.value
    }

    /// Updates the cell's [`CellValue`].
    pub(super) fn set_value(&mut self, value: CellValue) {
        self.cell.value = value;
    }
}

impl Display for ProcessedCell {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let alt_formatting = if f.alternate() { "\n  " } else { "" };
        write!(
            f,
            "ProcessedCell{{{alt_formatting}cell: {}, {alt_formatting}idx: {}, {alt_formatting}neighbors: {}, {alt_formatting}alt_neighbors: {}, {alt_formatting}remaining_mines: {}}}",
            self.cell, self.idx, self.neighbors, self.alt_neighbors, self.remaining_mines
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
        assert_eq!(pc.value(), CellValue::Unknown);
        assert_eq!(pc.neighbors, [E, S, SE].into_iter().collect());
        assert!(pc.alt_neighbors.is_empty());
        assert!(pc.remaining_neighbors.is_empty());
        assert_eq!(pc.remaining_mines, 0);
        assert_eq!(pc.partition_id, 0);

        pc.set_value(CellValue::Count(2));
        assert_eq!(pc.value(), CellValue::Count(2));

        let compact = format!("{pc}");
        let alternate = format!("{pc:#}");
        assert!(compact.contains("ProcessedCell{cell: (0, 0): 2"));
        assert!(alternate.contains("ProcessedCell{\n  cell: (0, 0): 2"));
    }
}

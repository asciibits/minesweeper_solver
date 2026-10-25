use std::fmt::{self, Display, Formatter};

use crate::solver::direction::Point;

/// Provide the x/y coordinates of a Cell with the given cell_id in a fixed
/// width field. This is the correlary to `cell_id`.
pub fn coordinates(cell_idx: usize, width: usize) -> Point {
    let y = cell_idx / width;
    let x = (cell_idx - y * width) as u32;
    let y = y as u32;
    Point { x, y }
}

/// Get a unique ID for the cell in a given-width field. Useful for indexing in
/// a Vector or Array. This is the correlary to `coordinates`.
pub fn cell_idx(p: Point, width: u32) -> usize {
    p.x as usize + p.y as usize * width as usize
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Cell {
    pub point: Point,
    pub value: CellValue,
}

impl Cell {
    pub fn new(point: Point) -> Cell {
        Cell {
            point,
            value: CellValue::Unknown,
        }
    }

    pub fn from_index(cell_idx: usize, width: usize) -> Cell {
        Cell::new(coordinates(cell_idx, width))
    }

    pub fn idx(&self, width: u32) -> usize {
        cell_idx(self.point, width)
    }
}

impl Display for Cell {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.point, self.value)
    }
}

#[derive(Debug, Default, Hash, Clone, Copy, PartialEq, Eq)]
pub enum CellValue {
    #[default]
    Unknown,
    Mine,
    Count(i8),
}

impl CellValue {
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

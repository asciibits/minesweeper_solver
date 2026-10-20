use std::fmt::{self, Display, Formatter};

/// Provide the x/y coordinates of a Cell with the given cell_id in a fixed
/// width field. This is the correlary to `cell_id`.
pub fn coordinates(cell_idx: usize, width: usize) -> (usize, usize) {
    let y = cell_idx / width;
    let x = cell_idx - y * width;
    (x, y)
}

/// Get a unique ID for the cell in a given-width field. Useful for indexing in
/// a Vector or Array. This is the correlary to `coordinates`.
pub fn cell_idx(x: usize, y: usize, width: usize) -> usize {
    x + y * width
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Cell {
    pub x: usize,
    pub y: usize,
    pub value: CellValue,
}

impl Cell {
    pub fn new(x: usize, y: usize) -> Cell {
        Cell {
            x,
            y,
            value: CellValue::Unknown,
        }
    }

    pub fn from_index(cell_idx: usize, width: usize) -> Cell {
        let (x, y) = coordinates(cell_idx, width);
        Cell::new(x, y)
    }

    pub fn idx(&self, width: usize) -> usize {
        cell_idx(self.x, self.y, width)
    }
}

impl Display for Cell {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // Customize your output format here
        write!(f, "({}, {}): {}", self.x, self.y, self.value)
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

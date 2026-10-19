pub mod field_printer;

#[derive(Debug, Clone)]
pub struct MineField {
    pub width: usize,
    pub height: usize,
    pub mine_count: usize,
    cells: Vec<Cell>,
}

impl MineField {
    pub fn new(width: usize, height: usize, mine_count: usize) -> MineField {
        assert!(width > 0);
        assert!(height > 0);
        assert!(mine_count > 0 && mine_count < width * height);
        let mut field = MineField {
            width,
            height,
            mine_count,
            cells: Vec::with_capacity(width * height),
        };

        for y in 0..height {
            for x in 0..width {
                field.cells.push(Cell::new(x, y, &field));
            }
        }

        field
    }
    pub fn cell(&self, x: usize, y: usize) -> &Cell {
        &self.cells[self.id(x, y)]
    }
    pub fn set_cell_value(&mut self, x: usize, y: usize, value: CellValue) {
        assert!(x < self.width);
        assert!(y < self.height);
        let id = self.id(x, y);
        self.cells[id].value = value;
    }

    // Helper function to get the unique ID
    fn id(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }
}

#[derive(Debug, Default, Hash, Clone, Copy, PartialEq, Eq)]
pub enum CellValue {
    #[default]
    Unknown,
    Mine,
    Count(i8),
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Cell {
    pub x: usize,
    pub y: usize,
    id: usize,
    pub value: CellValue,
}

impl Cell {
    fn new(x: usize, y: usize, mine_field: &MineField) -> Cell {
        Cell {
            x,
            y,
            id: mine_field.id(x, y),
            value: CellValue::Unknown,
        }
    }
}

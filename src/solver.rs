use cell::{Cell, CellValue, cell_idx};
use neighbors::Neighbors;
use std::{
    cmp::Ordering,
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    error::Error,
    fmt::{self, Display, Formatter},
};

pub mod cell;
pub mod field_printer;
mod neighbors;
mod stats;

#[derive(Debug, Clone)]
pub struct MineField {
    pub width: usize,
    pub height: usize,
    pub mine_count: usize,
    cells: Box<[ProcessedCell]>,
    dirty: HashMap<usize, CellValue>,
    partitions: HashMap<usize, Partition>,
    next_partiton_id: usize,
}

impl MineField {
    pub fn new(width: usize, height: usize, mine_count: usize) -> Result<MineField, SolverError> {
        if width.saturating_mul(height) > (usize::MAX >> 9) {
            return Err(SolverError::AreaTooLarge { width, height });
        }
        if mine_count >= width * height {
            return Err(SolverError::MineCountTooLarge {
                mine_count,
                width,
                height,
            });
        }
        let mut cells = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                cells.push(ProcessedCell::new(x, y, width, height));
            }
        }

        let field = MineField {
            width,
            height,
            mine_count,
            cells: cells.into_boxed_slice(),
            dirty: HashMap::with_capacity(width * height / 4),
            partitions: HashMap::with_capacity(20),
            next_partiton_id: 1,
        };

        Ok(field)
    }
    pub fn cell(&self, x: usize, y: usize) -> &Cell {
        &self.cells[self.cell_idx(x, y)].cell
    }
    fn set_cell_helper(&mut self, x: usize, y: usize, value: CellValue) -> Result<(), SolverError> {
        if x >= self.width || y >= self.height {
            return Err(SolverError::CoordinatesOutOfBounds {
                x,
                y,
                width: self.width,
                height: self.height,
            });
        }
        let cell = &mut self.cells[self.cell_idx(x, y)];
        if cell.value() != value {
            self.dirty.insert(cell.idx, cell.value());
            cell.set_value(value);
        }
        Ok(())
    }

    pub fn set_cell_value(
        &mut self,
        x: usize,
        y: usize,
        value: CellValue,
    ) -> Result<(), SolverError> {
        self.set_cell_helper(x, y, value)?;
        Ok(self.solve())
    }
    pub fn set_cell(&mut self, cell: &Cell) -> Result<(), SolverError> {
        self.set_cell_value(cell.x, cell.y, cell.value)
    }
    pub fn set_cells<'a>(
        &mut self,
        cells: impl IntoIterator<Item = &'a Cell>,
    ) -> Result<(), SolverError> {
        for cell in cells.into_iter() {
            self.set_cell_helper(cell.x, cell.y, cell.value)?;
        }
        Ok(self.solve())
    }

    fn filter_neighbors<P>(&self, neighbors: Neighbors, mut predicate: P) -> Neighbors
    where
        P: FnMut(&ProcessedCell) -> bool,
    {
        neighbors.filter(self.width, |id| predicate(&self.cells[id]))
    }

    fn solve(&mut self) {
        // update all indexed data
        let dirty_neighbors = self.index_dirty();

        // Clear existing dirty partitions and repartition
        let new_partition_ids = self.partition_dirty(&dirty_neighbors);

        // Find all basic logic solutions
        for partition_id in new_partition_ids {
            let partition = &self.partitions[&partition_id];

            #[derive(Debug, Copy, Hash, Clone, PartialEq, Eq)]
            struct OrderedCell {
                idx: usize,
                mine_count: i8,
                remaining_mines: i8,
                available_cells: i8,
                solution_count: u64,
            }

            impl OrderedCell {
                fn from_cell(cell: &ProcessedCell) -> Option<OrderedCell> {
                    if let CellValue::Count(mine_count) = cell.value() {
                        let available_cells = cell.alt_neighbors.len() as i8;
                        Some(OrderedCell {
                            idx: cell.idx,
                            mine_count,
                            remaining_mines: cell.remaining_mines,
                            available_cells,
                            solution_count: stats::choose(available_cells, cell.remaining_mines),
                        })
                    } else {
                        None
                    }
                }
            }

            impl PartialOrd for OrderedCell {
                fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                    Some(self.cmp(other))
                }
            }

            impl Ord for OrderedCell {
                fn cmp(&self, other: &Self) -> Ordering {
                    if self.solution_count <= 1 && other.solution_count <= 1 {
                        Ordering::Equal
                    } else {
                        self.solution_count.cmp(&other.solution_count)
                    }
                    .then(self.mine_count.cmp(&other.mine_count))
                    .then(self.remaining_mines.cmp(&other.remaining_mines))
                    .then(self.idx.cmp(&other.idx))
                }
            }

            let mut sorted_knowns = BTreeSet::new();
            for &known_cell_idx in &partition.known {
                let cell = &self.cells[known_cell_idx];
                if let Some(ordered_cell) = OrderedCell::from_cell(cell) {
                    sorted_knowns.insert(ordered_cell);
                }
            }

            dbg!(sorted_knowns);
        }
    }

    fn index_dirty(&mut self) -> HashSet<usize> {
        let mut dirty_neighbors = HashSet::with_capacity((self.width + self.height) * 2);
        for (&dirty_idx, &prev_value) in self.dirty.iter() {
            let dirty_cell = self.cells[dirty_idx];
            let mut remaining_mines = if let CellValue::Count(count) = dirty_cell.value() {
                count
            } else {
                0
            };
            struct NeighborDelta {
                neighbor_cell_idx: usize,
                is_known: bool,
            }
            let mut neighbors_to_update: [Option<NeighborDelta>; 8] = [const { Option::None }; 8];
            let mut ntu_idx = 0;
            let alt_neighbors = self.filter_neighbors(dirty_cell.neighbors, |neighbor| {
                if !self.dirty.contains_key(&neighbor.idx) {
                    dirty_neighbors.insert(neighbor.idx);
                    neighbors_to_update[ntu_idx] = Some(NeighborDelta {
                        neighbor_cell_idx: neighbor.idx,
                        is_known: neighbor.value().is_known(),
                    });
                    ntu_idx += 1;
                }

                if neighbor.value() == CellValue::Mine {
                    // user provided mine - weird
                    remaining_mines -= 1;
                }

                // mark the alt_neighbor only for known/unknown transitions, and don't include mines
                neighbor.value() != CellValue::Mine
                    && dirty_cell.value().is_known() != neighbor.value().is_known()
            });

            for idx in 0..ntu_idx {
                if let Some(NeighborDelta {
                    neighbor_cell_idx,
                    is_known,
                }) = neighbors_to_update[idx]
                {
                    let neighbor_cell = &mut self.cells[neighbor_cell_idx];

                    // update the neighbor's references
                    // let neighbor_cell = &mut self.cells[neighbor.idx];
                    if dirty_cell.value() == CellValue::Mine {
                        neighbor_cell.remaining_mines += 1;
                    }
                    if prev_value == CellValue::Mine {
                        neighbor_cell.remaining_mines -= 1;
                    }
                    if dirty_cell.value() != CellValue::Mine
                        && is_known != dirty_cell.value().is_known()
                    {
                        // add dirty_cell to neighbor's alt list
                        neighbor_cell.alt_neighbors.add(self.width, dirty_idx);
                    } else {
                        // remove dirty_cell from neighbor's alt list
                        neighbor_cell.alt_neighbors.remove(self.width, dirty_idx);
                    }
                }
            }

            let dirty_cell = &mut self.cells[dirty_idx];
            dirty_cell.alt_neighbors = alt_neighbors;
            dirty_cell.remaining_neighbors = alt_neighbors;
            dirty_cell.remaining_mines = remaining_mines;
        }

        dirty_neighbors
    }

    fn partition_dirty(&mut self, dirty_neighbors: &HashSet<usize>) -> Vec<usize> {
        self.dirty
            .keys()
            .map(|&i| self.cells[i].partition_id)
            .for_each(|p| {
                self.partitions.remove(&p);
            });
        dirty_neighbors
            .iter()
            .map(|&i| self.cells[i].partition_id)
            .for_each(|p| {
                self.partitions.remove(&p);
            });

        // Re-partition the new cells
        // clone the keys to allow for mutating self.cells[xx].partition_id
        let keys: Vec<usize> = self.dirty.keys().cloned().collect();
        let mut new_partition_ids = Vec::with_capacity(10);
        for cell_idx in keys {
            let cell = &self.cells[cell_idx];
            if !self.partitions.contains_key(&cell.partition_id) {
                if let Some(partition) = Partition::create(self, cell_idx) {
                    new_partition_ids.push(self.next_partiton_id);
                    self.partitions.insert(self.next_partiton_id, partition);
                    self.next_partiton_id += 1;
                }
            }
        }

        new_partition_ids
    }

    /// Get the cell index for the given x/y coordinates.
    fn cell_idx(&self, x: usize, y: usize) -> usize {
        cell_idx(x, y, self.width)
    }
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
struct ProcessedCell {
    // `cell` contains the provided raw data
    cell: Cell,

    // The id - calculated as cell.x + cell.y * field.width
    idx: usize,

    // The "all neighbors" id.
    neighbors: Neighbors,

    // The neighbors of different type (unknown cells have known neighbors and vice versa)
    alt_neighbors: Neighbors,

    // This is the set of all neighbors that are unknown and not inferred. It is
    // `alt_neighbors` with the inferred opens and mines removed.
    remaining_neighbors: Neighbors,

    // The number of mines needed to place into `remaining_neighbors`
    remaining_mines: i8,

    // The partition this cell belongs to.
    partition_id: usize,
}

impl ProcessedCell {
    fn new(x: usize, y: usize, width: usize, height: usize) -> ProcessedCell {
        let cell = Cell::new(x, y);
        let idx = cell_idx(x, y, width);
        let neighbors = Neighbors::all_neighbors(x, y, width, height);
        ProcessedCell {
            cell,
            idx,
            neighbors,
            alt_neighbors: Neighbors::EMPTY,
            remaining_neighbors: Neighbors::EMPTY,
            remaining_mines: 0,
            partition_id: 0,
        }
    }

    fn value(&self) -> CellValue {
        self.cell.value
    }

    fn set_value(&mut self, value: CellValue) {
        self.cell.value = value;
    }
}

impl Display for ProcessedCell {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ProcessedCell{{cell: {}, idx: {},\nneighbors: {},\nalt_neighbors: {},\nremaining_mines: {}}}",
            self.cell, self.idx, self.neighbors, self.alt_neighbors, self.remaining_mines
        )
    }
}

#[derive(Debug, Clone)]
pub struct Partition {
    known: Vec<usize>,
    unknown: Vec<usize>,
}

impl Partition {
    fn create(field: &mut MineField, cell_idx: usize) -> Option<Partition> {
        let mut known = Vec::with_capacity((field.width + field.height) * 2);
        let mut unknown = Vec::with_capacity((field.width + field.height) * 2);
        let mut members = vec![false; field.width * field.height];

        let mut to_process = VecDeque::with_capacity(field.width * field.height / 2);

        to_process.push_back(cell_idx);

        while let Some(cell_idx) = to_process.pop_front() {
            if !members[cell_idx] {
                members[cell_idx] = true;
                let cell = &mut field.cells[cell_idx];
                cell.partition_id = field.next_partiton_id;
                if cell.value() == CellValue::Mine {
                    continue;
                }
                if cell.value().is_known() {
                    known.push(cell_idx);
                } else {
                    unknown.push(cell_idx);
                }

                for neighbor_idx in cell.alt_neighbors.iter(field.width) {
                    if !members[neighbor_idx] {
                        to_process.push_back(neighbor_idx);
                    }
                }
            }
        }

        if known.is_empty() {
            None
        } else {
            Some(Partition { known, unknown })
        }
    }

    fn len(&self) -> usize {
        self.known.len() + self.unknown.len()
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum SolverError {
    AreaTooLarge {
        width: usize,
        height: usize,
    },
    MineCountTooLarge {
        mine_count: usize,
        width: usize,
        height: usize,
    },
    CoordinatesOutOfBounds {
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    },
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverError::AreaTooLarge { width, height } => write!(
                f,
                "The mine field area ({width} x {height}) must be below 2^{}.",
                usize::MAX.highest_one().unwrap() - 9
            ),
            SolverError::MineCountTooLarge {
                mine_count,
                width,
                height,
            } => {
                write!(
                    f,
                    "The mine count ({mine_count}) must be strictly less than the area of the mine field ({width} x {height}."
                )
            }
            SolverError::CoordinatesOutOfBounds {
                x,
                y,
                width,
                height,
            } => write!(
                f,
                "The provided coordinates ({x}, {y}) are outside the mine field ({width} x {height})."
            ),
        }
    }
}

impl Error for SolverError {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_neighbor_id() {
        let width = 10;
        let height = 10;
        let x = 0;
        let y = 0;
        let field = MineField::new(width, height, 0).unwrap();
    }

    #[test]
    fn do_fun_stuff() {}
}

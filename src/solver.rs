//! Core Minesweeper solver state and partition orchestration.

use cell::{Cell, CellValue, ProcessedCell};
use grid::{DirectionSet, Neighbor, Point};
use partition::{OrderedCell, Partition};
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    error::Error,
    fmt,
};

pub mod cell;
pub mod display;
pub mod grid;
pub mod partition;
mod stats;

/// Represents the state of a Minesweeper board and its active solving partitions.
#[derive(Debug, Clone)]
pub struct MineField {
    /// Width of the board in cells.
    pub width: u32,
    /// Height of the board in cells.
    pub height: u32,
    /// Total number of mines on the board.
    pub mine_count: u32,
    /// Row-major contiguous array of all board cells and their precomputed solver metadata.
    cells: Box<[ProcessedCell]>,
    /// Map of cell indices modified since the last [`MineField::solve`] pass to their previous [`CellValue`].
    dirty: HashMap<usize, CellValue>,
    /// Active connected components of frontier cells, keyed by partition ID.
    partitions: HashMap<usize, Partition>,
    /// Monotonically increasing counter used to assign unique partition IDs.
    next_partiton_id: usize,
}

impl MineField {
    /// Creates a new [`MineField`] with the given dimensions and total `mine_count`.
    ///
    /// All cells are initialized to [`CellValue::Unknown`].
    ///
    /// # Errors
    ///
    /// * Returns [`SolverError::AreaTooLarge`] if `width * height` exceeds `usize::MAX >> 9`.
    /// * Returns [`SolverError::MineCountTooLarge`] if `mine_count >= width * height`.
    pub fn new(width: u32, height: u32, mine_count: u32) -> Result<MineField, SolverError> {
        if (width as usize).saturating_mul(height as usize) > (usize::MAX >> 9) {
            return Err(SolverError::AreaTooLarge { width, height });
        }
        if mine_count as usize >= width as usize * height as usize {
            return Err(SolverError::MineCountTooLarge {
                mine_count,
                width,
                height,
            });
        }
        let mut cells = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                cells.push(ProcessedCell::new(Point { x, y }, width, height));
            }
        }

        let field = MineField {
            width,
            height,
            mine_count,
            cells: cells.into_boxed_slice(),
            dirty: HashMap::with_capacity(width as usize * height as usize / 4),
            partitions: HashMap::with_capacity(20),
            next_partiton_id: 1,
        };

        Ok(field)
    }

    /// Returns a reference to the [`Cell`] at the given coordinate `p`.
    ///
    /// # Panics
    ///
    /// Panics if `p` is outside the bounds of the field.
    pub fn cell(&self, p: Point) -> &Cell {
        &self.cells[self.cell_idx(p)].cell
    }

    /// Updates a single cell's value and records it in `self.dirty` without running the solver.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CoordinatesOutOfBounds`] if `p.x >= self.width` or `p.y >= self.height`.
    fn set_cell_helper(&mut self, p: Point, value: CellValue) -> Result<(), SolverError> {
        if p.x >= self.width || p.y >= self.height {
            return Err(SolverError::CoordinatesOutOfBounds {
                p,
                width: self.width,
                height: self.height,
            });
        }
        let cell = &mut self.cells[self.cell_idx(p)];
        if cell.value() != value {
            // always clobber the cell value. If the user sets a cell value twice before a
            // solve, we want the last value to be the one that is used.
            self.dirty.insert(cell.idx, cell.value());
            cell.set_value(value);
        }
        Ok(())
    }

    /// Sets the [`CellValue`] at coordinate `p` and triggers a solver pass.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CoordinatesOutOfBounds`] if `p` is outside the board dimensions.
    pub fn set_cell_value(&mut self, p: Point, value: CellValue) -> Result<(), SolverError> {
        self.set_cell_helper(p, value)?;
        self.solve();
        Ok(())
    }

    /// Updates the board with the given [`Cell`] and triggers a solver pass.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CoordinatesOutOfBounds`] if `cell.point` is outside the board dimensions.
    pub fn set_cell(&mut self, cell: &Cell) -> Result<(), SolverError> {
        self.set_cell_value(cell.point, cell.value)
    }

    /// Updates a batch of [`Cell`]s and runs a single solver pass after all updates are applied.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::CoordinatesOutOfBounds`] if any cell in `cells` lies outside the board.
    pub fn set_cells<'a>(
        &mut self,
        cells: impl IntoIterator<Item = &'a Cell>,
    ) -> Result<(), SolverError> {
        for cell in cells.into_iter() {
            self.set_cell_helper(cell.point, cell.value)?;
        }
        self.solve();
        Ok(())
    }

    /// Re-indexes modified cells, rebuilds affected frontier partitions, and runs deduction passes.
    fn solve(&mut self) {
        // clear the current dirty map, and get the values here
        let mut dirty = std::mem::take(&mut self.dirty);

        // update all indexed data
        let dirty_neighbors = self.index_dirty(&dirty);

        // Clear existing dirty partitions and repartition
        let new_partition_ids = self.partition_dirty(&dirty, &dirty_neighbors);

        self.basic_solve(&new_partition_ids);

        dirty.clear();
        self.dirty = dirty;
    }

    /// Updates neighbor masks (`alt_neighbors`, `remaining_neighbors`) and `remaining_mines`
    /// for all cells in `dirty` and their immediate neighbors.
    ///
    /// Returns the set of non-dirty neighbor cell indices whose adjacency state changed.
    fn index_dirty(&mut self, dirty: &HashMap<usize, CellValue>) -> HashSet<usize> {
        let mut dirty_neighbors =
            HashSet::with_capacity((self.width as usize + self.height as usize) * 2);
        for (&dirty_idx, &prev_value) in dirty.iter() {
            let dirty_cell = self.cells[dirty_idx];
            let mut remaining_mines = if let CellValue::Count(count) = dirty_cell.value() {
                count
            } else {
                0
            };

            let mut alt_neighbors = DirectionSet::new();
            for Neighbor {
                neighbor: neighbor_pt,
                from,
            } in dirty_cell.cell.point.neighbors(dirty_cell.neighbors)
            {
                let neighbor_idx = self.cell_idx(neighbor_pt);
                let neighbor_cell = &mut self.cells[neighbor_idx];
                let neighbor_val = neighbor_cell.value();

                // update the modified cell with the neighbor's data
                if neighbor_val == CellValue::Mine {
                    // user provided mine - weird. This handles one of our
                    // mines, so decrement the `remaining_mines` counter
                    remaining_mines -= 1;
                } else if dirty_cell.value() != CellValue::Mine
                    && dirty_cell.value().is_known() != neighbor_val.is_known()
                {
                    alt_neighbors.insert(from.reverse());
                }

                // update any neighbors
                if !dirty.contains_key(&neighbor_idx) {
                    dirty_neighbors.insert(neighbor_idx);
                    if dirty_cell.value() == CellValue::Mine {
                        neighbor_cell.remaining_mines -= 1;
                    }
                    if prev_value == CellValue::Mine {
                        neighbor_cell.remaining_mines += 1;
                    }
                    if dirty_cell.value() != CellValue::Mine
                        && neighbor_val.is_known() != dirty_cell.value().is_known()
                    {
                        neighbor_cell.alt_neighbors.insert(from);
                    } else {
                        neighbor_cell.alt_neighbors.remove(from);
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

    /// Invalidates partitions touched by `dirty` or `dirty_neighbors` and constructs
    /// new connected [`Partition`]s from the updated frontier.
    ///
    /// Returns the IDs of all newly created partitions.
    fn partition_dirty(
        &mut self,
        dirty: &HashMap<usize, CellValue>,
        dirty_neighbors: &HashSet<usize>,
    ) -> Vec<usize> {
        dirty
            .keys()
            .chain(dirty_neighbors.iter())
            .map(|&i| self.cells[i].partition_id)
            .for_each(|p| {
                self.partitions.remove(&p);
            });

        // Re-partition the new cells
        // clone the keys to allow for mutating self.cells[xx].partition_id
        let mut new_partition_ids = Vec::with_capacity(10);
        for cell_idx in dirty.keys().chain(dirty_neighbors.iter()).copied() {
            let cell = &self.cells[cell_idx];
            if !self.partitions.contains_key(&cell.partition_id) {
                let partition_id = self.next_partiton_id;
                if let Some(partition) = Partition::create(
                    &mut self.cells,
                    self.width,
                    self.height,
                    cell_idx,
                    partition_id,
                ) {
                    new_partition_ids.push(partition_id);
                    self.partitions.insert(partition_id, partition);
                    self.next_partiton_id += 1;
                }
            }
        }

        new_partition_ids
    }

    /// Performs local constraint deductions on the newly created partitions.
    fn basic_solve(&mut self, new_partition_ids: &[usize]) {
        // Find all basic logic solutions
        for partition_id in new_partition_ids {
            let partition = &self.partitions[partition_id];

            let mut sorted_knowns = BTreeSet::new();
            for &known_cell_idx in &partition.known {
                let cell = &self.cells[known_cell_idx];
                if let Some(ordered_cell) = OrderedCell::from_cell(cell) {
                    sorted_knowns.insert(ordered_cell);
                }
            }

            eprintln!("For Partition {partition_id}");
            dbg!(sorted_knowns);
        }
    }

    /// Get the cell index for the given x/y coordinates.
    fn cell_idx(&self, p: Point) -> usize {
        p.to_index(self.width)
    }
}

/// Errors that can occur when constructing or updating a [`MineField`].
#[derive(Debug)]
#[non_exhaustive]
pub enum SolverError {
    /// The requested board dimensions exceed the maximum supported cell count.
    AreaTooLarge {
        width: u32,
        height: u32,
    },
    /// The requested mine count is greater than or equal to the total number of cells.
    MineCountTooLarge {
        mine_count: u32,
        width: u32,
        height: u32,
    },
    /// A cell coordinate lies outside the board's `width` and `height`.
    CoordinatesOutOfBounds {
        p: Point,
        width: u32,
        height: u32,
    },
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverError::AreaTooLarge { width, height } => write!(
                f,
                "The mine field area ({width} x {height}) must be below 2^{}.",
                usize::BITS - 10
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
            SolverError::CoordinatesOutOfBounds { p, width, height } => write!(
                f,
                "The provided coordinates {p} are outside the mine field ({width} x {height})."
            ),
        }
    }
}

impl Error for SolverError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::grid::{E, S, SE, W};

    #[test]
    fn test_new_minefield_validation() {
        let field = MineField::new(10, 10, 15).expect("valid field");
        assert_eq!(field.width, 10);
        assert_eq!(field.height, 10);
        assert_eq!(field.mine_count, 15);
        assert_eq!(
            field.cell(Point { x: 0, y: 0 }).value,
            CellValue::Unknown
        );

        // AreaTooLarge
        let err = MineField::new(u32::MAX, u32::MAX, 10).unwrap_err();
        assert!(matches!(err, SolverError::AreaTooLarge { .. }));
        assert!(err.to_string().contains("must be below"));

        // MineCountTooLarge
        let err = MineField::new(4, 4, 16).unwrap_err();
        assert!(matches!(
            err,
            SolverError::MineCountTooLarge {
                mine_count: 16,
                width: 4,
                height: 4
            }
        ));
        assert!(err.to_string().contains("must be strictly less"));

        // 0x0 field rejected as MineCountTooLarge
        assert!(matches!(
            MineField::new(0, 0, 0),
            Err(SolverError::MineCountTooLarge { .. })
        ));
    }

    #[test]
    fn test_set_cell_out_of_bounds() {
        let mut field = MineField::new(3, 3, 1).unwrap();
        let err = field
            .set_cell_value(Point { x: 3, y: 0 }, CellValue::Count(1))
            .unwrap_err();
        assert!(matches!(
            err,
            SolverError::CoordinatesOutOfBounds {
                p: Point { x: 3, y: 0 },
                width: 3,
                height: 3
            }
        ));
        assert!(err.to_string().contains("outside the mine field"));

        let err_y = field
            .set_cell(&Cell {
                point: Point { x: 0, y: 3 },
                value: CellValue::Count(1),
            })
            .unwrap_err();
        assert!(matches!(err_y, SolverError::CoordinatesOutOfBounds { .. }));
    }

    #[test]
    fn test_index_dirty_and_remaining_mines_tracking() {
        let mut field = MineField::new(3, 3, 2).unwrap();

        // Reveal (0, 0) as Count(2)
        field
            .set_cell_value(Point { x: 0, y: 0 }, CellValue::Count(2))
            .unwrap();
        let c00 = field.cells[field.cell_idx(Point { x: 0, y: 0 })];
        assert_eq!(c00.remaining_mines, 2);
        assert_eq!(c00.alt_neighbors, [E, S, SE].into_iter().collect());

        // Neighbor (1, 0) is Unknown and should now list W in its alt_neighbors
        let c10 = field.cells[field.cell_idx(Point { x: 1, y: 0 })];
        assert!(c10.alt_neighbors.contains(W));

        // Mark (1, 0) as a Mine: (0, 0)'s remaining_mines should decrement to 1
        // and E should be removed from (0, 0)'s alt_neighbors
        field
            .set_cell_value(Point { x: 1, y: 0 }, CellValue::Mine)
            .unwrap();
        let c00 = field.cells[field.cell_idx(Point { x: 0, y: 0 })];
        assert_eq!(c00.remaining_mines, 1);
        assert_eq!(c00.alt_neighbors, [S, SE].into_iter().collect());

        // Change (1, 0) from Mine to Count(1): (0, 0)'s remaining_mines should restore to 2
        field
            .set_cell_value(Point { x: 1, y: 0 }, CellValue::Count(1))
            .unwrap();
        let c00 = field.cells[field.cell_idx(Point { x: 0, y: 0 })];
        assert_eq!(c00.remaining_mines, 2);
        assert_eq!(c00.alt_neighbors, [S, SE].into_iter().collect());
    }

    #[test]
    fn test_partitioning_and_repartitioning_across_updates() {
        // 5x1 strip: [Count(1), Unknown, Unknown, Unknown, Count(1)]
        let mut field = MineField::new(5, 1, 2).unwrap();
        field
            .set_cells(&[
                Cell {
                    point: Point { x: 0, y: 0 },
                    value: CellValue::Count(1),
                },
                Cell {
                    point: Point { x: 4, y: 0 },
                    value: CellValue::Count(1),
                },
            ])
            .unwrap();

        // Two disconnected frontier components: {(0,0), (1,0)} and {(3,0), (4,0)}
        assert_eq!(field.partitions.len(), 2);
        let p_left = field.cells[0].partition_id;
        let p_right = field.cells[4].partition_id;
        assert_ne!(p_left, p_right);
        assert_eq!(field.partitions[&p_left].len(), 2);
        assert_eq!(field.partitions[&p_right].len(), 2);

        // Reveal the middle cell (2, 0) as Count(1): now (1, 0) and (3, 0) are both
        // adjacent to (2, 0), merging both components into a single 5-cell partition!
        field
            .set_cell_value(Point { x: 2, y: 0 }, CellValue::Count(1))
            .unwrap();
        assert_eq!(field.partitions.len(), 1);
        let merged_partition = field.partitions.values().next().unwrap();
        assert_eq!(merged_partition.len(), 5);
    }
}

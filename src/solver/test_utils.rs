//! Test utilities for constructing and updating [`MineField`] instances from compact ASCII specs
//! or mine-placement builders.
//!
//! # Specification Format
//!
//! [`MineField::create_minefield`], [`MineField::update_minefield`], and [`FieldBuilder::from_spec`] accept an iterator
//! of string slices representing rows of the board (top-to-bottom, with `y = 0` first and `x = 0`
//! leftmost). Strings containing embedded newline (`\n`) characters are automatically split into
//! separate rows.
//!
//! Each character in a row maps to a coordinate `(x, y)` as follows:
//! - `'0'..='8'`: Sets the cell to [`UserCellValue::Safe(n)`](UserCellValue::Safe) (or, in
//!   [`FieldBuilder::from_spec`], asserts that the computed adjacent mine count equals `n`).
//! - `'*'`, `'X'`, or `'x'`: Sets the cell to [`UserCellValue::Mine`].
//! - `'_'`: Explicitly sets the cell to [`UserCellValue::Unknown`] (useful with [`MineField::update_minefield`]
//!   to revert a previously known cell back to unknown).
//! - `'.'` or `' '`: Skips the cell ("do nothing"). During [`MineField::create_minefield`] and
//!   [`FieldBuilder::from_spec`], the coordinate still contributes to the board dimensions and
//!   defaults to [`UserCellValue::Unknown`]; during [`MineField::update_minefield`], the existing cell value at
//!   `(x, y)` is left untouched.
//! - Any other character panics with `"Invalid field character"`.

use crate::{
    MineField, Point,
    solver::{
        IndexedOffsets,
        cell::{UserCell, UserCellValue, cell_idx, coordinates},
        grid::{DirectionSet, Neighbor},
        indexed_neighbor_iter,
    },
};

use std::collections::HashSet;

/// Helper for building a consistent [`MineField`] from known or randomly generated mine locations
/// and selectively exposing or flood-filling cells during tests.
///
/// Unlike [`MineField::create_minefield`], which directly sets cell states on a [`MineField`] without
/// requiring all mines to be placed, `FieldBuilder` maintains the complete ground-truth board
/// (`field_data`)—computing the exact adjacent mine count for every non-mine cell—while keeping
/// the underlying [`MineField`] cells unexposed ([`UserCellValue::Unknown`]) until [`expose_cells`](Self::expose_cells)
/// or [`expand_cells`](Self::expand_cells) is called.
///
/// # Examples
///
/// ```
/// # use minesweeper_solver::{CellValue, FieldBuilder, Point};
/// // Build a 3x3 board with a single mine at the bottom-right corner (2, 2):
/// let mut builder = FieldBuilder::from_points(3, 3, [Point { x: 2, y: 2 }]);
///
/// // Flood-fill from top-left (0, 0), which has Safe(0), automatically exposing
/// // the connected zero region and its clue boundary:
/// builder.expand_cells([Point { x: 0, y: 0 }]);
///
/// let field = builder.into_mine_field();
/// assert_eq!(field.cell(Point { x: 0, y: 0 }).value, CellValue::Safe(0));
/// assert_eq!(field.cell(Point { x: 1, y: 1 }).value, CellValue::Safe(1));
/// assert_eq!(field.cell(Point { x: 2, y: 2 }).value, CellValue::Unknown);
/// ```
#[derive(Debug)]
pub struct FieldBuilder {
    /// Ground-truth board values in row-major order: `0..=8` for clue counts, or [`Self::MINE`] for mines.
    field_data: Box<[i8]>,
    /// The solver's [`MineField`] instance whose cells are progressively revealed via [`Self::expose_cells`] or [`Self::expand_cells`].
    mine_field: MineField,
}

impl FieldBuilder {
    /// Sentinel value stored in `field_data` to represent a mine.
    ///
    /// Chosen as `-10` so that even if all 8 neighbors are mines (incrementing the cell 8 times
    /// during neighbor-count accumulation), the value remains negative (`-10 + 8 = -2 < 0`).
    const MINE: i8 = -10;

    /// Encodes a [`UserCellValue`] from an ASCII spec into a temporary `i8` representation before
    /// neighbor mine counts are accumulated.
    fn value_to_i8(value: UserCellValue) -> i8 {
        match value {
            UserCellValue::Mine => Self::MINE,
            UserCellValue::Unknown => 0,
            // a little hacky, but store the user provided values in bits 4-7
            // they will be compared to the final bits 0-3 in the last pass to
            // verify the user values. Add one so this value is distinct from
            // not being specified.
            UserCellValue::Safe(n) => ((n + 1) << 4) as i8,
        }
    }

    /// Creates a [`FieldBuilder`] of dimensions `width` × `height` with mines placed at `mines`.
    ///
    /// All cells in the underlying [`MineField`] start as [`UserCellValue::Unknown`].
    ///
    /// # Panics
    ///
    /// Panics if any point in `mines` lies outside `width` × `height`, or if [`MineField::new`]
    /// fails validation (e.g., all cells are mines).
    pub fn from_points(width: u32, height: u32, mines: impl IntoIterator<Item = Point>) -> Self {
        let mut field = vec![0i8; width as usize * height as usize];
        for pt in mines {
            field[cell_idx(pt, width)] = Self::MINE;
        }

        Self::from_vec(width, height, field)
    }

    /// Creates a [`FieldBuilder`] from an ASCII `spec`, computing true neighbor counts from the
    /// specified mines (`'*'`, `'X'`, `'x'`) and verifying any user-supplied clue digits (`'0'..='8'`).
    ///
    /// All cells in the underlying [`MineField`] start as [`UserCellValue::Unknown`] until revealed
    /// with [`expose_cells`](Self::expose_cells) or [`expand_cells`](Self::expand_cells).
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `spec` contains an invalid character.
    /// - Any digit `'0'..='8'` in `spec` does not match the actual number of mines adjacent to that coordinate.
    /// - The inferred board dimensions and mine count fail [`MineField::new`] validation.
    pub fn from_spec<'i>(spec: impl IntoIterator<Item = &'i str>) -> Self {
        let (width, height, cell_iter) = MineField::cells_and_size_from_spec(spec);
        let mut field = vec![0i8; width as usize * height as usize];
        for UserCell { point, value } in cell_iter {
            field[cell_idx(point, width)] = Self::value_to_i8(value);
        }

        Self::from_vec(width, height, field)
    }

    /// Creates a [`FieldBuilder`] of dimensions `width` × `height` with `mine_count` randomly placed mines.
    pub fn random(
        width: u32,
        height: u32,
        mine_count: u32,
        shuffle: impl FnMut(&mut [i8]),
    ) -> Self {
        Self::random_helper(width, height, mine_count, [], shuffle)
    }

    /// Creates a [`FieldBuilder`] of dimensions `width` × `height` with `mine_count` randomly placed mines,
    /// guaranteeing that `zero_cell` and all of its neighbors are mine-free (so `zero_cell` has count `0`).
    pub fn random_with_zero_region(
        width: u32,
        height: u32,
        mine_count: u32,
        zero_cell: Point,
        shuffle: impl FnMut(&mut [i8]),
    ) -> Self {
        Self::random_helper(
            width,
            height,
            mine_count,
            zero_cell
                .neighbors(DirectionSet::all_neighbors(zero_cell, width, height))
                .map(|n| n.neighbor)
                .chain([zero_cell]),
            shuffle,
        )
    }

    /// Places `mine_count` mines randomly across a `width` × `height` board while keeping all
    /// `reserved_points` free of mines.
    fn random_helper(
        width: u32,
        height: u32,
        mine_count: u32,
        reserved_points: impl IntoIterator<Item = Point>,
        mut shuffle: impl FnMut(&mut [i8]),
    ) -> Self {
        let len = width as usize * height as usize;
        let mut field = vec![0i8; len];
        let mut reserved_indices: Vec<_> = reserved_points
            .into_iter()
            .map(|pt| cell_idx(pt, width))
            .collect();
        reserved_indices.sort_unstable();
        reserved_indices.dedup();
        let end_shuffle = len - reserved_indices.len();
        assert!(
            mine_count as usize <= end_shuffle,
            "Too many mines for reserved region"
        );
        field[..mine_count as usize].fill(Self::MINE);

        shuffle(&mut field[0..end_shuffle]);

        for (i, reserve_idx) in (end_shuffle..).zip(reserved_indices) {
            field.swap(i, reserve_idx);
        }

        Self::from_vec(width, height, field)
    }

    /// Accumulates neighbor mine counts across `field`, validates any user-specified counts packed
    /// into the upper nibble, and initializes the [`MineField`].
    fn from_vec(width: u32, height: u32, mut field: Vec<i8>) -> Self {
        assert_eq!(width as usize * height as usize, field.len());

        let offsets = IndexedOffsets::new(width);

        // apply mine counts to neighbors
        let mut idx = 0;
        for y in 0..height {
            for x in 0..width {
                if field[idx] < 0 && field[idx] >= Self::MINE {
                    let pt = Point { x, y };
                    for Neighbor { neighbor, .. } in indexed_neighbor_iter(
                        idx,
                        DirectionSet::all_neighbors(pt, width, height),
                        offsets,
                    ) {
                        field[neighbor] += 1;
                    }
                }
                idx += 1;
            }
        }

        // second pass - validate user numbers and normalize mine values
        let mut mine_count = 0;
        for (idx, val) in field.iter_mut().enumerate() {
            if *val < 0 && *val >= Self::MINE {
                *val = Self::MINE;
                mine_count += 1;
            } else {
                let user_val = (*val as u8 >> 4) as i8 - 1;
                *val &= 0xf;
                assert!(
                    user_val == -1 || user_val == *val,
                    "Invalid user provided value: {user_val} at coordinates {}. \
                        It does not reflect the actual mine count {val}",
                    coordinates(idx, width)
                );
            }
        }

        Self {
            field_data: field.into_boxed_slice(),
            mine_field: MineField::new(width, height, mine_count).unwrap(),
        }
    }

    /// Returns a mutable reference to the underlying [`MineField`].
    pub fn mine_field(&mut self) -> &mut MineField {
        &mut self.mine_field
    }

    /// Consumes the builder and returns the underlying [`MineField`].
    pub fn into_mine_field(self) -> MineField {
        self.mine_field
    }

    /// Converts the ground-truth `field_data` entry at `p` into a [`CellValue`].
    fn cell_value(field_data: &[i8], width: u32, p: Point) -> UserCellValue {
        match field_data[cell_idx(p, width)] {
            n @ 0..=8 => UserCellValue::Safe(n as u8),
            Self::MINE => UserCellValue::Mine,
            _ => unreachable!(),
        }
    }

    /// Reveals the ground-truth [`CellValue`]s for the specified `points` on the underlying [`MineField`].
    pub fn expose_cells(&mut self, points: impl IntoIterator<Item = Point>) -> &mut Self {
        let field_data = &self.field_data;
        let width = self.mine_field.width();
        self.mine_field
            .set_cells(points.into_iter().map(|point| UserCell {
                point,
                value: Self::cell_value(field_data, width, point),
            }))
            .unwrap();
        self
    }

    /// Reveals `points` on the underlying [`MineField`] and recursively flood-fills any revealed
    /// `0`-count cells to expose their entire connected zero-region and its adjacent clue boundary.
    pub fn expand_cells(&mut self, points: impl IntoIterator<Item = Point>) -> &mut Self {
        let field_data = &self.field_data;
        let width = self.mine_field.width;
        let height = self.mine_field.height;
        let mut all_points = HashSet::new();
        let mut zero_queue = Vec::new();
        for p in points.into_iter() {
            all_points.insert(p);
            let idx = cell_idx(p, width);
            if field_data[idx] == 0 {
                zero_queue.push(p);
            }
        }
        while let Some(p) = zero_queue.pop() {
            for Neighbor { neighbor, .. } in
                p.neighbors(DirectionSet::all_neighbors(p, width, height))
            {
                if all_points.insert(neighbor) {
                    let n_idx = cell_idx(neighbor, width);
                    if field_data[n_idx] == 0 {
                        zero_queue.push(neighbor);
                    }
                }
            }
        }

        self.expose_cells(all_points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rng, seq::SliceRandom};

    #[test]
    fn test_field_builder_from_points_and_expose_cells() {
        // 3x2 board with mines at (0, 0) and (2, 1)
        let mut builder =
            FieldBuilder::from_points(3, 2, [Point { x: 0, y: 0 }, Point { x: 2, y: 1 }]);
        assert_eq!(builder.mine_field().width(), 3);
        assert_eq!(builder.mine_field().height(), 2);
        assert_eq!(builder.mine_field().mine_count(), 2);

        // Initially all cells on the MineField are Unknown
        for y in 0..2 {
            for x in 0..3 {
                assert_eq!(
                    builder.mine_field().cell(Point { x, y }).value,
                    UserCellValue::Unknown
                );
            }
        }

        // Expose (1, 0) (adjacent to both mines -> Count(2)) and (0, 0) (Mine)
        builder.expose_cells([Point { x: 1, y: 0 }, Point { x: 0, y: 0 }]);
        let field = builder.into_mine_field();
        assert_eq!(field.cell(Point { x: 0, y: 0 }).value, UserCellValue::Mine);
        assert_eq!(
            field.cell(Point { x: 1, y: 0 }).value,
            UserCellValue::Safe(2)
        );
        assert_eq!(
            field.cell(Point { x: 2, y: 0 }).value,
            UserCellValue::Unknown
        );
    }

    #[test]
    fn test_field_builder_from_spec_and_high_counts() {
        // 3x3 board with 8 mines surrounding center (1, 1), verifying '8' and mine specifiers '*', 'X', 'x'
        let mut builder = FieldBuilder::from_spec(["*Xx", "*8*", "xX*"]);
        assert_eq!(builder.mine_field().width(), 3);
        assert_eq!(builder.mine_field().height(), 3);
        assert_eq!(builder.mine_field().mine_count(), 8);

        builder.expose_cells([Point { x: 1, y: 1 }]);
        assert_eq!(
            builder.mine_field().cell(Point { x: 1, y: 1 }).value,
            UserCellValue::Safe(8)
        );

        // Also test a valid spec with mixed counts (0, 1, 2, 7) and '_'
        let mut builder7 = FieldBuilder::from_spec(["*X*", "*7*", "*_*", "121", "000"]);
        builder7.expose_cells([
            Point { x: 1, y: 1 },
            Point { x: 1, y: 2 },
            Point { x: 1, y: 3 },
            Point { x: 1, y: 4 },
        ]);
        let field7 = builder7.into_mine_field();
        assert_eq!(
            field7.cell(Point { x: 1, y: 1 }).value,
            UserCellValue::Safe(7)
        );
        assert_eq!(
            field7.cell(Point { x: 1, y: 2 }).value,
            UserCellValue::Safe(4)
        );
        assert_eq!(
            field7.cell(Point { x: 1, y: 3 }).value,
            UserCellValue::Safe(2)
        );
        assert_eq!(
            field7.cell(Point { x: 1, y: 4 }).value,
            UserCellValue::Safe(0)
        );
    }

    #[test]
    #[should_panic(expected = "Invalid user provided value: 2 at coordinates (0, 0)")]
    fn test_field_builder_from_spec_mismatched_count_panics() {
        // (0, 0) only has 1 adjacent mine at (1, 0), so specifying '2' must panic
        FieldBuilder::from_spec(["2*."]);
    }

    #[test]
    fn test_field_builder_expand_cells_flood_fill() {
        // 4x3 board with a single mine at bottom-right (3, 2):
        // Ground-truth counts:
        // (0,0)=0  (1,0)=0  (2,0)=0  (3,0)=0
        // (0,1)=0  (1,1)=0  (2,1)=1  (3,1)=1
        // (0,2)=0  (1,2)=0  (2,2)=1  (3,2)=*
        let mut builder = FieldBuilder::from_spec(["....", "..11", "..1*"]);

        // Expanding from a non-zero cell (2, 1) only exposes that single cell
        builder.expand_cells([Point { x: 2, y: 1 }]);
        assert_eq!(
            builder.mine_field().cell(Point { x: 2, y: 1 }).value,
            UserCellValue::Safe(1)
        );
        assert_eq!(
            builder.mine_field().cell(Point { x: 0, y: 0 }).value,
            UserCellValue::Unknown
        );

        // Expanding from (0, 0) flood-fills all 0-cells and their adjacent clue boundary,
        // leaving only (3, 2) unexposed (and inferred as Mine by the solver!)
        builder.expand_cells([Point { x: 0, y: 0 }]);
        let field = builder.into_mine_field();
        for y in 0..3 {
            for x in 0..4 {
                let pt = Point { x, y };
                if pt == (Point { x: 3, y: 2 }) {
                    assert_eq!(field.cell(pt).value, UserCellValue::Unknown);
                    assert_eq!(field.inferred_mines(), vec![pt]);
                } else {
                    assert!(matches!(field.cell(pt).value, UserCellValue::Safe(_)));
                }
            }
        }
    }

    #[test]
    fn test_field_builder_random_and_zero_region() {
        let random_builder = FieldBuilder::random(6, 5, 7, |cells| cells.shuffle(&mut rng()));
        assert_eq!(random_builder.mine_field.width(), 6);
        assert_eq!(random_builder.mine_field.height(), 5);
        assert_eq!(random_builder.mine_field.mine_count(), 7);
        let actual_mines = random_builder
            .field_data
            .iter()
            .filter(|&&v| v == FieldBuilder::MINE)
            .count();
        assert_eq!(actual_mines, 7);

        let zero_pt = Point { x: 2, y: 2 };
        let mut zero_builder = FieldBuilder::random_with_zero_region(6, 5, 10, zero_pt, |cells| {
            cells.shuffle(&mut rng())
        });
        zero_builder.expand_cells([zero_pt]);
        let field = zero_builder.into_mine_field();
        assert_eq!(field.cell(zero_pt).value, UserCellValue::Safe(0));
        for Neighbor { neighbor, .. } in
            zero_pt.neighbors(DirectionSet::all_neighbors(zero_pt, 6, 5))
        {
            assert!(matches!(field.cell(neighbor).value, UserCellValue::Safe(_)));
        }
    }
}

//! Test utilities for constructing and updating [`MineField`] instances from compact ASCII specs
//! or mine-placement builders.
//!
//! # Specification Format
//!
//! [`MineField::create`], [`MineField::update`], and [`FieldBuilder::from_spec`] accept an iterator
//! of string slices representing rows of the board (top-to-bottom, with `y = 0` first and `x = 0`
//! leftmost). Strings containing embedded newline (`\n`) characters are automatically split into
//! separate rows.
//!
//! Each character in a row maps to a coordinate `(x, y)` as follows:
//! - `'0'..='8'`: Sets the cell to [`crate::CellValue::Safe(n)`](crate::CellValue::Safe) (or, in
//!   [`FieldBuilder::from_spec`], asserts that the computed adjacent mine count equals `n`).
//! - `'*'`: Sets the cell to [`crate::CellValue::Mine`].
//! - `'_'`, `'x'`, or `'#'`: Explicitly sets the cell to [`crate::CellValue::Unknown`] (useful with [`MineField::update`]
//!   to revert a previously known cell back to unknown; `'x'` represents an inferred mine and `'#'` represents an inferred safe cell in text output).
//! - `'.'` or `' '`: Skips the cell ("do nothing"). During [`MineField::create`] and
//!   [`FieldBuilder::from_spec`], the coordinate still contributes to the board dimensions and
//!   defaults to [`crate::CellValue::Unknown`]; during [`MineField::update`], the existing cell value at
//!   `(x, y)` is left untouched. Note that leading whitespace is trimmed from each line, so use `'.'` for leading unspecified cells.
//! - Any other character panics with `"Invalid field character"`.

use crate::{
    MineField, Point,
    solver::{
        Cell, CellIdx, CellValue, FastNumberHasherBuilder,
        cell::{UserCellValue, cell_idx, coordinates},
        grid::{DirectionSet, Neighbor},
    },
};

use core::slice;
use std::{
    collections::HashSet,
    fmt::{self, Debug, Display},
    ops::{Index, IndexMut},
};

/// Helper for building a consistent [`MineField`] from known or randomly generated mine locations
/// and selectively exposing or flood-filling cells during tests.
///
/// Unlike [`MineField::create`], which directly sets cell states on a [`MineField`] without
/// requiring all mines to be placed, `FieldBuilder` maintains the complete ground-truth board
/// (`mines`)—computing the exact adjacent mine count for every non-mine cell—while keeping
/// the underlying [`MineField`] cells unexposed ([`crate::CellValue::Unknown`]) until `expose_cells`
/// or `expand_cells` is called.
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
/// builder.expand_points([Point { x: 0, y: 0 }]);
///
/// let field = builder.into_mine_field();
/// assert_eq!(field.cell(Point { x: 0, y: 0 }).value, CellValue::Safe(0));
/// assert_eq!(field.cell(Point { x: 1, y: 1 }).value, CellValue::Safe(1));
/// assert_eq!(field.cell(Point { x: 2, y: 2 }).value, CellValue::Unknown);
/// ```
#[derive(Debug)]
pub struct FieldBuilder {
    /// Ground-truth board values in row-major order: `0..=8` for clue counts, or [`Self::MINE`] for mines.
    mines: Box<[MineOrCount]>,
    /// The solver's [`MineField`] instance whose cells are progressively revealed via [`Self::expose_cells`] or [`Self::expand_cells`].
    field: MineField,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
struct MineOrCount(i8);

impl MineOrCount {
    const MINE_VALUE: i8 = -1;
    const MINE: MineOrCount = MineOrCount(Self::MINE_VALUE);
    const ZERO: MineOrCount = MineOrCount(0);

    fn is_mine(&self) -> bool {
        self.0 < 0
    }

    fn count(&self) -> i8 {
        self.0
    }
}

impl From<MineOrCount> for CellValue {
    fn from(value: MineOrCount) -> Self {
        match value.0 {
            n @ 0..=8 => CellValue::for_mine_count(n as u8),
            MineOrCount::MINE_VALUE => CellValue::MINE,
            _ => unreachable!(),
        }
    }
}

impl From<MineOrCount> for UserCellValue {
    fn from(value: MineOrCount) -> Self {
        match value.0 {
            n @ 0..=8 => UserCellValue::Safe(n as u8),
            MineOrCount::MINE_VALUE => UserCellValue::Mine,
            _ => unreachable!(),
        }
    }
}

impl Index<CellIdx> for [MineOrCount] {
    type Output = MineOrCount;

    #[inline]
    fn index(&self, index: CellIdx) -> &Self::Output {
        &self[index.0 as usize]
    }
}

impl IndexMut<CellIdx> for [MineOrCount] {
    #[inline]
    fn index_mut(&mut self, index: CellIdx) -> &mut Self::Output {
        &mut self[index.0 as usize]
    }
}

impl Index<CellIdx> for Vec<MineOrCount> {
    type Output = MineOrCount;

    #[inline]
    fn index(&self, index: CellIdx) -> &Self::Output {
        &self[index.0 as usize]
    }
}

impl IndexMut<CellIdx> for Vec<MineOrCount> {
    #[inline]
    fn index_mut(&mut self, index: CellIdx) -> &mut Self::Output {
        &mut self[index.0 as usize]
    }
}

impl FieldBuilder {
    /// Creates a [`FieldBuilder`] of dimensions `width` × `height` with mines placed at `mines`.
    ///
    /// All cells in the underlying [`MineField`] start as [`crate::CellValue::Unknown`].
    ///
    /// # Panics
    ///
    /// Panics if any point in `mines` lies outside `width` × `height`, or if [`MineField::new`]
    /// fails validation (e.g., all cells are mines).
    pub fn from_points(width: u32, height: u32, mines: impl IntoIterator<Item = Point>) -> Self {
        let mut field = vec![MineOrCount::ZERO; width as usize * height as usize];
        for pt in mines {
            assert!(
                pt.x < width && pt.y < height,
                "Point ({}, {}) is out of bounds for {width}x{height}",
                pt.x,
                pt.y
            );
            field[cell_idx(pt, width)] = MineOrCount::MINE;
        }

        Self::from_vec(width, height, field)
    }

    /// Creates a [`FieldBuilder`] from an ASCII `spec`.
    ///
    /// The `spec` can be formatted in one of two ways:
    ///
    /// 1. **Single field spec (ground truth only)**:
    ///    Contains only the mine layout (`'*'` for mines, `'.'` for safe cells, or optional verified
    ///    clue digits `'0'..='8'`). True neighbor mine counts are computed for all non-mine cells,
    ///    and all cells in the underlying [`MineField`] start as [`CellValue::Unknown`](crate::CellValue::Unknown)
    ///    until revealed with `expose_cells` or `expand_cells`.
    ///
    /// 2. **Two-part field spec (ground truth and field state)**:
    ///    Consists of two field specs separated by a single line containing one contiguous set of
    ///    `'='` characters (such as `===`).
    ///    - The upper section defines the ground-truth mine layout (same as the single spec format).
    ///    - The lower section defines cell states applied to the underlying [`MineField`] via
    ///      [`MineField::update`](crate::MineField::update), allowing tests to initialize boards with
    ///      pre-revealed clues, flags, or inferences.
    ///
    /// This two-part format directly lines up with the output of [`Display::fmt`](Self#impl-Display-for-FieldBuilder),
    /// making it possible to serialize and deserialize field builder states or copy test scenarios directly
    /// from displayed board output.
    ///
    /// Leading whitespace on each row is trimmed, so multi-line string literals can follow code indentation.
    /// Use `'.'` to represent leading unspecified cells.
    ///
    /// # Note
    ///
    /// Numbered clue cells in the ground-truth portion should generally not be provided as they are
    /// calculated directly from the placed mines. If specified, they must match the calculated values.
    ///
    /// # Examples
    ///
    /// Initializing with both ground-truth mines and pre-revealed field cells separated by `'='` characters:
    ///
    /// ```
    /// # use minesweeper_solver::{CellValue, FieldBuilder, Point};
    /// // 3x2 board with a mine at (0, 0), separated by '===':
    /// // - Upper section: ground-truth layout (*.. / ...)
    /// // - Lower section: (1, 0) is revealed as Safe(1)
    /// let mut builder = FieldBuilder::from_spec(
    ///     "*..
    ///      ...
    ///      ===
    ///      .1.
    ///      ...",
    /// );
    ///
    /// assert_eq!(builder.mine_field().width(), 3);
    /// assert_eq!(builder.mine_field().height(), 2);
    ///
    /// // The underlying MineField has (1, 0) revealed as Safe(1):
    /// assert_eq!(
    ///     builder.mine_field().cell(Point { x: 1, y: 0 }).value,
    ///     CellValue::Safe(1)
    /// );
    /// assert_eq!(
    ///     builder.mine_field().cell(Point { x: 0, y: 0 }).value,
    ///     CellValue::Unknown
    /// );
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `spec` contains an invalid character.
    /// - Any digit `'0'..='8'` in the ground-truth spec does not match the actual number of mines adjacent to that coordinate.
    /// - The inferred board dimensions and mine count fail [`MineField::new`] validation.
    pub fn from_spec(spec: &str) -> Self {
        let mut offset = 0;
        let mut sep_range = None;
        for line in spec.split_inclusive('\n') {
            let trimmed = line.trim();
            if !trimmed.is_empty() && trimmed.chars().all(|c| c == '=') {
                sep_range = Some((offset, offset + line.len()));
                break;
            }
            offset += line.len();
        }

        let (mine_spec, field_spec) = match sep_range {
            Some((start, end)) => (&spec[..start], &spec[end..]),
            None => (spec, ""),
        };

        let (width, height, cell_iter) = MineField::cells_and_size_from_spec(mine_spec);
        let mut field = vec![MineOrCount::ZERO; width as usize * height as usize];
        let mut user_checks = Vec::new();
        for (idx, value) in cell_iter {
            if value.is_mine() {
                field[idx] = MineOrCount::MINE;
            } else if value.has_count() {
                user_checks.push((idx, value.mine_count()));
            }
        }
        let mut builder = Self::from_vec(width, height, field);
        for (idx, expected) in user_checks {
            let actual = builder.mines[idx];
            assert!(
                actual.0 == expected as i8,
                "Invalid user provided value: {expected} at coordinates {}. \
                 It does not reflect the actual mine count {}",
                coordinates(idx, width),
                actual.count()
            );
        }
        if !field_spec.trim().is_empty() {
            builder.field.update(field_spec);
        }
        builder
    }

    /// Creates a [`FieldBuilder`] of dimensions `width` × `height` with `mine_count` randomly placed mines.
    ///
    /// The caller provides a `shuffle` closure that randomly permutes the provided slice of cells.
    /// This allows callers to use their own random number generator (such as `slice.shuffle(&mut rng)`
    /// from [`rand::seq::SliceRandom`](https://docs.rs/rand/latest/rand/seq/trait.SliceRandom.html))
    /// or a deterministic shuffling algorithm without requiring `minesweeper_solver` to depend directly on `rand`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minesweeper_solver::FieldBuilder;
    /// // Using a simple deterministic closure:
    /// let builder = FieldBuilder::random(4, 4, 3, |cells| cells.reverse());
    /// let field = builder.into_mine_field();
    /// assert_eq!(field.mine_count(), 3);
    /// ```
    #[cfg(test)]
    pub(super) fn random(
        width: u32,
        height: u32,
        mine_count: u32,
        shuffle: impl FnMut(&mut [i8]),
    ) -> Self {
        Self::random_helper(width, height, mine_count, [], shuffle)
    }

    /// Creates a [`FieldBuilder`] of dimensions `width` × `height` with `mine_count` randomly placed mines,
    /// guaranteeing that `zero_cell` and all of its neighbors are mine-free (so `zero_cell` has count `0`).
    ///
    /// Like `FieldBuilder::random`, the `shuffle` closure permutes the slice of available (non-reserved)
    /// cells so callers can supply their own random number generator.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minesweeper_solver::{CellValue, FieldBuilder, Point};
    /// let zero_pt = Point { x: 2, y: 2 };
    /// let mut builder =
    ///     FieldBuilder::random_with_zero_region(5, 5, 3, zero_pt, |cells| cells.reverse());
    /// builder.expand_points([zero_pt]);
    /// let field = builder.into_mine_field();
    /// assert_eq!(field.cell(zero_pt).value, CellValue::Safe(0));
    /// ```
    pub fn random_with_zero_region(
        width: u32,
        height: u32,
        mine_count: u32,
        zero_cell: Point,
        shuffle: impl FnMut(&mut [i8]),
    ) -> Self {
        assert!(
            zero_cell.x < width && zero_cell.y < height,
            "zero_cell ({}, {}) is out of bounds for {width}x{height}",
            zero_cell.x,
            zero_cell.y
        );
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
        let mut field = vec![MineOrCount::ZERO; len];
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
        field[..mine_count as usize].fill(MineOrCount::MINE);

        let ptr = field.as_mut_ptr() as *mut i8;
        // SAFETY:
        // 1. `MineOrCount` is `#[repr(transparent)]` over `i8`, guaranteeing identical memory layout, size, and alignment.
        // 2. `ptr` is non-null and valid for writes up to `end_shuffle` elements (`end_shuffle <= field.len()`).
        // 3. No other references to `field` exist while `bytes` is borrowed by `shuffle`.
        let bytes = unsafe { slice::from_raw_parts_mut(ptr, end_shuffle) };
        shuffle(bytes);

        for (i, reserve_idx) in (end_shuffle..).zip(reserved_indices) {
            field.swap(i, reserve_idx.0 as usize);
        }

        Self::from_vec(width, height, field)
    }

    /// Accumulates neighbor mine counts across `field` and initializes the [`FieldBuilder`].
    fn from_vec(width: u32, height: u32, mut field: Vec<MineOrCount>) -> Self {
        assert_eq!(width as usize * height as usize, field.len());

        let mut mine_count = 0;

        for idx in 0..field.len() {
            if field[idx].is_mine() {
                let idx = CellIdx(idx as u32);
                mine_count += 1;
                let pt = coordinates(idx, width);
                for Neighbor { neighbor, .. } in
                    Cell::neighbor_iter(idx, width, DirectionSet::all_neighbors(pt, width, height))
                {
                    if !field[neighbor].is_mine() {
                        field[neighbor] = MineOrCount(field[neighbor].count() + 1);
                    }
                }
            }
        }

        Self {
            mines: field.into_boxed_slice(),
            field: MineField::with_mine_count(width, height, mine_count).unwrap(),
        }
    }

    /// Returns a mutable reference to the underlying [`MineField`].
    pub fn mine_field(&mut self) -> &mut MineField {
        &mut self.field
    }

    /// Consumes the builder and returns the underlying [`MineField`].
    pub fn into_mine_field(self) -> MineField {
        self.field
    }

    /// Reveals the ground-truth [`CellValue`](crate::CellValue)s for the specified `points` on the underlying [`MineField`].
    pub(super) fn expose_cells(&mut self, points: impl IntoIterator<Item = CellIdx>) -> &mut Self {
        self.field
            .pending
            .extend(points.into_iter().map(|idx| (idx, self.mines[idx].into())));
        self.field.solve();
        self
    }

    pub fn expose_points(&mut self, points: impl IntoIterator<Item = Point>) -> &mut Self {
        let width = self.field.width;
        self.expose_cells(points.into_iter().map(|p| cell_idx(p, width)))
    }

    /// Reveals `points` on the underlying [`MineField`] and recursively flood-fills any revealed
    /// `0`-count cells to expose their entire connected zero-region and its adjacent clue boundary.
    pub(super) fn expand_cells(&mut self, ids: impl IntoIterator<Item = CellIdx>) -> &mut Self {
        let mut all_ids =
            HashSet::with_capacity_and_hasher(self.mines.len(), FastNumberHasherBuilder);
        let mut zero_queue = Vec::with_capacity(self.mines.len());

        for idx in ids {
            all_ids.insert(idx);
            if self.mines[idx].count() == 0 {
                zero_queue.push(idx);
            }
        }
        while let Some(idx) = zero_queue.pop() {
            let cell = self.field.cells[idx];
            for Neighbor { neighbor, .. } in
                Cell::neighbor_iter(idx, self.field.width, cell.neighbors())
            {
                if all_ids.insert(neighbor) && self.mines[neighbor].count() == 0 {
                    zero_queue.push(neighbor);
                }
            }
        }

        self.expose_cells(all_ids)
    }

    /// Reveals `points` on the underlying [`MineField`] and recursively flood-fills any revealed
    /// `0`-count cells to expose their entire connected zero-region and its adjacent clue boundary.
    pub fn expand_points(&mut self, points: impl IntoIterator<Item = Point>) -> &mut Self {
        let width = self.field.width;
        self.expand_cells(points.into_iter().map(|p| cell_idx(p, width)))
    }

    // Selects a random member of a live partition to expose.
    #[cfg(test)]
    pub(super) fn hint(
        &self,
        members: &[CellIdx],
        mut idx_selector: impl FnMut(usize) -> usize,
    ) -> Option<CellIdx> {
        let members: Vec<_> = members
            .iter()
            .filter(|&i| self.mines[i.0 as usize] != MineOrCount::MINE)
            .filter_map(|&i| self.field.cells[i].value().is_unknown().then_some(i))
            .collect();
        (!members.is_empty()).then_some(members[idx_selector(members.len())])
    }
}

impl Display for FieldBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut idx = 0;
        for _ in 0..self.field.height {
            for _ in 0..self.field.width {
                match self.mines[idx] {
                    MineOrCount::MINE => write!(f, "*"),
                    _ => write!(f, "."),
                }?;
                idx += 1;
            }
            writeln!(f)?;
        }
        for _ in 0..self.field.width {
            write!(f, "=")?;
        }
        writeln!(f)?;

        Display::fmt(&self.field, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(debug_assertions)]
    use crate::solver::display::ColoredMineField;
    use crate::solver::{FieldArgs, Metrics};
    use rand::{
        Rng, RngExt, SeedableRng, rng,
        rngs::StdRng,
        seq::{IteratorRandom, SliceRandom},
    };

    fn shuffle(rng: &mut impl Rng) -> impl FnMut(&mut [i8]) {
        |cells| cells.shuffle(rng)
    }

    fn selector(rng: &mut impl Rng) -> impl FnMut(usize) -> usize {
        |max| rng.random_range(0..max)
    }

    #[test]
    fn test_field_builder_from_points_and_expose_cells() {
        // 3x2 board with mines at (0, 0) and (2, 1)
        let mut builder =
            FieldBuilder::from_points(3, 2, [Point { x: 0, y: 0 }, Point { x: 2, y: 1 }]);
        assert_eq!(builder.mine_field().width(), 3);
        assert_eq!(builder.mine_field().height(), 2);
        assert_eq!(builder.mine_field().mine_count(), Some(2));

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
        builder.expose_points([Point { x: 1, y: 0 }, Point { x: 0, y: 0 }]);
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
        // 3x3 board with 8 mines surrounding center (1, 1), verifying '8' and mine specifier '*'
        let mut builder = FieldBuilder::from_spec(
            "***
             *8*
             ***",
        );
        assert_eq!(builder.mine_field().width(), 3);
        assert_eq!(builder.mine_field().height(), 3);
        assert_eq!(builder.mine_field().mine_count(), Some(8));

        builder.expose_points([Point { x: 1, y: 1 }]);
        assert_eq!(
            builder.mine_field().cell(Point { x: 1, y: 1 }).value,
            UserCellValue::Safe(8)
        );

        // Also test a valid spec with mixed counts (0, 1, 2, 7) and '_'
        let mut builder7 = FieldBuilder::from_spec(
            "***
             *7*
             *_*
             121
             000",
        );
        builder7.expose_points([
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
        FieldBuilder::from_spec("2*.");
    }

    #[test]
    fn test_field_builder_expand_cells_flood_fill() {
        // 4x3 board with a single mine at bottom-right (3, 2):
        // Ground-truth counts:
        // (0,0)=0  (1,0)=0  (2,0)=0  (3,0)=0
        // (0,1)=0  (1,1)=0  (2,1)=1  (3,1)=1
        // (0,2)=0  (1,2)=0  (2,2)=1  (3,2)=*
        let mut builder = FieldBuilder::from_spec(
            "....
             ..11
             ..1*",
        );

        // Expanding from a non-zero cell (2, 1) only exposes that single cell
        builder.expand_points([Point { x: 2, y: 1 }]);
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
        builder.expand_points([Point { x: 0, y: 0 }]);
        let field = builder.into_mine_field();
        for y in 0..3 {
            for x in 0..4 {
                let pt = Point { x, y };
                if pt == (Point { x: 3, y: 2 }) {
                    assert_eq!(field.cell(pt).value, UserCellValue::Unknown);
                    assert_eq!(
                        field
                            .recent_inferred_mine_cells()
                            .map(|i| coordinates(i, 4))
                            .collect::<Vec<Point>>(),
                        vec![pt]
                    );
                } else {
                    assert!(matches!(field.cell(pt).value, UserCellValue::Safe(_)));
                }
            }
        }
    }

    #[test]
    fn test_field_builder_random_and_zero_region() {
        let random_builder = FieldBuilder::random(6, 5, 7, shuffle(&mut rng()));
        assert_eq!(random_builder.field.width(), 6);
        assert_eq!(random_builder.field.height(), 5);
        assert_eq!(random_builder.field.mine_count(), Some(7));
        let actual_mines = random_builder
            .mines
            .iter()
            .filter(|&&v| v == MineOrCount::MINE)
            .count();
        assert_eq!(actual_mines, 7);

        let zero_pt = Point { x: 2, y: 2 };
        let mut zero_builder =
            FieldBuilder::random_with_zero_region(6, 5, 10, zero_pt, shuffle(&mut rng()));
        zero_builder.expand_points([zero_pt]);
        let field = zero_builder.into_mine_field();
        assert_eq!(field.cell(zero_pt).value, UserCellValue::Safe(0));
        for Neighbor { neighbor, .. } in
            zero_pt.neighbors(DirectionSet::all_neighbors(zero_pt, 6, 5))
        {
            assert!(matches!(field.cell(neighbor).value, UserCellValue::Safe(_)));
        }
    }

    #[test]
    fn test_field_builder_from_spec_two_part() {
        let mut builder = FieldBuilder::from_spec(
            "*...
             ....
             ...*
             ====
             .1..
             ....
             ..1.",
        );

        assert_eq!(builder.field.width(), 4);
        assert_eq!(builder.field.height(), 3);
        assert_eq!(builder.field.mine_count(), Some(2));

        // Ground-truth check
        assert_eq!(builder.mines[0], MineOrCount::MINE); // (0, 0)
        assert_eq!(builder.mines[1].count(), 1); // (1, 0) has 1 neighbor mine at (0, 0)
        assert_eq!(builder.mines[11], MineOrCount::MINE); // (3, 2)
        assert_eq!(builder.mines[10].count(), 1); // (2, 2) has 1 neighbor mine at (3, 2)

        // Underlying MineField state check: pre-revealed cells from lower spec
        assert_eq!(
            builder.field.cell(Point { x: 1, y: 0 }).value,
            UserCellValue::Safe(1)
        );
        assert_eq!(
            builder.field.cell(Point { x: 2, y: 2 }).value,
            UserCellValue::Safe(1)
        );
        assert_eq!(
            builder.field.cell(Point { x: 0, y: 0 }).value,
            UserCellValue::Unknown
        );
        assert_eq!(
            builder.field.cell(Point { x: 3, y: 2 }).value,
            UserCellValue::Unknown
        );

        // Exposing remaining cells verifies ground-truth consistency
        builder.expose_points([Point { x: 1, y: 1 }]);
        assert_eq!(
            builder.field.cell(Point { x: 1, y: 1 }).value,
            UserCellValue::Safe(1)
        );
    }

    #[test]
    fn test_field_builder_from_spec_display_round_trip() {
        let mut builder = FieldBuilder::from_spec(
            "*..
             ...
             ..*
             ===
             .1.
             ...
             ...",
        );
        // Expose a cell to create a richer state with safe and inferred cells
        builder.expand_points([Point { x: 0, y: 1 }]);

        let displayed = builder.to_string();
        // The display output should contain ground-truth mines, the '===' separator, and field cells
        assert!(displayed.contains("==="));

        let round_tripped = FieldBuilder::from_spec(&displayed);
        assert_eq!(round_tripped.field.width(), builder.field.width());
        assert_eq!(round_tripped.field.height(), builder.field.height());
        assert_eq!(round_tripped.field.mine_count(), builder.field.mine_count());
        assert_eq!(round_tripped.mines, builder.mines);

        for y in 0..builder.field.height() {
            for x in 0..builder.field.width() {
                let pt = Point { x, y };
                assert_eq!(round_tripped.field.cell(pt), builder.field.cell(pt));
            }
        }
        assert_eq!(round_tripped.to_string(), displayed);
    }

    #[test]
    fn test_field_builder_from_spec_various_separators_and_inferred_markers() {
        // Test with different separator lengths (e.g. single '=' or long '========')
        let builder_short_sep = FieldBuilder::from_spec(
            "*.*
             ...
             =
             .2.
             ...",
        );
        assert_eq!(builder_short_sep.field.mine_count(), Some(2));
        assert_eq!(
            builder_short_sep.field.cell(Point { x: 1, y: 0 }).value,
            UserCellValue::Safe(2)
        );

        // Test with inferred display characters ('x' and '#') in the lower spec
        let builder_inferred = FieldBuilder::from_spec(
            "*..
             ...
             ==========
             x#.
             ...",
        );
        assert_eq!(builder_inferred.field.mine_count(), Some(1));
        // 'x' and '#' are parsed as Unknown, leaving them available for solver re-inference
        assert_eq!(
            builder_inferred.field.cell(Point { x: 0, y: 0 }).value,
            UserCellValue::Unknown
        );
        assert_eq!(
            builder_inferred.field.cell(Point { x: 1, y: 0 }).value,
            UserCellValue::Unknown
        );
    }

    #[test]
    fn test_field_builder_from_spec_single_part_preserves_unrevealed() {
        // Without an '=' separator line, all cells in the underlying MineField remain Unknown
        let builder = FieldBuilder::from_spec(
            "*..
             .*.
             ..*",
        );
        assert_eq!(builder.field.width(), 3);
        assert_eq!(builder.field.height(), 3);
        assert_eq!(builder.field.mine_count(), Some(3));
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(
                    builder.field.cell(Point { x, y }).value,
                    UserCellValue::Unknown
                );
            }
        }
    }

    #[cfg(test)]
    fn validate_incremental_field(inc_field: &MineField, validation_metrics: &mut Metrics) {
        let fresh_field = inc_field.rebuild_from_scratch();
        validation_metrics.tally_game(&fresh_field.metrics);

        // validate cells are identical, aside from a partition_id remapping
        for ((idx, inc_cell), fresh_cell) in inc_field
            .cells
            .iter()
            .copied()
            .enumerate()
            .zip(fresh_field.cells.iter().copied())
        {
            if inc_cell != fresh_cell {
                #[cfg(debug_assertions)]
                {
                    eprintln!("Incremental:");
                    ColoredMineField::eprintln(inc_field);
                    eprintln!("Fresh:");
                    ColoredMineField::eprintln(&fresh_field);
                }
                panic!("Cells [{idx}] don't match: {inc_cell}, {fresh_cell}");
            }
        }
    }

    #[test]
    #[ignore]
    fn test_multiple_random_incremental_play_through() {
        let rng = &mut rng();

        let mut metrics = Metrics::default();
        let mut validation_metrics = Metrics::default();

        let count = 100;
        let width = 30;
        let height = 24;
        let mine_count = 130;

        for _seed in 0..count {
            // Uncomment below for some determinism
            let seed = rng.next_u64(); // use _seed for determinism
            eprintln!("Using random seed: {seed}");
            test_single_random_incremental_play_through(
                width,
                height,
                mine_count,
                &mut metrics,
                &mut validation_metrics,
                &mut StdRng::seed_from_u64(seed),
            );
        }

        println!("For incremental:\n{metrics:#}");
        println!("For validation:\n{validation_metrics:#}");
    }

    fn test_single_random_incremental_play_through(
        width: u32,
        height: u32,
        mine_count: u32,
        metrics: &mut Metrics,
        validation_metrics: &mut Metrics,
        rng: &mut impl Rng,
    ) {
        let start_pt = Point {
            x: rng.random_range(0..width),
            y: rng.random_range(0..height),
        };
        let mut builder = FieldBuilder::random_with_zero_region(
            width,
            height,
            mine_count,
            start_pt,
            shuffle(rng),
        );
        builder.expand_points([start_pt]);

        let mut loop_count = 0;
        loop {
            #[cfg(feature = "debug")]
            {
                use crate::solver::display::DetailMineField;

                DetailMineField::eprintln(&builder.field);
                ColoredMineField::println(&builder.field);
            }
            let inferred: Vec<_> = builder.field.recent_inferred_safe_cells().collect();
            if inferred.is_empty() {
                let partition = builder
                    .field
                    .partitions
                    .iter()
                    .map(|(&idx, &len)| (idx, len))
                    .choose(rng);
                if let Some((member, len)) = partition {
                    let members =
                        MineField::members(member, len, &mut FieldArgs::new(&mut builder.field));
                    let idx = builder.hint(&members, selector(rng)).unwrap_or_else(|| {
                        panic!(
                            "Didn't get a hint from members: {:?}",
                            members
                                .iter()
                                .map(|i| i.0.to_string())
                                .collect::<Vec<_>>()
                                .join(",")
                        )
                    });
                    if builder.field.cells[idx].value() != CellValue::UNKNOWN {
                        eprintln!(
                            "Members: [{:?}]",
                            members
                                .iter()
                                .map(|i| i.0.to_string())
                                .collect::<Vec<_>>()
                                .join(",")
                        );
                        eprintln!("Hinted member is not UNKNOWN: {idx}");
                        panic!();
                    }
                    assert_eq!(builder.field.cells[idx].value(), CellValue::UNKNOWN);
                    builder.expand_cells([idx]);
                    assert_ne!(builder.field.cells[idx].value(), CellValue::UNKNOWN);
                } else {
                    break;
                }
            } else {
                for &i in inferred.iter() {
                    assert_eq!(builder.field.cells[i].value(), CellValue::INFERRED_SAFE);
                }
                builder.expand_cells(inferred);
            }

            // compare against a fresh (non incremental) MineField
            validate_incremental_field(&builder.field, validation_metrics);

            loop_count += 1;
            if loop_count >= width * height {
                panic!("Loop count exceeded limit: {loop_count}");
            }
        }
        metrics.tally_game(&builder.field.metrics);
    }
}

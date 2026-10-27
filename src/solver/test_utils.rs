//! Test utilities for constructing and updating [`MineField`] instances from compact ASCII specs.
//!
//! # Specification Format
//!
//! Both [`create_minefield`] and [`update_minefield`] accept an iterator of string slices
//! representing rows of the board (top-to-bottom, with `y = 0` first and `x = 0` leftmost).
//! Strings containing embedded newline (`\n`) characters are automatically split into separate rows.
//!
//! Each character in a row maps to a coordinate `(x, y)` as follows:
//! - `'0'..='8'`: Sets the cell to [`CellValue::Count(n)`](CellValue::Count).
//! - `'*'`, `'X'`, or `'x'`: Sets the cell to [`CellValue::Mine`].
//! - `'_'`: Explicitly sets the cell to [`CellValue::Unknown`] (useful with [`update_minefield`]
//!   to revert a previously known cell back to unknown).
//! - `'.'` or `' '`: Skips the cell ("do nothing"). During [`create_minefield`], the coordinate still
//!   contributes to the board dimensions and defaults to [`CellValue::Unknown`]; during
//!   [`update_minefield`], the existing cell value at `(x, y)` is left untouched.
//! - Any other character panics with `"Invalid field character"`.

use crate::{Cell, CellValue, MineField, Point};

use std::cmp::max;

/// Creates a new [`MineField`] with dimensions inferred from `field_spec` and populates its initial cells.
///
/// The board's `width` and `height` are determined by the maximum `x + 1` and `y + 1` coordinates
/// encountered across all rows in `field_spec` (including `'.'` and `' '` characters).
///
/// # Examples
///
/// ```ignore
/// // Creates a 4x2 MineField with 1 total mine:
/// // - Count(1) at (1, 0)
/// // - Mine at (2, 0)
/// // - Count(1) at (3, 1)
/// // - Unknown everywhere else
/// let field = create_minefield([".1*.", "...1"], 1);
/// assert_eq!(field.width(), 4);
/// assert_eq!(field.height(), 2);
/// assert_eq!(field.cell(Point { x: 1, y: 0 }).value, CellValue::Count(1));
/// assert_eq!(field.cell(Point { x: 2, y: 0 }).value, CellValue::Mine);
/// assert_eq!(field.cell(Point { x: 3, y: 1 }).value, CellValue::Count(1));
/// ```
///
/// # Panics
///
/// Panics if:
/// - `field_spec` contains any character outside `'0'..='8'`, `'*'`, `'X'`, `'x'`, `'_'`, `'.'`, `' '`, or `'\n'`.
/// - The inferred board dimensions and `mine_count` fail [`MineField::new`] validation (e.g., an empty
///   spec or `mine_count >= width * height`).
#[track_caller]
pub(super) fn create_minefield<'a>(
    field_spec: impl IntoIterator<Item = &'a str>,
    mine_count: u32,
) -> MineField {
    let cells: Vec<_> = spec_to_iter(field_spec).collect();

    let size = cells
        .iter()
        .map(|(x, y, _)| (x + 1, y + 1))
        .fold((0, 0), |acc, r| (max(acc.0, r.0), max(acc.1, r.1)));

    let mut field = MineField::new(size.0, size.1, mine_count).unwrap();
    field.set_cells(to_cell_iter(cells)).unwrap();

    field
}

/// Applies a batch of cell updates to an existing [`MineField`] using the ASCII `field_spec` format.
///
/// Coordinates with `'.'` or `' '` are filtered out and left unmodified on `field`. Coordinates
/// with `'_'` are explicitly set to [`CellValue::Unknown`].
///
/// # Examples
///
/// ```ignore
/// let mut field = create_minefield(["1..1", "...."], 2);
///
/// // Reveal (1, 0) as Count(2) and flag (2, 1) as a Mine, leaving all '.' cells untouched:
/// update_minefield(&mut field, [".2", "..*"]);
/// assert_eq!(field.cell(Point { x: 0, y: 0 }).value, CellValue::Count(1));
/// assert_eq!(field.cell(Point { x: 1, y: 0 }).value, CellValue::Count(2));
/// assert_eq!(field.cell(Point { x: 2, y: 1 }).value, CellValue::Mine);
///
/// // Revert (1, 0) back to CellValue::Unknown using '_':
/// update_minefield(&mut field, ["._"]);
/// assert_eq!(field.cell(Point { x: 1, y: 0 }).value, CellValue::Unknown);
/// ```
///
/// # Panics
///
/// Panics if:
/// - `field_spec` contains an invalid character.
/// - Any non-skipped coordinate in `field_spec` lies outside `field`'s dimensions.
#[track_caller]
pub(super) fn update_minefield<'a>(
    field: &mut MineField,
    field_spec: impl IntoIterator<Item = &'a str>,
) {
    field.set_cells(cells_from_spec(field_spec)).unwrap();
}

/// Parses `field_spec` into an iterator of [`Cell`] updates, omitting skipped (`'.'` and `' '`) positions.
fn cells_from_spec<'a>(
    field_spec: impl IntoIterator<Item = &'a str>,
) -> impl Iterator<Item = Cell> {
    to_cell_iter(spec_to_iter(field_spec))
}

/// Converts `(x, y, Option<CellValue>)` tuples into [`Cell`]s, filtering out `None` entries.
fn to_cell_iter(
    iter: impl IntoIterator<Item = (u32, u32, Option<CellValue>)>,
) -> impl Iterator<Item = Cell> {
    iter.into_iter().filter_map(|(x, y, v)| {
        v.map(|value| Cell {
            point: Point { x, y },
            value,
        })
    })
}

/// Parses `field_spec` into `(x, y, Option<CellValue>)` tuples for every character position in the spec.
fn spec_to_iter<'a>(
    field_spec: impl IntoIterator<Item = &'a str>,
) -> impl Iterator<Item = (u32, u32, Option<CellValue>)> {
    fn char_to_value(c: char) -> Option<CellValue> {
        match c {
            '*' | 'X' | 'x' => Some(CellValue::Mine),
            n @ '0'..='8' => Some(CellValue::Count(n as i8 - b'0' as i8)),
            '_' => Some(CellValue::Unknown),
            ' ' | '.' => None,
            _ => panic!("Invalid field character: {c:?}"),
        }
    }

    field_spec
        .into_iter()
        .flat_map(|s| s.split("\n"))
        .enumerate()
        .flat_map(|(y, l)| {
            l.chars()
                .map(char_to_value)
                .enumerate()
                .map(move |(x, value)| (x as u32, y as u32, value))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::cell::InferredValue;

    #[test]
    fn test_create_minefield_basic_example() {
        // " 1* \n   1" -> 4x2 field with:
        // - Count(1) at (1, 0)
        // - Mine at (2, 0)
        // - Count(1) at (3, 1)
        let field = create_minefield([" 1* ", "   1"], 1);
        assert_eq!(field.width(), 4);
        assert_eq!(field.height(), 2);
        assert_eq!(field.mine_count(), 1);

        assert_eq!(field.cell(Point { x: 0, y: 0 }).value, CellValue::Unknown);
        assert_eq!(field.cell(Point { x: 1, y: 0 }).value, CellValue::Count(1));
        assert_eq!(field.cell(Point { x: 2, y: 0 }).value, CellValue::Mine);
        assert_eq!(field.cell(Point { x: 3, y: 0 }).value, CellValue::Unknown);

        assert_eq!(field.cell(Point { x: 0, y: 1 }).value, CellValue::Unknown);
        assert_eq!(field.cell(Point { x: 1, y: 1 }).value, CellValue::Unknown);
        assert_eq!(field.cell(Point { x: 2, y: 1 }).value, CellValue::Unknown);
        assert_eq!(field.cell(Point { x: 3, y: 1 }).value, CellValue::Count(1));

        // Verify the solver ran on the populated cells:
        // Both (1, 0)=Count(1) and (3, 1)=Count(1) are satisfied by the Mine at (2, 0),
        // so all remaining unknown cells are inferred Open!
        for pt in [
            Point { x: 0, y: 0 },
            Point { x: 3, y: 0 },
            Point { x: 0, y: 1 },
            Point { x: 1, y: 1 },
            Point { x: 2, y: 1 },
        ] {
            let idx = pt.to_index(field.width());
            assert_eq!(field.cells()[idx].inferred_value(), InferredValue::Open);
        }
    }

    #[test]
    fn test_create_minefield_all_supported_characters() {
        // Test '0'..='8', '*', 'X', 'x', ' ', '.', and '_'
        let field = create_minefield(["01234", "5678*", "Xx._ "], 3);
        assert_eq!(field.width(), 5);
        assert_eq!(field.height(), 3);
        assert_eq!(field.mine_count(), 3);

        for x in 0..5u32 {
            assert_eq!(
                field.cell(Point { x, y: 0 }).value,
                CellValue::Count(x as i8)
            );
        }
        for x in 0..4u32 {
            assert_eq!(
                field.cell(Point { x, y: 1 }).value,
                CellValue::Count((x + 5) as i8)
            );
        }
        assert_eq!(field.cell(Point { x: 4, y: 1 }).value, CellValue::Mine);

        // 'X' and 'x' both produce CellValue::Mine
        assert_eq!(field.cell(Point { x: 0, y: 2 }).value, CellValue::Mine);
        assert_eq!(field.cell(Point { x: 1, y: 2 }).value, CellValue::Mine);
        // '.' and ' ' (filtered out) and '_' (explicit Unknown) all result in Unknown on a new board
        for x in 2..5u32 {
            assert_eq!(field.cell(Point { x, y: 2 }).value, CellValue::Unknown);
        }
    }

    #[test]
    fn test_cells_from_spec_and_update_minefield() {
        // Verify ' ' and '.' are filtered out while '_', '*', 'X', 'x', and digits emit Cells
        let cells: Vec<Cell> = cells_from_spec([" .1_Xx*"]).collect();
        assert_eq!(
            cells,
            vec![
                Cell {
                    point: Point { x: 2, y: 0 },
                    value: CellValue::Count(1),
                },
                Cell {
                    point: Point { x: 3, y: 0 },
                    value: CellValue::Unknown,
                },
                Cell {
                    point: Point { x: 4, y: 0 },
                    value: CellValue::Mine,
                },
                Cell {
                    point: Point { x: 5, y: 0 },
                    value: CellValue::Mine,
                },
                Cell {
                    point: Point { x: 6, y: 0 },
                    value: CellValue::Mine,
                },
            ]
        );

        // Start with a 4x2 field where (0, 0)=Count(1) and (2, 0)=Count(2)
        let mut field = create_minefield(["1.2.", "...."], 2);
        assert_eq!(field.cell(Point { x: 0, y: 0 }).value, CellValue::Count(1));
        assert_eq!(field.cell(Point { x: 2, y: 0 }).value, CellValue::Count(2));

        // Update with '.' at (0, 0) and (2, 0) (preserving their values),
        // 'X' at (1, 0), 'x' at (3, 1), and '1' at (0, 1)
        update_minefield(&mut field, [".X..", "1..x"]);
        assert_eq!(field.cell(Point { x: 0, y: 0 }).value, CellValue::Count(1));
        assert_eq!(field.cell(Point { x: 1, y: 0 }).value, CellValue::Mine);
        assert_eq!(field.cell(Point { x: 2, y: 0 }).value, CellValue::Count(2));
        assert_eq!(field.cell(Point { x: 0, y: 1 }).value, CellValue::Count(1));
        assert_eq!(field.cell(Point { x: 3, y: 1 }).value, CellValue::Mine);

        // Use '_' in update_minefield to explicitly revert (1, 0) and (2, 0) back to Unknown,
        // while '.' at (0, 0) preserves Count(1)
        update_minefield(&mut field, [".__."]);
        assert_eq!(field.cell(Point { x: 0, y: 0 }).value, CellValue::Count(1));
        assert_eq!(field.cell(Point { x: 1, y: 0 }).value, CellValue::Unknown);
        assert_eq!(field.cell(Point { x: 2, y: 0 }).value, CellValue::Unknown);
    }

    #[test]
    fn test_create_minefield_ragged_lines_and_1d_strips() {
        // Ragged lines: row 0 has length 2, row 1 has length 5 -> produces a 5x2 field
        // where missing trailing cells on row 0 default to Unknown.
        let ragged = create_minefield(["1*", "....2"], 1);
        assert_eq!(ragged.width(), 5);
        assert_eq!(ragged.height(), 2);
        assert_eq!(ragged.cell(Point { x: 0, y: 0 }).value, CellValue::Count(1));
        assert_eq!(ragged.cell(Point { x: 1, y: 0 }).value, CellValue::Mine);
        assert_eq!(ragged.cell(Point { x: 4, y: 0 }).value, CellValue::Unknown);
        assert_eq!(ragged.cell(Point { x: 4, y: 1 }).value, CellValue::Count(2));

        // 1x4 vertical strip
        let vertical = create_minefield(["1", ".", "*", "1"], 1);
        assert_eq!(vertical.width(), 1);
        assert_eq!(vertical.height(), 4);
        assert_eq!(
            vertical.cell(Point { x: 0, y: 0 }).value,
            CellValue::Count(1)
        );
        assert_eq!(
            vertical.cell(Point { x: 0, y: 1 }).value,
            CellValue::Unknown
        );
        assert_eq!(vertical.cell(Point { x: 0, y: 2 }).value, CellValue::Mine);
        assert_eq!(
            vertical.cell(Point { x: 0, y: 3 }).value,
            CellValue::Count(1)
        );
    }

    #[test]
    #[should_panic(expected = "Invalid field character")]
    fn test_create_minefield_invalid_char_9_panics() {
        create_minefield(["19*"], 1);
    }

    #[test]
    #[should_panic(expected = "Invalid field character")]
    fn test_create_minefield_invalid_char_letter_panics() {
        create_minefield(["1Z*"], 1);
    }

    #[test]
    #[should_panic]
    fn test_create_minefield_empty_spec_panics() {
        create_minefield([""], 0);
    }
}

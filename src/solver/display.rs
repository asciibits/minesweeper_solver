//! Terminal pretty-printing and [`Display`] formatting for [`MineField`].

use crate::solver::cell::{InferredValue, UserCellValue, UserCellWithInference};

use super::MineField;
use colored::Colorize;
use std::{
    fmt::{self, Display, Formatter},
    ops::Deref,
};

struct ColoredMineField<'a>(&'a MineField);

impl Deref for ColoredMineField<'_> {
    type Target = MineField;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl Display for ColoredMineField<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.height == 0 {
            return Ok(());
        }

        print_pre_header(self, f)?;
        print_header(self, f)?;
        print_row(self, 0, f)?;
        for row in 1..self.height {
            print_separator(self, f)?;
            print_row(self, row, f)?;
        }
        print_footer(self, f)?;

        Ok(())
    }
}

/// Formats `field` as an ANSI-colored Unicode box-drawing grid.
///
/// # Errors
///
/// Returns [`fmt::Error`] only if formatting into the underlying buffer fails.
pub fn print_field(field: &MineField) -> Result<String, fmt::Error> {
    Ok(ColoredMineField(field).to_string())
}

/// Writes a single board row (prefixed by its starting linear cell index) and its cell values.
fn print_row(field: &MineField, row: u32, f: &mut Formatter<'_>) -> fmt::Result {
    if field.width == 0 {
        return Ok(());
    }

    write!(f, "{:4}", row * field.width)?;
    for col in 0..field.width {
        write!(f, "{}", "│ ".bold())?;
        let cell = field.cells()[col as usize + row as usize * field.width as usize];
        let user_value: UserCellWithInference = cell.value().into();
        let printable = match user_value {
            UserCellWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::None,
            } => " ".on_bright_black(),
            UserCellWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::Mine,
            } => " ".on_bright_red(),
            UserCellWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::Safe,
            } => " ".on_green(),
            UserCellWithInference {
                value: UserCellValue::Mine,
                inferred: _,
            } => " ".on_bright_red(),
            UserCellWithInference {
                value: UserCellValue::Safe(c),
                inferred: _,
            } => match c {
                1 => "1".bright_blue(),
                2 => "2".green(),
                3 => "3".red(),
                4 => "4".purple(),
                5 => "5".yellow(),
                6 => "6".cyan(),
                7 => "7".bright_white(),
                8 => "8".bright_black(),
                _ => " ".clear(),
            },
        };

        let printable = match cell.partition_id() {
            2 => printable.bold().on_yellow(),
            3 => printable.bold().on_magenta(),
            4 => printable.bold().on_white(),
            _ => printable,
        };

        write!(f, "{printable}")?;
        write!(f, " ")?;
    }
    writeln!(f, "{}", "│".bold())?;
    Ok(())
}

/// Writes the column coordinate header above the top border.
fn print_pre_header(field: &MineField, f: &mut Formatter<'_>) -> fmt::Result {
    if field.width == 0 {
        return Ok(());
    }
    write!(f, "     ")?;
    for i in 0..field.width {
        write!(f, "{:^4}", i)?;
    }
    writeln!(f)?;
    Ok(())
}

/// Writes the top Unicode box border (`┌───┬───┐`).
fn print_header(field: &MineField, f: &mut Formatter<'_>) -> fmt::Result {
    if field.width == 0 {
        return Ok(());
    }

    write!(f, "    ")?;
    write!(f, "{}", "┌───".bold())?;
    for _ in 1..field.width {
        write!(f, "{}", "┬───".bold())?;
    }
    writeln!(f, "{}", "┐".bold())?;
    Ok(())
}

/// Writes an interior row separator (`├───┼───┤`).
fn print_separator(field: &MineField, f: &mut Formatter<'_>) -> fmt::Result {
    if field.width == 0 {
        return Ok(());
    }

    write!(f, "    ")?;
    write!(f, "{}", "├───".bold())?;
    for _ in 1..field.width {
        write!(f, "{}", "┼───".bold())?;
    }
    writeln!(f, "{}", "┤".bold())?;
    Ok(())
}

/// Writes the bottom Unicode box border (`└───┴───┘`).
fn print_footer(field: &MineField, f: &mut Formatter<'_>) -> fmt::Result {
    if field.width == 0 {
        return Ok(());
    }

    write!(f, "    ")?;
    write!(f, "{}", "└───".bold())?;
    for _ in 1..field.width {
        write!(f, "{}", "┴───".bold())?;
    }
    writeln!(f, "{}", "┘".bold())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::solver::test_utils::FieldBuilder;

    use super::super::*;
    use super::*;

    fn create_test_field(
        width: u32,
        height: u32,
        mine_count: u32,
        zero_point: Point,
    ) -> FieldBuilder {
        let mut field_builder =
            FieldBuilder::random_with_zero_region(width, height, mine_count, zero_point);
        field_builder.expand_cells([zero_point]);

        field_builder
    }

    #[test]
    fn test_print() {
        let mut builder = create_test_field(30, 16, 99, Point { x: 10, y: 5 });
        println!("{}", ColoredMineField(builder.mine_field()));
        println!("{:#}", builder.mine_field());
        eprintln!("{:#}", DetailMineField(builder.mine_field()));
    }

    #[test]
    fn test_display_all_cell_states() {
        let field = MineField::create_minefield(["*0123", "45678"], 1);
        let rendered = print_field(&field).unwrap();
        assert_eq!(rendered, ColoredMineField(&field).to_string());
        assert!(rendered.contains("┌───"));
        assert!(rendered.contains("├───"));
        assert!(rendered.contains("└───"));
    }
}

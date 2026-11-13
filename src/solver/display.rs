//! Terminal pretty-printing and [`Display`] formatting for [`MineField`].

use crate::solver::cell::{InferredValue, UserCellValue, UserValueWithInference};

use super::MineField;
use colored::Colorize;
use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
    ops::Deref,
};

pub(super) struct ColoredMineField<'a>(&'a MineField);

impl<'a> ColoredMineField<'a> {
    #[allow(dead_code)]
    pub(super) fn eprintln(field: &'a MineField) {
        eprintln!("{}", ColoredMineField(field));
    }

    #[cfg(debug_assertions)]
    #[allow(dead_code)]
    pub(super) fn println(field: &'a MineField) {
        println!("{}", ColoredMineField(field));
    }
}

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

        // get the current partitions
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

/// Writes a single board row (prefixed by its starting linear cell index) and its cell values.
fn print_row(field: &MineField, row: u32, f: &mut Formatter<'_>) -> fmt::Result {
    if field.width == 0 {
        return Ok(());
    }

    write!(f, "{:4}", row * field.width)?;
    for col in 0..field.width {
        write!(f, "{}", "│ ".bold())?;
        let cell = field.cells[col as usize + row as usize * field.width as usize];
        let user_value: UserValueWithInference = cell.value().into();
        let printable = match user_value {
            UserValueWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::None,
            } => " ".on_bright_black(),
            UserValueWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::Mine,
            } => " ".on_bright_red(),
            UserValueWithInference {
                value: UserCellValue::Unknown,
                inferred: InferredValue::Safe,
            } => " ".on_green(),
            UserValueWithInference {
                value: UserCellValue::Mine,
                inferred: _,
            } => " ".on_bright_red(),
            UserValueWithInference {
                value: UserCellValue::Wall,
                inferred: _,
            } => " ".on_white(),
            UserValueWithInference {
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

        let partition_ids: HashMap<u8, usize> = field
            .partitions
            .keys()
            .copied()
            .map(|i| field.cells[i].partition_id())
            .enumerate()
            .map(|(i, pid)| (pid, i))
            .collect();
        let printable = if let Some(&i) = partition_ids.get(&cell.partition_id()) {
            match i % 2 {
                0 => printable.bold().on_yellow(),
                1 => printable.bold().on_magenta(),
                _ => unreachable!(),
            }
        } else {
            printable
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

#[derive(Debug)]
pub(super) struct DetailMineField<'a>(&'a MineField);

impl<'a> DetailMineField<'a> {
    #[allow(dead_code)]
    pub(crate) fn eprintln(field: &'a MineField) {
        eprintln!("{:#}", DetailMineField(field));
    }
}

fn write_comma_separated<I, T, F>(
    f: &mut fmt::Formatter<'_>,
    iter: I,
    mut format_item: F,
) -> fmt::Result
where
    I: IntoIterator<Item = T>,
    F: FnMut(&mut fmt::Formatter<'_>, T) -> fmt::Result,
{
    let mut first = true;
    for item in iter {
        if !first {
            write!(f, ",")?;
        }
        first = false;
        format_item(f, item)?;
    }
    Ok(())
}

impl<'a> Display for DetailMineField<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let spacer = if f.alternate() { "\n " } else { " " };
        write!(f, "MineField ({}x{}", self.0.width, self.0.height)?;
        if let Some(mine_count) = self.0.mine_count {
            write!(f, ", {mine_count}*")?;
        }
        write!(f, ") {{")?;
        for (idx, cell) in self.0.cells.iter().enumerate() {
            // Cells no longer carry their own index, so prefix it for debuggability.
            write!(f, "{spacer}{idx}: ")?;
            cell.fmt(f)?;
            write!(f, ",")?;
        }
        write!(
            f,
            "{spacer}next partition id: {},",
            self.0.next_partition_id
        )?;
        write!(f, "{spacer}pending: [")?;
        write_comma_separated(f, &self.0.pending, |f, c| {
            write!(f, "({}: {})", c.idx, c.value)
        })?;
        write!(f, "],")?;
        write!(f, "{spacer}unsolved: [")?;
        write_comma_separated(f, self.0.desolved_queue.iter(), |f, i| write!(f, "{i}"))?;
        write!(f, "],")?;
        write!(f, "{spacer}solved: [")?;
        write_comma_separated(f, self.0.solved_queue.iter(), |f, i| write!(f, "{i}"))?;
        write!(f, "],")?;
        write!(f, "{spacer}pending partition: [")?;
        write_comma_separated(f, self.0.pending_partition_queue.iter(), |f, i| {
            write!(f, "{i}")
        })?;
        write!(f, "],")?;
        write!(f, "{spacer}recent inferences: [")?;
        write_comma_separated(f, &self.0.recent_inferred, |f, i| write!(f, "{i}"))?;
        write!(f, "],")?;
        write!(f, "{spacer}recent invalid cells: [")?;
        write_comma_separated(f, &self.0.recent_invalid, |f, i| write!(f, "{i}"))?;
        write!(f, "],")?;
        write!(f, "{spacer}partitions: [")?;
        for (&idx, len) in self.0.partitions.iter() {
            write!(
                f,
                "{spacer}  {}: member: {idx}, len: {len}",
                self.0.cells[idx].partition_id()
            )?;
        }
        write!(f, "]")?;
        if f.alternate() {
            writeln!(f)?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use crate::solver::test_utils::FieldBuilder;

    use super::super::*;
    use super::*;
    use rand::{rng, seq::SliceRandom};

    fn create_test_field(
        width: u32,
        height: u32,
        mine_count: u32,
        zero_point: Point,
    ) -> FieldBuilder {
        let mut field_builder =
            FieldBuilder::random_with_zero_region(width, height, mine_count, zero_point, |cells| {
                cells.shuffle(&mut rng())
            });
        field_builder.expand_points([zero_point]);

        field_builder
    }

    #[test]
    fn test_print() {
        let mut builder = create_test_field(30, 16, 99, Point { x: 10, y: 5 });
        builder
            .mine_field()
            .set_cells([
                UserCell {
                    point: Point { x: 23, y: 0 },
                    value: UserCellValue::Wall,
                },
                UserCell {
                    point: Point { x: 25, y: 0 },
                    value: UserCellValue::Wall,
                },
                UserCell {
                    point: Point { x: 23, y: 1 },
                    value: UserCellValue::Wall,
                },
                UserCell {
                    point: Point { x: 24, y: 1 },
                    value: UserCellValue::Wall,
                },
                UserCell {
                    point: Point { x: 25, y: 1 },
                    value: UserCellValue::Wall,
                },
            ])
            .unwrap();
        println!("{}", ColoredMineField(builder.mine_field()));
        println!("{}", builder.mine_field());
        // eprintln!("{:#}", DetailMineField(builder.mine_field()));
    }

    #[test]
    fn test_detail_minefield_display() {
        let field = MineField::create("1.\n..");
        let details = format!("{:#}", DetailMineField(&field));
        assert!(details.contains("MineField (2x2)"));
        assert!(details.contains("pending: ["));
        assert!(details.contains("partitions: ["));
    }
}

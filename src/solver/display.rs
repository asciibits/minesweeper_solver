//! Terminal pretty-printing and [`Display`] formatting for [`MineField`].

use crate::solver::{
    CellIdx, CellValue,
    cell::{InferredValue, UserCellValue, UserValueWithInference},
};

use super::MineField;
use colored::Colorize;
use std::fmt::{self, Display, Formatter};

pub(super) struct ColoredMineField<'a> {
    cells: &'a [CellValue],
    width: u32,
    height: u32,
}

impl<'a> ColoredMineField<'a> {
    #[allow(dead_code)]
    pub(super) fn eprintln(field: &'a MineField) {
        eprintln!(
            "{}",
            ColoredMineField {
                cells: &field.cells,
                width: field.width_with_border,
                height: field.height_with_border
            }
        );
    }

    #[cfg(debug_assertions)]
    #[allow(dead_code)]
    pub(super) fn println(field: &'a MineField) {
        println!(
            "{}",
            ColoredMineField {
                cells: &field.cells,
                width: field.width_with_border,
                height: field.height_with_border
            }
        );
    }

    /// Writes a single board row (prefixed by its starting linear cell index) and its cell values.
    fn print_row(&self, row: u32, f: &mut Formatter<'_>) -> fmt::Result {
        if self.width == 0 {
            return Ok(());
        }

        write!(f, "{:4}", row * self.width)?;
        for col in 0..self.width {
            let idx = (col + row * self.width) as usize;
            if col == 0 || !self.cells[idx - 1].is_wall() || !self.cells[idx].is_wall() {
                write!(f, "{}", "│ ".bold())?;
            } else {
                write!(f, "{}", "  ".on_white())?;
            }
            let cell = self.cells[idx];
            let user_value: UserValueWithInference = cell.into();
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
                    value: UserCellValue::Opened(c),
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

            // let partition_ids: HashMap<PartitionId, usize> = field
            //     .partitions
            //     .keys()
            //     .copied()
            //     .map(|i| field.partition_ids[i])
            //     .enumerate()
            //     .map(|(i, pid)| (pid, i))
            //     .collect();
            // let printable = if let Some(&i) = partition_ids.get(&field.partition_ids[idx]) {
            //     match i % 2 {
            //         0 => printable.bold().on_yellow(),
            //         1 => printable.bold().on_magenta(),
            //         _ => unreachable!(),
            //     }
            // } else {
            //     printable
            // };

            write!(f, "{printable}")?;

            if col == self.width - 1 || !self.cells[idx].is_wall() || !self.cells[idx + 1].is_wall()
            {
                write!(f, "{}", " ".bold())?;
            } else {
                write!(f, "{}", " ".on_white())?;
            }
        }
        writeln!(f, "{}", "│".bold())?;
        Ok(())
    }

    /// Writes the column coordinate header above the top border.
    fn print_pre_header(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.width == 0 {
            return Ok(());
        }
        write!(f, "     ")?;
        for i in 0..self.width {
            write!(f, "{:^4}", i)?;
        }
        writeln!(f)?;
        Ok(())
    }

    /// Writes the top Unicode box border (`┌───┬───┐`).
    fn print_header(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.width == 0 {
            return Ok(());
        }

        write!(f, "    ")?;
        write!(f, "{}", "┌───".bold())?;
        for i in 1..self.width as usize {
            if self.cells[i - 1].is_wall() && self.cells[i].is_wall() {
                write!(f, "{}", "────".bold())?;
            } else {
                write!(f, "{}", "┬───".bold())?;
            }
        }
        writeln!(f, "{}", "┐".bold())?;
        Ok(())
    }
    /// Writes an interior row separator (`├───┼───┤`).
    fn print_separator(&self, row: u32, f: &mut Formatter<'_>) -> fmt::Result {
        if self.width == 0 {
            return Ok(());
        }

        write!(f, "    ")?;
        let mut idx = row as usize * self.width as usize;
        if self.cells[idx - self.width as usize].is_wall() && self.cells[idx].is_wall() {
            write!(f, "{}{} ", "│ ".bold(), " ".on_white())?;
        } else {
            write!(f, "{}", "├───".bold())?;
        }
        for _ in 1..self.width as usize {
            idx += 1;
            let nw = self.cells[idx - self.width as usize - 1].is_wall();
            let ne = self.cells[idx - self.width as usize].is_wall();
            let sw = self.cells[idx - 1].is_wall();
            let se = self.cells[idx].is_wall();
            let post_bold = if ne
                && se
                && idx + 1 < self.cells.len()
                && self.cells[idx - self.width as usize + 1].is_wall()
                && self.cells[idx + 1].is_wall()
            {
                " ".on_white()
            } else {
                " ".bold()
            };
            let lines = match (nw, ne, sw, se) {
                (false, false, false, false) => ("┼".bold(), "─".bold(), "─".bold(), "─".bold()),
                (false, false, false, true) => ("┼".bold(), "─".bold(), "─".bold(), "─".bold()),
                (false, false, true, false) => ("┼".bold(), "─".bold(), "─".bold(), "─".bold()),
                (false, false, true, true) => ("┴".bold(), "─".bold(), "─".bold(), "─".bold()),
                (false, true, false, false) => ("┼".bold(), "─".bold(), "─".bold(), "─".bold()),
                (false, true, false, true) => ("┤".bold(), " ".bold(), " ".on_white(), post_bold),
                (false, true, true, false) => ("┼".bold(), "─".bold(), "─".bold(), "─".bold()),
                (false, true, true, true) => ("┘".bold(), " ".bold(), " ".on_white(), post_bold),
                (true, false, false, false) => ("┼".bold(), "─".bold(), "─".bold(), "─".bold()),
                (true, false, false, true) => ("┼".bold(), "─".bold(), "─".bold(), "─".bold()),
                (true, false, true, false) => ("├".bold(), "─".bold(), "─".bold(), "─".bold()),
                (true, false, true, true) => ("└".bold(), "─".bold(), "─".bold(), "─".bold()),
                (true, true, false, false) => ("┬".bold(), "─".bold(), "─".bold(), "─".bold()),
                (true, true, false, true) => ("┐".bold(), " ".bold(), " ".on_white(), post_bold),
                (true, true, true, false) => ("┌".bold(), "─".bold(), "─".bold(), "─".bold()),
                (true, true, true, true) => {
                    (" ".on_white(), " ".on_white(), " ".on_white(), post_bold)
                }
            };
            write!(f, "{}{}{}{}", lines.0, lines.1, lines.2, lines.3)?;
        }
        if self.cells[idx - self.width as usize].is_wall() && self.cells[idx].is_wall() {
            write!(f, "{}", "│".bold())?;
        } else {
            writeln!(f, "{}", "┤".bold())?;
        }
        Ok(())
    }

    /// Writes the bottom Unicode box border (`└───┴───┘`).
    fn print_footer(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.width == 0 {
            return Ok(());
        }

        write!(f, "    ")?;
        write!(f, "{}", "└───".bold())?;
        let idx = self.cells.len() - self.width as usize - 2;
        for i in 1..self.width as usize {
            if self.cells[idx + i].is_wall() && self.cells[idx + i + 1].is_wall() {
                write!(f, "{}", "────".bold())?;
            } else {
                write!(f, "{}", "┴───".bold())?;
            }
        }
        writeln!(f, "{}", "┘".bold())?;
        Ok(())
    }
}

impl Display for ColoredMineField<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.height == 0 {
            return Ok(());
        }

        // get the current partitions
        self.print_pre_header(f)?;
        self.print_header(f)?;
        self.print_row(0, f)?;
        for row in 1..self.height {
            self.print_separator(row, f)?;
            self.print_row(row, f)?;
        }
        self.print_footer(f)?;

        Ok(())
    }
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
        let spacer = if f.alternate() { "\n  " } else { " " };
        let cell_spacer = if f.alternate() { "\n    " } else { " " };
        write!(
            f,
            "MineField ({}x{}",
            self.0.width(), self.0.height()
        )?;
        if let Some(mine_count) = self.0.mine_count {
            write!(f, ", {mine_count}*")?;
        }
        write!(f, ") {{")?;
        for (idx, (cell, flags, neighbors, pid)) in self.0.cell_data_iter().enumerate() {
            let idx = CellIdx(idx as u32);
            // Cells no longer carry their own index, so prefix it for debuggability.
            write!(f, "{spacer}{idx}: Cell {{{cell_spacer}value: {},", cell)?;
            write!(f, "{cell_spacer}unknown: {},", neighbors.unknown())?;
            write!(f, "{cell_spacer}opened: {},", neighbors.opened())?;
            write!(f, "{cell_spacer}mine: {},", neighbors.mine())?;
            write!(f, "{cell_spacer}inferred: {},", neighbors.inferred())?;
            write!(f, "{cell_spacer}flags: {flags},")?;
            write!(f, "{cell_spacer}pid: {pid},")?;
        }
        write!(f, "{spacer}}},")?;
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
                self.0.partition_ids[idx]
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
                UserCell {
                    point: Point { x: 23, y: 2 },
                    value: UserCellValue::Wall,
                },
                UserCell {
                    point: Point { x: 25, y: 2 },
                    value: UserCellValue::Wall,
                },
                UserCell {
                    point: Point { x: 23, y: 3 },
                    value: UserCellValue::Wall,
                },
                UserCell {
                    point: Point { x: 24, y: 3 },
                    value: UserCellValue::Wall,
                },
                UserCell {
                    point: Point { x: 25, y: 3 },
                    value: UserCellValue::Wall,
                },
                UserCell {
                    point: Point { x: 16, y: 0 },
                    value: UserCellValue::Wall,
                },
                UserCell {
                    point: Point { x: 17, y: 0 },
                    value: UserCellValue::Wall,
                },
                UserCell {
                    point: Point { x: 18, y: 0 },
                    value: UserCellValue::Wall,
                },
                UserCell {
                    point: Point { x: 16, y: 1 },
                    value: UserCellValue::Wall,
                },
                UserCell {
                    point: Point { x: 17, y: 1 },
                    value: UserCellValue::Wall,
                },
                UserCell {
                    point: Point { x: 18, y: 1 },
                    value: UserCellValue::Wall,
                },
            ])
            .unwrap();
        ColoredMineField::eprintln(builder.mine_field());
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

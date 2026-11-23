//! Terminal pretty-printing and [`Display`] formatting for [`MineField`].

use crate::solver::{
    CellIdx, PartitionId,
    cell::{InferredValue, UserCellValue, UserValueWithInference},
    coordinates,
};

use super::MineField;
use colored::Colorize;
use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
};

pub(super) struct ColoredMineField<'a> {
    field: &'a MineField,
}

impl<'a> ColoredMineField<'a> {
    #[allow(dead_code)]
    pub(super) fn eprintln(field: &'a MineField) {
        eprintln!("{}", ColoredMineField { field });
    }

    #[cfg(debug_assertions)]
    #[allow(dead_code)]
    pub(super) fn println(field: &'a MineField) {
        println!("{}", ColoredMineField { field });
    }

    /// Writes a single board row (prefixed by its starting linear cell index) and its cell values.
    fn print_row(
        &self,
        row: u32,
        f: &mut Formatter<'_>,
        partition_ids: &HashMap<PartitionId, usize>,
    ) -> fmt::Result {
        if self.field.width() == 0 {
            return Ok(());
        }

        let width = self.field.width_with_border;
        write!(f, "{:4} ", row * width)?;
        for col in 1..width {
            let idx = (col + row * width) as usize;
            if !self.field.cells[idx - 1].value.is_wall() || !self.field.cells[idx].value.is_wall()
            {
                write!(f, "{}", "│ ".bold())?;
            } else {
                write!(f, "{}", "  ".on_white())?;
            }
            let cell = self.field.cells[idx].value;
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
                    1 => "1".bright_blue().bold(),
                    2 => "2".green().bold(),
                    3 => "3".red().bold(),
                    4 => "4".purple().bold(),
                    5 => "5".yellow().bold(),
                    6 => "6".cyan().bold(),
                    7 => "7".bright_white().bold(),
                    8 => "8".bright_black().bold(),
                    _ => " ".clear().bold(),
                },
            };

            let printable = if let Some(&i) = partition_ids.get(&self.field.cells[idx].partition_id)
            {
                match i % 3 {
                    0 => printable.bold().on_bright_yellow(),
                    1 => printable.bold().on_magenta(),
                    2 => printable.bold().on_white(),
                    _ => unreachable!(),
                }
            } else {
                printable
            };

            write!(f, "{printable}")?;

            if col == width - 1
                || !self.field.cells[idx].value.is_wall()
                || !self.field.cells[idx + 1].value.is_wall()
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
        if self.field.width_with_border == 0 {
            return Ok(());
        }
        write!(f, "      ")?;
        for i in 1..self.field.width_with_border {
            write!(f, "{:^4}", i)?;
        }
        writeln!(f)?;
        Ok(())
    }

    /// Writes the top Unicode box border (`┌───┬───┐`).
    fn print_header(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let width = self.field.width_with_border;
        if width <= 1 {
            return Ok(());
        }

        write!(f, "     ")?;
        write!(f, "{}", "┌───".bold())?;
        for i in width as usize + 2..2 * width as usize {
            if self.field.cells[i - 1].value.is_wall() && self.field.cells[i].value.is_wall() {
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
        let width = self.field.width_with_border;
        if width == 0 {
            return Ok(());
        }

        write!(f, "     ")?;
        let mut idx = row as usize * width as usize + 1;
        if self.field.cells[idx - width as usize + 1].value.is_wall()
            && self.field.cells[idx + 1].value.is_wall()
        {
            write!(f, "{}{} ", "│ ".bold(), " ".on_white())?;
        } else {
            write!(f, "{}", "├───".bold())?;
        }
        for _ in 2..width as usize {
            idx += 1;
            let nw = self.field.cells[idx - width as usize - 1].value.is_wall();
            let ne = self.field.cells[idx - width as usize].value.is_wall();
            let sw = self.field.cells[idx - 1].value.is_wall();
            let se = self.field.cells[idx].value.is_wall();
            let post_bold = if ne
                && se
                && idx + 1 < self.field.cells.len()
                && self.field.cells[idx - width as usize + 1].value.is_wall()
                && self.field.cells[idx + 1].value.is_wall()
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
        if self.field.cells[idx - width as usize].value.is_wall()
            && self.field.cells[idx].value.is_wall()
        {
            write!(f, "{}", "│".bold())?;
        } else {
            writeln!(f, "{}", "┤".bold())?;
        }
        Ok(())
    }

    /// Writes the bottom Unicode box border (`└───┴───┘`).
    fn print_footer(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let width = self.field.width_with_border;
        if width == 0 {
            return Ok(());
        }

        write!(f, "     ")?;
        write!(f, "{}", "└───".bold())?;
        let idx = self.field.cells.len() - 2 * width as usize - 2;
        for i in 2..width as usize {
            if self.field.cells[idx + i].value.is_wall()
                && self.field.cells[idx + i + 1].value.is_wall()
            {
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
        let partition_ids: HashMap<PartitionId, usize> = self
            .field
            .partitions
            .keys()
            .enumerate()
            .map(|(i, &pid)| (pid, i))
            .collect();

        if self.field.height_with_border == 2 {
            return Ok(());
        }

        // get the current partitions
        self.print_pre_header(f)?;
        self.print_header(f)?;
        self.print_row(1, f, &partition_ids)?;
        for row in 2..self.field.height_with_border - 1 {
            self.print_separator(row, f)?;
            self.print_row(row, f, &partition_ids)?;
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
        write!(f, "MineField ({}x{}", self.0.width(), self.0.height())?;
        if let Some(mine_count) = self.0.mine_count {
            write!(f, ", {mine_count}*")?;
        }
        write!(f, ") {{")?;
        write!(f, "{spacer}cells: [")?;
        for (idx, &cell) in self.0.cells.iter().enumerate() {
            let pt = coordinates(idx as u32, self.0.width_with_border);
            if pt.x == 0 || pt.y == 0 || pt.y == self.0.height() + 1 {
                continue;
            }
            let idx = CellIdx(idx as u32);
            write!(f, "{spacer}  {idx:3}: {cell}")?;
        }
        write!(f, "{spacer}],")?;
        write!(f, "{spacer}pending: [")?;
        write_comma_separated(f, &self.0.pending, |f, c| {
            write!(f, "({}: {})", c.idx, c.value)
        })?;
        write!(f, "],")?;
        write!(f, "{spacer}desolved: [")?;
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
        write!(
            f,
            "{spacer}next partition id: {},",
            self.0.next_partition_id
        )?;
        write!(f, "{spacer}partitions: [")?;
        for (&pid, partition) in self.0.partitions.iter() {
            write!(f, "{spacer}  {:3}: {partition},", pid.val())?;
        }
        write!(f, "{spacer}],")?;
        if f.alternate() {
            writeln!(f)?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colored_minefield_display() {
        let field = MineField::create(
            "1.*@@1
             0123@@
             45678.",
        );
        let colored = format!("{}", ColoredMineField { field: &field });

        // Header contains 1-based column indices (1..=6) and Unicode top border
        assert!(colored.contains(" 1 "));
        assert!(colored.contains(" 6 "));
        assert!(colored.contains("┌───"));
        // Row prefixes equal row * width_with_border (`1 * 7 = 7`, `2 * 7 = 14`, `3 * 7 = 21`)
        assert!(colored.contains("   7 "));
        assert!(colored.contains("  14 "));
        assert!(colored.contains("  21 "));
        // Clues 1..=8 are present in rendered cells
        for digit in '1'..='8' {
            assert!(colored.contains(digit));
        }

        // Empty-interior height (0 rows would fail MineField::new, so test minimum 1x1)
        let small = MineField::new(1, 1).unwrap();
        let small_colored = format!("{}", ColoredMineField { field: &small });
        assert!(small_colored.contains("   1 "));
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

use crate::solver::CellValue;

use super::MineField;
use colored::Colorize;

pub fn print_field(field: &MineField) -> String {
    let mut text = String::new();

    if field.height <= 0 {
        return text;
    }

    print_header(field, &mut text);
    print_row(field, 0, &mut text);
    for row in 1..field.height {
        print_separator(field, &mut text);
        print_row(field, row, &mut text);
    }
    print_footer(field, &mut text);

    text
}

fn print_row(field: &MineField, row: usize, text: &mut String) {
    if field.width <= 0 {
        return;
    }

    for col in 0..field.width {
        text.push_str(&"│ ".bold().to_string());
        let printable = match field.cell(col, row).value {
            CellValue::Unknown => &" ".on_bright_black().to_string(),
            CellValue::Mine => &" ".on_bright_red().to_string(),
            CellValue::Count(count) => match count {
                1 => &"1".bright_blue().to_string(),
                2 => &"2".green().to_string(),
                3 => &"3".red().to_string(),
                4 => &"4".purple().to_string(),
                5 => &"5".yellow().to_string(),
                6 => &"6".cyan().to_string(),
                7 => &"6".bright_white().to_string(),
                8 => &"8".bright_black().to_string(),
                _ => " ",
            },
        };
        text.push_str(printable);
        text.push_str(" ");
    }
    text.push_str(&"│".bold().to_string());
    text.push_str("\n");
}

fn print_header(field: &MineField, text: &mut String) {
    if field.width <= 0 {
        return;
    }

    text.push_str(&"┌───".bold().to_string());
    for _ in 1..field.width {
        text.push_str(&"┬───".bold().to_string());
    }
    text.push_str(&"┐".bold().to_string());
    text.push_str("\n");
}

fn print_separator(field: &MineField, text: &mut String) {
    if field.width <= 0 {
        return;
    }

    text.push_str(&"├───".bold().to_string());
    for _ in 1..field.width {
        text.push_str(&"┼───".bold().to_string());
    }
    text.push_str(&"┤".bold().to_string());
    text.push_str("\n");
}

fn print_footer(field: &MineField, text: &mut String) {
    if field.width <= 0 {
        return;
    }

    text.push_str(&"└───".bold().to_string());
    for _ in 1..field.width {
        text.push_str(&"┴───".bold().to_string());
    }
    text.push_str(&"┘".bold().to_string());
    text.push_str("\n");
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use rand::rng;
    use rand::seq::SliceRandom;
    use std::cmp::min;

    fn create_test_field(width: usize, height: usize, mine_count: usize) -> MineField {
        let mut field = MineField::new(width, height, mine_count);
        let mut counts = vec![vec![0i8; height]; width];
        let mut coords: Vec<(usize, usize)> = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                coords.push((x, y));
            }
        }
        coords.shuffle(&mut rng());
        for i in 0..mine_count {
            let (x, y) = coords[i];
            counts[x][y] -= 10;
            for ny in y.saturating_sub(1)..min(y + 2, height) {
                for nx in x.saturating_sub(1)..min(x + 2, width) {
                    counts[nx][ny] += 1;
                }
            }
        }

        for y in 0..height {
            for x in 0..(width / 2) {
                field.set_cell_value(
                    x,
                    y,
                    if counts[x][y] < 0 {
                        CellValue::Mine
                    } else {
                        CellValue::Count(counts[x][y])
                    },
                );
            }
        }

        field
    }

    #[test]
    fn test_print() {
        let field = create_test_field(30, 16, 99);
        println!("{}", print_field(&field));
    }
}

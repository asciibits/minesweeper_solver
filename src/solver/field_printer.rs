use crate::solver::{cell::CellValue, direction::Point};

use super::MineField;
use colored::Colorize;

pub fn print_field(field: &MineField) -> String {
    let mut text = String::new();

    if field.height <= 0 {
        return text;
    }

    print_pre_header(field, &mut text);
    print_header(field, &mut text);
    print_row(field, 0, &mut text);
    for row in 1..field.height {
        print_separator(field, &mut text);
        print_row(field, row, &mut text);
    }
    print_footer(field, &mut text);

    text
}

fn print_row(field: &MineField, row: u32, text: &mut String) {
    if field.width <= 0 {
        return;
    }

    text.push_str(&format!("{:4}", row * field.width));
    for col in 0..field.width {
        text.push_str(&"│ ".bold().to_string());
        let printable = match field.cell(Point { x: col, y: row }).value {
            CellValue::Unknown => " ".on_bright_black(),
            CellValue::Mine => " ".on_bright_red(),
            CellValue::Count(count) => match count {
                1 => "1".bright_blue(),
                2 => "2".green(),
                3 => "3".red(),
                4 => "4".purple(),
                5 => "5".yellow(),
                6 => "6".cyan(),
                7 => "6".bright_white(),
                8 => "8".bright_black(),
                _ => " ".clear(),
            },
        };

        // let printable = match field.cell_partition_id(col, row) {
        //     Some(0) => printable.bold().on_yellow(),
        //     Some(1) => printable.bold().on_magenta(),
        //     Some(2) => printable.bold().on_white(),
        //     _ => printable,
        // };

        text.push_str(&printable.to_string());
        text.push_str(" ");
    }
    text.push_str(&"│".bold().to_string());
    text.push_str("\n");
}

fn print_pre_header(field: &MineField, text: &mut String) {
    if field.width <= 0 {
        return;
    }
    text.push_str("     ");
    for i in 0..field.width {
        text.push_str(&format!("{:^4}", i));
    }
    text.push_str("\n");
}

fn print_header(field: &MineField, text: &mut String) {
    if field.width <= 0 {
        return;
    }

    text.push_str("    ");
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

    text.push_str("    ");
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

    text.push_str("    ");
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
    use std::collections::HashSet;

    fn create_test_field(
        width: u32,
        height: u32,
        mine_count: u32,
        open_x: i32,
        open_y: i32,
    ) -> Result<MineField, SolverError> {
        let mut field = MineField::new(width, height, mine_count)?;
        let mut counts = vec![vec![0i8; height as usize]; width as usize];
        let mut coords: Vec<Point> = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                if (x as i32 - open_x).abs() > 1 || (y as i32 - open_y) > 1 {
                    coords.push(Point { x, y });
                }
            }
        }
        coords.shuffle(&mut rng());
        for i in 0..mine_count as usize {
            let Point { x, y } = coords[i];
            counts[x as usize][y as usize] -= 10;
            for ny in y.saturating_sub(1)..min(y + 2, height) {
                for nx in x.saturating_sub(1)..min(x + 2, width) {
                    counts[nx as usize][ny as usize] += 1;
                }
            }
        }

        // open area
        let mut open_cells: HashSet<Cell> =
            HashSet::with_capacity(width as usize * height as usize);
        fn open_cell(x: u32, y: u32, counts: &Vec<Vec<i8>>, cells: &mut HashSet<Cell>) {
            let value = match counts[x as usize][y as usize] {
                ..0 => CellValue::Mine,
                n => CellValue::Count(n),
            };
            if cells.insert(Cell {
                point: Point { x, y },
                value,
            }) && counts[x as usize][y as usize] == 0
            {
                for nx in x.saturating_sub(1)..min(x + 2, counts.len() as u32) {
                    for ny in y.saturating_sub(1)..min(y + 2, counts[nx as usize].len() as u32) {
                        open_cell(nx, ny, counts, cells);
                    }
                }
            }
        }
        open_cell(open_x as u32, open_y as u32, &counts, &mut open_cells);
        field.set_cells(&open_cells)?;

        Ok(field)
    }

    #[test]
    fn test_print() {
        let field = create_test_field(10, 10, 12, 5, 5).unwrap();
        println!("{}", print_field(&field));
        for cell in field.cells {
            eprintln!("{:#}", cell);
        }
        // println!("Partition: {:#?}", field.partitions().iter());
    }
}

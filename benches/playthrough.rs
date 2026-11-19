#![feature(test)]

extern crate test;

use minesweeper_solver::{Cell, CellValue, FieldBuilder, InferredValue, MineField, Point};
use rand::{RngExt, SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::hint::black_box;
use test::Bencher;

fn play_single_game(
    seed: u64,
    width: u32,
    height: u32,
    mine_count: u32,
    mut on_step: impl FnMut(&mut FieldBuilder),
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let start_pt = Point {
        x: rng.random_range(0..width),
        y: rng.random_range(0..height),
    };
    let mut builder =
        FieldBuilder::random_with_zero_region(width, height, mine_count, start_pt, |cells| {
            cells.shuffle(&mut rng)
        });

    builder.expand_points([start_pt]);
    on_step(&mut builder);

    loop {
        let inferred: Vec<Point> = {
            let field = builder.mine_field();
            field
                .recent_inferred()
                .filter(|&pt| field.cell(pt).inferred == InferredValue::Safe)
                .collect()
        };
        if !inferred.is_empty() {
            builder.expand_points(inferred);
        } else if let Some(pt) = builder.safe_frontier_hint(|max| rng.random_range(0..max)) {
            builder.expand_points([pt]);
        } else {
            break;
        }
        on_step(&mut builder);
    }
}

fn record_game_batches(seed: u64, width: u32, height: u32, mine_count: u32) -> Vec<Vec<Cell>> {
    let mut batches = Vec::new();
    let mut snapshot = vec![CellValue::Unknown; (width * height) as usize];

    play_single_game(seed, width, height, mine_count, |builder| {
        let mut batch = Vec::new();
        for c in builder.mine_field().cells() {
            let i = (c.cell.point.x + c.cell.point.y * width) as usize;
            if c.cell.value != snapshot[i] {
                snapshot[i] = c.cell.value;
                batch.push(c.cell);
            }
        }
        if !batch.is_empty() {
            batches.push(batch);
        }
    });

    batches
}

#[bench]
fn bench_solver_incremental_replay_1000_games(b: &mut Bencher) {
    let games: Vec<Vec<Vec<Cell>>> = (0..1000)
        .map(|seed| record_game_batches(seed, 30, 24, 130))
        .collect();

    b.iter(|| {
        for game in &games {
            let mut field = MineField::with_mine_count(30, 24, 130).unwrap();
            for batch in game {
                field.set_cells(batch.iter().copied()).unwrap();
            }
            black_box(field);
        }
    });
}

#[bench]
fn bench_end_to_end_incremental_playthrough_1000_games(b: &mut Bencher) {
    b.iter(|| {
        for seed in 0..1000 {
            play_single_game(seed, 30, 24, 130, |builder| {
                black_box(&*builder);
            });
        }
    });
}

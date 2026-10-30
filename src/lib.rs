//! A Minesweeper board representation and constraint-based solver.
//!
//! This crate tracks known cell counts, mines, and unknown frontier cells on a
//! [`MineField`], partitioning the active boundary into independent connected
//! components ([`Partition`]) to deduce safe cells and mine locations.

pub mod solver;

pub use solver::{
    MineField, SolverError,
    cell::{UserCell as Cell, UserCellValue as CellValue},
    grid::Point,
    partition::Partition,
};

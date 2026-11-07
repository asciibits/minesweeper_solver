//! A Minesweeper board representation and constraint-based solver.
//!
//! This crate tracks known cell counts, mines, and unknown frontier cells on a
//! [`MineField`], partitioning the active boundary into independent connected
//! components to deduce safe cells and mine locations.

mod solver;

pub use solver::{
    MineField, SolverError,
    cell::{
        InferredValue, UserCell as Cell, UserCellValue as CellValue,
        UserValueWithInference as CellWithInference,
    },
    grid::Point,
    test_utils::FieldBuilder,
};

//! Connected components ([`Partition`]) of frontier cells and cell priority ordering.

use crate::solver::{CellIdx, FieldArgs, cell::Cell, grid::Neighbor};

fn walk_known(
    idx: CellIdx,
    field_args: &mut FieldArgs,
    visit: &mut impl FnMut(CellIdx, &mut FieldArgs),
) -> usize {
    // process a "known" cell - i.e. a safe cell with a mine count
    let cell = &mut field_args.cells[idx];
    if cell.partition_id() == *field_args.next_partition_id || cell.value().is_inferred() {
        return 0;
    }
    cell.set_partition_id(*field_args.next_partition_id);

    // scan neighbors for inferred cells - both inferred mines and inferred safe
    let neighbors = cell.neighbors() - cell.unknown_neighbors();
    let mut len = 1;
    visit(idx, field_args);
    if let Some(metrics) = field_args.metrics.as_mut() {
        metrics.inc_neighbor_visits(neighbors);
    }
    for Neighbor { neighbor, .. } in Cell::neighbor_iter(idx, field_args.width, neighbors) {
        len += walk_unknown(neighbor, field_args, visit);
    }
    len
}

fn walk_unknown(
    idx: CellIdx,
    field_args: &mut FieldArgs,
    visit: &mut impl FnMut(CellIdx, &mut FieldArgs),
) -> usize {
    // process an "inferred" cell
    let cell = &mut field_args.cells[idx];
    if cell.partition_id() == *field_args.next_partition_id || !cell.value().is_inferred() {
        return 0;
    }
    cell.set_partition_id(*field_args.next_partition_id);

    let neighbors = cell.safe_neighbors();
    if let Some(metrics) = field_args.metrics.as_mut() {
        metrics.inc_neighbor_visits(neighbors);
    }

    // scan neighbors for known cells
    let mut len = 1;
    visit(idx, field_args);
    for Neighbor { neighbor, .. } in Cell::neighbor_iter(idx, field_args.width, neighbors) {
        len += walk_known(neighbor, field_args, visit);
    }
    len
}

// Step through the cells of a partition. Note: This function scans assuming the starting cell
// has been mutated, and is unknown whether it was originally a known vs unknown. So, the
// starting cell looks for *all* unknown + mine neighbors.
pub(super) fn walk_from_invalid_start(
    idx: CellIdx,
    field_args: &mut FieldArgs,
    visit: &mut impl FnMut(CellIdx, &mut FieldArgs),
) -> usize {
    let mut len = walk_known(idx, field_args, visit);
    len += walk_unknown(idx, field_args, visit);

    *field_args.next_partition_id += 1;
    if let Some(metrics) = field_args.metrics.as_mut() {
        metrics.inc_partitions_created();
        metrics.set_max_partition_size(len as usize);
    }
    len
}

pub(super) fn walk_from_known(
    member: CellIdx,
    field_args: &mut FieldArgs,
    visit: &mut impl FnMut(CellIdx, &mut FieldArgs),
) -> usize {
    let len = walk_known(member, field_args, visit);
    *field_args.next_partition_id += 1;

    if let Some(metrics) = field_args.metrics.as_mut() {
        metrics.inc_partitions_created();
        metrics.set_max_partition_size(len as usize);
    }
    len
}

/// Returns the slice of linear cell indices belonging to this partition.
#[cfg(test)]
pub(super) fn members(
    member: CellIdx,
    capacity: usize,
    field_args: &mut FieldArgs,
) -> Vec<CellIdx> {
    let mut members = Vec::with_capacity(capacity);
    let members_ref = &mut members;

    walk_from_known(member, field_args, &mut |i, _| {
        members_ref.push(i);
    });
    members
}

#[cfg(test)]
mod tests {

    #[test]
    #[ignore]
    fn test_partition_create_skips_mines_and_isolated_cells() {
        // let mut cells = Cell::grid(2, 1);
        // // Cell 0 is Unknown with no safe_neighbors (isolated)
        // assert!(create_partition(&mut cells, IndexedOffsets::new(2), CellIdx(0),).is_none());

        // // Cell 1 is a Mine even if safe_neighbors is non-empty
        // cells[1].set_value(CellValue::MINE);
        // cells[1].set_safe(W);
        // assert!(create_partition(&mut cells, IndexedOffsets::new(2), CellIdx(1),).is_none());
    }

    #[test]
    #[ignore]
    fn test_partition_create_connected_bipartite_component() {
        // // 3x1 board: [Count(1), Unknown, Count(2)]
        // let mut cells = Cell::grid(3, 1);
        // cells[0].set_value(CellValue::for_mine_count(1));

        // cells[1].set_value(CellValue::UNKNOWN);
        // cells[1].set_safe(W);
        // cells[1].set_safe(E);

        // cells[2].set_value(CellValue::for_mine_count(2));

        // let (id, partition) = create_partition(&mut cells, IndexedOffsets::new(3), CellIdx(0))
        //     .expect("should create partition");
        // assert!(!partition.is_empty());
        // assert_eq!(partition.len(), 3);
        // assert_eq!(partition.members(), &[CellIdx(0), CellIdx(1), CellIdx(2)]);
        // for c in &cells {
        //     assert_eq!(c.partition_id(), id);
        // }
    }
}

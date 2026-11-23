#![allow(dead_code)]

use std::{
    array,
    collections::{HashMap, HashSet},
    fmt::{self, Display, Formatter},
    ops::{BitXor, BitXorAssign},
};

use crate::solver::{Cell, CellIdx, FastNumberHasherBuilder};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct BitMask<const N: usize>([u64; N]);

impl<const N: usize> BitMask<N> {
    const SIZE: usize = N * 64;
    const ZERO: Self = Self([0u64; N]);
    const ONE: Self = const {
        let mut val = Self::ZERO;
        val.0[0] = 1;
        val
    };
    const ONES: Self = Self([u64::MAX; N]);
}

impl<const N: usize> Default for BitMask<N> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const N: usize> BitXor for BitMask<N> {
    type Output = BitMask<N>;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(array::from_fn(|i| self.0[i] ^ rhs.0[i]))
    }
}

impl<const N: usize> BitXorAssign for BitMask<N> {
    fn bitxor_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.0[i] ^= rhs.0[i];
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct UnsolvedCell<const N: usize> {
    unknown_mask: BitMask<N>,
    opened_siblings: BitMask<N>,
    missing_mine_count: u8,
}

trait PartitionSolver {
    fn solve(
        cells: &[Cell],
        width: u32,
        opened_members: &[CellIdx],
        unknown_members: &[CellIdx],
    ) -> PartitionStatsByMineCount;
}

#[derive(Debug, Clone)]
struct MaskedPartitionSolver<const N: usize> {
    mine_mask: BitMask<N>,
    opened_map: HashMap<CellIdx, usize, FastNumberHasherBuilder>,
    unknown_map: HashMap<CellIdx, usize, FastNumberHasherBuilder>,
}

impl<const N: usize> MaskedPartitionSolver<N> {
    fn solve_partition(&mut self) -> PartitionStatsByMineCount {
        unimplemented!();
    }
}

impl<const N: usize> PartitionSolver for MaskedPartitionSolver<N> {
    fn solve(
        _cells: &[Cell],
        _width: u32,
        opened_members: &[CellIdx],
        unknown_members: &[CellIdx],
    ) -> PartitionStatsByMineCount {
        let mut opened_map =
            HashMap::with_capacity_and_hasher(opened_members.len(), FastNumberHasherBuilder);
        opened_map.extend(opened_members.iter().enumerate().map(|(k, &v)| (v, k)));

        let mut unknown_map =
            HashMap::with_capacity_and_hasher(opened_members.len(), FastNumberHasherBuilder);
        // sort unknown map to keep the unknown cells that share the same set of opened neighbors together
        // unknown_members.iter().map(|&idx| {
        //     let cell = cells[idx];
        //     let opened_neighbors = cell.neighbors.opened();
        //     let idx = idx.val();
        //     let neighbor_mask = opened_neighbors.mask();
        //     if neighbor_mask & DirectionSet::W_DIRECTIONS.mask() == 0 {
        //         idx += 1;
        //         neighbor_mask >>= 1;
        //         if neighbor_mask & DirectionSet::W_DIRECTIONS.mask() == 0 {
        //             idx += 1;
        //             neighbor_mask >>= 1;
        //         }
        //     }
        //     if neighbor_mask & DirectionSet::N_DIRECTIONS.mask() == 0 {
        //         idx += width;
        //         neighbor_mask >>= 1;
        //         if neighbor_mask & DirectionSet::N_DIRECTIONS.mask() == 0 {
        //             idx += 1;
        //             neighbor_mask >>= 1;
        //         }
        //     }

        //     let unknown_group_idx = idx.val() << 9 |
        // });

        unknown_map.extend(unknown_members.iter().enumerate().map(|(k, &v)| (v, k)));
        let mine_mask = BitMask::<N>::default();

        let mut solver = Self {
            mine_mask,
            opened_map,
            unknown_map,
        };

        solver.solve_partition()
    }
}

#[derive(Debug, Clone)]
pub(super) struct PartitionStats {
    pub(super) n: u64,
    pub(super) cell_mine_tally: HashSet<CellIdx, u64>,
}

#[derive(Debug, Clone)]
pub(super) struct PartitionStatsByMineCount {
    pub(super) partition_stats_per_mine_count: HashSet<u16, PartitionStats>,
}

#[derive(Debug, Default, Clone)]
pub(super) struct Partition {
    opened_members: Box<[CellIdx]>,
    unknown_members: Box<[CellIdx]>,
    stats: Option<PartitionStatsByMineCount>,
}

impl Partition {
    pub(super) fn new(
        opened_members: impl Into<Box<[CellIdx]>>,
        unknown_members: impl Into<Box<[CellIdx]>>,
    ) -> Partition {
        Partition {
            opened_members: opened_members.into(),
            unknown_members: unknown_members.into(),
            stats: None,
        }
    }

    #[cfg(test)]
    pub(super) fn members(&self) -> impl Iterator<Item = CellIdx> {
        self.opened_members
            .iter()
            .copied()
            .chain(self.unknown_members.iter().copied())
    }

    pub(super) fn opened_members(&self) -> impl Iterator<Item = CellIdx> {
        self.opened_members.iter().copied()
    }

    pub(super) fn unknown_members(&self) -> impl Iterator<Item = CellIdx> {
        self.unknown_members.iter().copied()
    }

    pub(super) fn len(&self) -> u16 {
        let len = self.opened_members.len() + self.unknown_members.len();
        debug_assert!(len < u16::MAX as usize);
        len as u16
    }

    pub(super) fn solve(&mut self, cells: &[Cell], width: u32) -> &PartitionStatsByMineCount {
        self.stats.get_or_insert_with(|| {
            if self.opened_members.len() <= 64 && self.unknown_members.len() <= 64 {
                MaskedPartitionSolver::<1>::solve(
                    cells,
                    width,
                    &self.opened_members,
                    &self.unknown_members,
                )
            } else if self.opened_members.len() <= 128 && self.unknown_members.len() <= 128 {
                MaskedPartitionSolver::<2>::solve(
                    cells,
                    width,
                    &self.opened_members,
                    &self.unknown_members,
                )
            } else {
                unimplemented!()
            }
        })
    }
}

impl Display for Partition {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Partition {{opened_members: [{}], unknown_members: [{}]}}",
            self.opened_members
                .iter()
                .map(|&i| i.val().to_string())
                .collect::<Vec<_>>()
                .join(","),
            self.unknown_members
                .iter()
                .map(|&i| i.val().to_string())
                .collect::<Vec<_>>()
                .join(","),
        )
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct PartitionId(u16);

impl PartitionId {
    pub(super) const ZERO: PartitionId = PartitionId(0);
    pub(super) const ONE: PartitionId = PartitionId(1);
    #[cfg(test)]
    pub(super) const MAX: PartitionId = PartitionId(u16::MAX);
    pub(super) const COMPACT_THRESHOLD: PartitionId = PartitionId(u16::MAX >> 1);

    #[inline]
    #[must_use]
    pub(super) fn inc(&self) -> PartitionId {
        PartitionId(
            self.0
                .checked_add(1)
                .expect("PartitionId overflow: too many partitions"),
        )
    }

    #[inline]
    #[must_use]
    #[cfg(any(test, feature = "debug"))]
    pub(super) fn val(&self) -> u16 {
        self.0
    }
}

impl Display for PartitionId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::{MineField, Point, solver::partition::PartitionId};

    #[test]
    fn test_partition_id_ops() {
        let mut pid = PartitionId::ZERO;
        assert_eq!(pid.to_string(), "0");
        pid = pid.inc();
        assert_eq!(pid, PartitionId(1));
        pid.0 = PartitionId::COMPACT_THRESHOLD.0;
        assert_eq!(pid, PartitionId::COMPACT_THRESHOLD);
        assert_eq!(PartitionId::COMPACT_THRESHOLD.0, u16::MAX >> 1);

        let overflow = std::panic::catch_unwind(|| {
            let max_pid = PartitionId::MAX;
            _ = max_pid.inc();
        });
        assert!(overflow.is_err());
    }

    #[test]
    fn test_compact_partition_ids() {
        let spec = "1...1
                    .....
                    1...1";
        let mut field = MineField::create(spec);
        assert!(!field.partitions.is_empty());
        let orig_partition_count = field.partitions.len();

        // Artificially elevate next_partition_id to u16::MAX
        field.next_partition_id = PartitionId::MAX;
        // Inject an obsolete cell partition_id that is NOT in field.partitions
        field.cells[9].partition_id.0 = PartitionId::COMPACT_THRESHOLD.0 + 155;

        field.compact_partition();

        assert_eq!(field.partitions.len(), orig_partition_count);
        assert_eq!(field.next_partition_id.0, orig_partition_count as u16 + 1);
        // Dangling partition ID on cell 9 was reset
        assert_eq!(field.cells[9].partition_id.0, 0);

        // All active partition members have IDs <= orig_partition_count
        for &pid in field.partitions.keys() {
            assert!(pid.0 <= orig_partition_count as u16);
        }

        // Solving further when next_partition_id > COMPACT_THRESHOLD automatically triggers compaction
        field.next_partition_id = PartitionId(PartitionId::COMPACT_THRESHOLD.0 + 911);
        field
            .set_cell(crate::Cell {
                point: Point { x: 0, y: 1 },
                value: crate::CellValue::Opened(2),
            })
            .unwrap();

        assert_eq!(field.next_partition_id.0, orig_partition_count as u16 + 2);
    }
}

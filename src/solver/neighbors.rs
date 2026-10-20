use std::{
    cmp::min,
    fmt::{self, Display, Formatter},
};

use crate::solver::cell::{cell_idx, coordinates};

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub struct Neighbors(usize);

impl Neighbors {
    pub const EMPTY: Neighbors = Neighbors(0);

    pub fn single(cell_idx: usize) -> Neighbors {
        Neighbors((cell_idx << 9) | 1)
    }
    pub fn all_neighbors(x: usize, y: usize, width: usize, height: usize) -> Neighbors {
        let cell_idx = cell_idx(x, y, width);

        // Create a number to identify where the cell is. The values look like:'
        //
        // +3 +0    +0 +6
        // --------------
        //  4  1 ..  1  7  -- +1
        //  3  0 ..  0  6  -- +0
        //  ..             -- +0
        //  3  0 ..  0  6  -- +0
        //  5  2 ..  2  8  -- +2
        let cell_state = if x == 0 {
            3
        } else if x >= width - 1 {
            6
        } else {
            0
        } + if y == 0 {
            1
        } else if y >= height - 1 {
            2
        } else {
            0
        };

        Neighbors(match cell_state {
            0 => ((cell_idx - width - 1) << 9) | 0b111_101_111, // middle cells
            1 => ((cell_idx - 1) << 9) | 0b000_111_101,         // top row
            2 => ((cell_idx - width - 1) << 9) | 0b000_101_111, // bottom row
            3 => ((cell_idx - width) << 9) | 0b011_010_011,     // left col
            4 => (cell_idx << 9) | 0b000_011_010,               // NW corner
            5 => ((cell_idx - width) << 9) | 0b000_010_011,     // SW corner
            6 => ((cell_idx - width - 1) << 9) | 0b011_001_011, // right col
            7 => ((cell_idx - 1) << 9) | 0b000_011_001,         // NE corner
            8 => ((cell_idx - width - 1) << 9) | 0b000_001_011, // SE corner
            _ => unreachable!(),
        })
    }
    pub fn root_cell_idx(&self) -> usize {
        self.0 >> 9
    }

    pub fn neighbor_mask(&self) -> usize {
        self.0 & 0b111_111_111
    }

    pub fn len(&self) -> u32 {
        self.neighbor_mask().count_ones()
    }

    pub fn contains(&self, width: usize, mut cell_idx: usize) -> bool {
        let root = self.root_cell_idx();
        let mut mask: usize = 1;
        if cell_idx >= root + width {
            cell_idx -= width;
            mask <<= 3;
            if cell_idx >= root + width {
                cell_idx -= width;
                mask <<= 3;
            }
        }
        if cell_idx > root {
            cell_idx -= 1;
            mask <<= 1;
            if cell_idx > root {
                cell_idx -= 1;
                mask <<= 1;
            }
        }
        cell_idx == root && self.0 & mask != 0
    }

    pub fn add(&mut self, width: usize, cell_idx: usize) {
        self.union(width, Neighbors::single(cell_idx));
    }

    pub fn remove(&mut self, width: usize, cell_idx: usize) {
        self.subtract(width, Neighbors::single(cell_idx));
    }

    /// Add a single cell to to the set of Neighbors. Note: This method
    /// requires that the new cell fits within a 3x3 grid with all existing
    /// cells. If it does not, then the results are undefined (current
    /// implementation will panic in debug, but in release it will leave
    /// the state in a completely broken state)
    pub fn union(&mut self, width: usize, other: Neighbors) {
        self.binary_op(width, other, |a, b| a | b);
    }

    pub fn subtract(&mut self, width: usize, other: Neighbors) {
        self.binary_op(width, other, |a, b| a & !b);
    }

    pub fn intersect(&mut self, width: usize, other: Neighbors) {
        self.binary_op(width, other, |a, b| a & b);
    }

    fn binary_op<B>(&mut self, width: usize, other: Neighbors, binary_op: B)
    where
        B: Fn(usize, usize) -> usize,
    {
        if self.0 == 0 {
            self.0 = other.0;
            return;
        } else if other.0 == 0 {
            return;
        }
        let (x0, y0) = coordinates(self.root_cell_idx(), width);
        let (x1, y1) = coordinates(other.root_cell_idx(), width);
        let min_x = min(x0, x1);
        let min_y = min(y0, y1);

        // In debug mode, ensure the union is valid
        debug_assert!(
            x0 - min_x
                <= if self.0 & 0b110_110_110 == 0 {
                    2
                } else if self.0 & 0b100_100_100 == 0 {
                    1
                } else {
                    0
                },
        );
        debug_assert!(
            x1 - min_x
                <= if other.0 & 0b110_110_110 == 0 {
                    2
                } else if other.0 & 0b100_100_100 == 0 {
                    1
                } else {
                    0
                }
        );
        debug_assert!(
            y0 - min_y
                <= if self.0 & 0b111_111_000 == 0 {
                    2
                } else if self.0 & 0b111_000_000 == 0 {
                    1
                } else {
                    0
                }
        );
        debug_assert!(
            y1 - min_y
                <= if other.0 & 0b111_111_000 == 0 {
                    2
                } else if other.0 & 0b111_000_000 == 0 {
                    1
                } else {
                    0
                }
        );

        self.0 = (cell_idx(min_x, min_y, width) << 9)
            | binary_op(
                self.neighbor_mask() << (x0 - min_x + 3 * (y0 - min_y)),
                other.neighbor_mask() << (x1 - min_x + 3 * (y1 - min_y)),
            );
        self.normalize(width);
    }

    /// Move the cell_idx to the most lower-right position while still preserving
    /// the neighbor iterator
    fn normalize(&mut self, width: usize) {
        let mut mask = self.neighbor_mask();
        if mask == 0 {
            self.0 = 0;
            return;
        };
        let mut root = self.root_cell_idx();
        if mask & 0b001_001_001 == 0 {
            mask >>= 1;
            root += 1;
            if mask & 0b001_001_001 == 0 {
                mask >>= 1;
                root += 1;
            }
        }
        if mask & 0b000_000_111 == 0 {
            mask >>= 3;
            root += width;
            if mask & 0b111_000_000 == 0 {
                mask >>= 3;
                root += width;
            }
        }
        self.0 = (root << 9) | mask;
    }

    fn neighbor_index_to_cell_idx(&self, width: usize, neighbor_index: u32) -> usize {
        let root = self.root_cell_idx();
        match neighbor_index {
            0 => root,
            1 => root + 1,
            2 => root + 2,
            3 => root + width,
            4 => root + width + 1,
            5 => root + width + 2,
            6 => root + width * 2,
            7 => root + width * 2 + 1,
            8 => root + width * 2 + 2,
            _ => unreachable!(),
        }
    }

    pub fn filter<P>(&self, width: usize, mut predicate: P) -> Neighbors
    where
        P: FnMut(usize) -> bool,
    {
        let mask = BinaryIterator(self.neighbor_mask())
            .map(|m| {
                (
                    m,
                    self.neighbor_index_to_cell_idx(width, m.trailing_zeros()),
                )
            })
            .filter(|&(_, cell_idx)| predicate(cell_idx))
            .fold(0, |acc, (b, _)| acc | b);
        if mask == 0 {
            return Neighbors(0);
        }
        let mut result = Neighbors((self.0 & !0b111_111_111) | mask);
        result.normalize(width);
        result
    }

    pub fn iter(&self, width: usize) -> impl Iterator<Item = usize> {
        BinaryIterator(self.neighbor_mask())
            .map(move |m| self.neighbor_index_to_cell_idx(width, m.trailing_zeros()))
    }

    pub fn into_iter(self, width: usize) -> impl Iterator<Item = usize> {
        BinaryIterator(self.neighbor_mask())
            .map(move |m| self.neighbor_index_to_cell_idx(width, m.trailing_zeros()))
    }
}

impl Display for Neighbors {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // Customize your output format here
        write!(
            f,
            "root_cell: {}, neighbors: {:b}",
            self.root_cell_idx(),
            self.neighbor_mask()
        )
    }
}

/// An iterator that returns all the `1` digits in a binary number.
/// Ex. the values 11 in binary is 0b1011; this iterator will generate the
/// values: [0b0001, 0b0010, 0b1000]
struct BinaryIterator(usize);

impl Iterator for BinaryIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            return None;
        }
        let t = self.0;
        self.0 &= t - 1;
        Some(self.0 ^ t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_iterator_iterates() {
        assert_eq!(
            BinaryIterator(0b1011001).collect::<Vec<usize>>(),
            vec![0b1, 0b1000, 0b10000, 0b1000000]
        );
        assert_eq!(BinaryIterator(0).collect::<Vec<usize>>(), vec![]);
    }

    #[test]
    fn neighbors_full_in_middle() {
        let neighbors = Neighbors::all_neighbors(5, 5, 10, 10);
        assert_eq!(
            neighbors.iter(10).collect::<Vec<usize>>(),
            vec![44, 45, 46, 54, 56, 64, 65, 66]
        );
        let neighbors = Neighbors::all_neighbors(1, 1, 10, 10);
        assert_eq!(
            neighbors.iter(10).collect::<Vec<usize>>(),
            vec![0, 1, 2, 10, 12, 20, 21, 22]
        );
        let neighbors = Neighbors::all_neighbors(8, 8, 10, 10);
        assert_eq!(
            neighbors.iter(10).collect::<Vec<usize>>(),
            vec![77, 78, 79, 87, 89, 97, 98, 99]
        );
        let neighbors = Neighbors::all_neighbors(8, 1, 10, 10);
        assert_eq!(
            neighbors.iter(10).collect::<Vec<usize>>(),
            vec![7, 8, 9, 17, 19, 27, 28, 29]
        );
        let neighbors = Neighbors::all_neighbors(1, 8, 10, 10);
        assert_eq!(
            neighbors.iter(10).collect::<Vec<usize>>(),
            vec![70, 71, 72, 80, 82, 90, 91, 92]
        );
    }

    #[test]
    fn neighbors_full_on_edges() {
        let neighbors = Neighbors::all_neighbors(0, 5, 10, 10);
        assert_eq!(
            neighbors.iter(10).collect::<Vec<usize>>(),
            vec![40, 41, 51, 60, 61]
        );
        let neighbors = Neighbors::all_neighbors(9, 5, 10, 10);
        assert_eq!(
            neighbors.iter(10).collect::<Vec<usize>>(),
            vec![48, 49, 58, 68, 69]
        );
        let neighbors = Neighbors::all_neighbors(5, 0, 10, 10);
        assert_eq!(
            neighbors.iter(10).collect::<Vec<usize>>(),
            vec![4, 6, 14, 15, 16]
        );
        let neighbors = Neighbors::all_neighbors(5, 9, 10, 10);
        assert_eq!(
            neighbors.iter(10).collect::<Vec<usize>>(),
            vec![84, 85, 86, 94, 96]
        );
    }

    #[test]
    fn neighbors_full_on_corners() {
        let neighbors = Neighbors::all_neighbors(0, 0, 10, 10);
        assert_eq!(neighbors.iter(10).collect::<Vec<usize>>(), vec![1, 10, 11]);
        let neighbors = Neighbors::all_neighbors(9, 0, 10, 10);
        assert_eq!(neighbors.iter(10).collect::<Vec<usize>>(), vec![8, 18, 19]);
        let neighbors = Neighbors::all_neighbors(0, 9, 10, 10);
        assert_eq!(neighbors.iter(10).collect::<Vec<usize>>(), vec![80, 81, 91]);
        let neighbors = Neighbors::all_neighbors(9, 9, 10, 10);
        assert_eq!(neighbors.iter(10).collect::<Vec<usize>>(), vec![88, 89, 98]);
    }

    #[test]
    fn neighbors_single() {
        let neighbors = Neighbors::single(55);
        assert_eq!(neighbors.iter(10).collect::<Vec<usize>>(), vec![55]);

        let neighbors = Neighbors::single(0);
        assert_eq!(neighbors.iter(10).collect::<Vec<usize>>(), vec![0]);

        let neighbors = Neighbors::single(90);
        assert_eq!(neighbors.iter(10).collect::<Vec<usize>>(), vec![90]);

        let neighbors = Neighbors::single(9);
        assert_eq!(neighbors.iter(10).collect::<Vec<usize>>(), vec![9]);

        let neighbors = Neighbors::single(99);
        assert_eq!(neighbors.iter(10).collect::<Vec<usize>>(), vec![99]);
    }

    #[test]
    fn neighbor_contains_members() {
        // test middle
        let neighbors = Neighbors::all_neighbors(5, 5, 10, 10);
        vec![44, 45, 46, 54, 56, 64, 65, 66]
            .into_iter()
            .for_each(|idx| assert!(neighbors.contains(10, idx)));
        vec![33, 35, 37, 53, 55, 57, 73, 75, 77]
            .into_iter()
            .for_each(|idx| assert!(!neighbors.contains(10, idx)));

        // test NW corner
        let neighbors = Neighbors::all_neighbors(0, 0, 10, 10);
        vec![1, 10, 11]
            .into_iter()
            .for_each(|idx| assert!(neighbors.contains(10, idx)));
        vec![0, 2, 12, 20, 21, 22]
            .into_iter()
            .for_each(|idx| assert!(!neighbors.contains(10, idx)));

        // test SE corner
        let neighbors = Neighbors::all_neighbors(9, 9, 10, 10);
        vec![88, 89, 98]
            .into_iter()
            .for_each(|idx| assert!(neighbors.contains(10, idx)));
        vec![77, 78, 79, 87, 97, 99]
            .into_iter()
            .for_each(|idx| assert!(!neighbors.contains(10, idx)));
    }

    #[test]
    fn filter_values() {
        let neighbors = Neighbors::all_neighbors(5, 5, 10, 10);
        assert_eq!(
            neighbors
                .filter(10, |id| id != 44 && id != 66)
                .iter(10)
                .collect::<Vec<usize>>(),
            vec![45, 46, 54, 56, 64, 65]
        );
    }

    #[test]
    fn union() {
        let mut n = Neighbors::single(cell_idx(5, 5, 10));
        let n2 = Neighbors::single(cell_idx(4, 4, 10));
        n.union(10, n2);
        assert_eq!(n.0, (44 << 9) | 0b000_010_001);
        assert_eq!(n.iter(10).collect::<Vec<usize>>(), vec![44, 55]);

        let mut n = Neighbors::single(cell_idx(4, 5, 10));
        let n2 = Neighbors::single(cell_idx(5, 4, 10));
        n.union(10, n2);
        assert_eq!(n.0, (44 << 9) | 0b000_001_010);
        assert_eq!(n.iter(10).collect::<Vec<usize>>(), vec![45, 54]);

        let mut n = Neighbors::single(cell_idx(4, 6, 10));
        let n2 = Neighbors::single(cell_idx(6, 4, 10));
        n.union(10, n2);
        assert_eq!(n.0, (44 << 9) | 0b001_000_100);
        assert_eq!(n.iter(10).collect::<Vec<usize>>(), vec![46, 64]);
    }

    #[test]
    #[should_panic]
    fn invaalid_union() {
        let mut n1 = Neighbors::single(cell_idx(3, 3, 10));
        let n2 = Neighbors::single(cell_idx(6, 6, 10));
        n1.union(10, n2);
    }

    #[test]
    fn add() {
        let mut n = Neighbors::single(cell_idx(5, 5, 10));
        n.add(10, cell_idx(4, 4, 10));
        assert_eq!(n.0, (44 << 9) | 0b000_010_001);
        assert_eq!(n.iter(10).collect::<Vec<usize>>(), vec![44, 55]);

        let mut n = Neighbors::single(cell_idx(4, 5, 10));
        n.add(10, cell_idx(5, 4, 10));
        assert_eq!(n.0, (44 << 9) | 0b000_001_010);
        assert_eq!(n.iter(10).collect::<Vec<usize>>(), vec![45, 54]);

        let mut n = Neighbors::single(cell_idx(4, 6, 10));
        n.add(10, cell_idx(6, 4, 10));
        assert_eq!(n.0, (44 << 9) | 0b001_000_100);
        assert_eq!(n.iter(10).collect::<Vec<usize>>(), vec![46, 64]);
    }

    #[test]
    fn subtract() {
        let mut n = Neighbors::all_neighbors(5, 5, 10, 10);
        let n2 = Neighbors::single(cell_idx(4, 4, 10));
        n.subtract(10, n2);
        assert_eq!(n.root_cell_idx(), 44);
        assert_eq!(n.neighbor_mask(), 0b111_101_110);

        let mut n = Neighbors::all_neighbors(5, 5, 10, 10);
        let n2 = Neighbors((55 << 9) | 0b000_011_011);
        n.subtract(10, n2);
        assert_eq!(n.root_cell_idx(), 44);
        assert_eq!(n.neighbor_mask(), 0b001_001_111);

        let mut n = Neighbors::all_neighbors(5, 5, 10, 10);
        let n2 = Neighbors((44 << 9) | 0b001_001_001);
        n.subtract(10, n2);
        assert_eq!(n.root_cell_idx(), 45);
        assert_eq!(n.neighbor_mask(), 0b011_010_011);
    }

    #[test]
    #[should_panic]
    fn invalid_subtract() {
        let mut n1 = Neighbors::single(cell_idx(3, 3, 10));
        let n2 = Neighbors::single(cell_idx(6, 6, 10));
        n1.subtract(10, n2);
    }

    #[test]
    fn remove() {
        let mut n = Neighbors::all_neighbors(5, 5, 10, 10);
        n.remove(10, cell_idx(4, 4, 10));
        assert_eq!(n.root_cell_idx(), 44);
        assert_eq!(n.neighbor_mask(), 0b111_101_110);

        let mut n = Neighbors::single(cell_idx(5, 5, 10));
        n.remove(10, cell_idx(5, 5, 10));
        assert_eq!(n.root_cell_idx(), 0);
        assert_eq!(n.neighbor_mask(), 0);
    }
}

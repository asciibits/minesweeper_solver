//! 2D grid coordinates, compass directions, and bitset-backed neighbor iteration.

use std::{
    fmt::{self, Display, Formatter},
    ops::{BitAnd, BitOr, BitOrAssign, Not, Sub},
};

/// A 2D coordinate (`x`, `y`) on the Minesweeper grid, with `(0, 0)` at the top-left corner.
#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq)]
pub struct Point {
    /// Zero-based column index from the left edge.
    pub x: u32,
    /// Zero-based row index from the top edge.
    pub y: u32,
}

impl Point {
    /// Provide the x/y coordinates of a Cell with the given `cell_idx` in a fixed
    /// width field. This is the corollary to [`Point::to_index`].
    pub const fn from_index(cell_idx: usize, width: u32) -> Point {
        let width = width as usize;
        let y = cell_idx / width;
        let x = (cell_idx - y * width) as u32;
        let y = y as u32;
        Point { x, y }
    }

    /// Get a unique ID for the point in a given-width field. Useful for indexing in
    /// a Vector or Array. This is the corollary to [`Point::from_index`].
    pub const fn to_index(self, width: u32) -> usize {
        self.x as usize + self.y as usize * width as usize
    }

    /// Returns an iterator over the neighboring [`Point`]s specified by `directions`.
    pub fn neighbors(&self, directions: DirectionSet) -> NeighborIterator {
        NeighborIterator {
            p: *self,
            iter: directions.iter(),
        }
    }

    /// Returns the adjacent [`Point`] in direction `to`, paired with the reverse [`Direction`]
    /// pointing from that neighbor back to `self`.
    pub fn neighbor(&self, to: Direction) -> Neighbor<Point> {
        Neighbor {
            neighbor: Point {
                x: self.x.wrapping_add_signed(to.dx as i32),
                y: self.y.wrapping_add_signed(to.dy as i32),
            },
            from: to.reverse(),
        }
    }
}

impl Display for Point {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

/// An iterator yielding the [`Neighbor<Point>`]s of a [`Point`] for each direction in a [`DirectionSet`].
#[derive(Debug, Clone, Copy)]
pub struct NeighborIterator {
    p: Point,
    iter: DirectionSetIterator,
}

/// Associates a neighboring item `T` (such as a [`Point`] or cell reference) with the
/// [`Direction`] pointing from that neighbor back to the origin cell.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Neighbor<T> {
    /// The neighboring value or coordinate.
    pub neighbor: T,
    /// The direction from `neighbor` back to the origin cell.
    pub from: Direction,
}

impl<T> Neighbor<T> {
    /// Transforms the inner `neighbor` value using `f` while preserving the `from` direction.
    pub fn map<F, U>(self, f: F) -> Neighbor<U>
    where
        F: FnOnce(T) -> U,
    {
        Neighbor {
            neighbor: f(self.neighbor),
            from: self.from,
        }
    }
}

impl Iterator for NeighborIterator {
    type Item = Neighbor<Point>;

    fn next(&mut self) -> Option<Self::Item> {
        let d = self.iter.next()?;
        Some(self.p.neighbor(d))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
impl ExactSizeIterator for NeighborIterator {}

/// Human-readable compass abbreviations ordered by bit position (`0..8`).
const DIRECTION_NAMES: [&str; 8] = ["NW", "N", "NE", "W", "E", "SW", "S", "SE"];

/// All 8 compass directions ordered by bitmask position (`NW`, `N`, `NE`, `W`, `E`, `SW`, `S`, `SE`).
///
/// Opposite directions use bit-reversed masks (`mask.reverse_bits()`), allowing O(1)
/// reversal via [`Direction::reverse`].
pub const ALL_DIRECTIONS: [Direction; 8] = [
    // NW
    Direction {
        mask: 0b000_00_001,
        dx: -1,
        dy: -1,
    },
    // N
    Direction {
        mask: 0b000_00_010,
        dx: 0,
        dy: -1,
    },
    // NE
    Direction {
        mask: 0b000_00_100,
        dx: 1,
        dy: -1,
    },
    // W
    Direction {
        mask: 0b000_01_000,
        dx: -1,
        dy: 0,
    },
    // E
    Direction {
        mask: 0b000_10_000,
        dx: 1,
        dy: 0,
    },
    // SW
    Direction {
        mask: 0b001_00_000,
        dx: -1,
        dy: 1,
    },
    // S
    Direction {
        mask: 0b010_00_000,
        dx: 0,
        dy: 1,
    },
    // SE
    Direction {
        mask: 0b100_00_000,
        dx: 1,
        dy: 1,
    },
];

/// North-West (`dx = -1, dy = -1`).
pub const NW: Direction = ALL_DIRECTIONS[0];
/// North (`dx = 0, dy = -1`).
pub const N: Direction = ALL_DIRECTIONS[1];
/// North-East (`dx = 1, dy = -1`).
pub const NE: Direction = ALL_DIRECTIONS[2];
/// West (`dx = -1, dy = 0`).
pub const W: Direction = ALL_DIRECTIONS[3];
/// East (`dx = 1, dy = 0`).
pub const E: Direction = ALL_DIRECTIONS[4];
/// South-West (`dx = -1, dy = 1`).
pub const SW: Direction = ALL_DIRECTIONS[5];
/// South (`dx = 0, dy = 1`).
pub const S: Direction = ALL_DIRECTIONS[6];
/// South-East (`dx = 1, dy = 1`).
pub const SE: Direction = ALL_DIRECTIONS[7];

/// A compact 8-bit set of [`Direction`]s representing a subset of a cell's 8 neighbors.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DirectionSet(u8);

impl DirectionSet {
    /// Given a point in a fixed size field, return the set of neighbors, taking
    /// edges into consideration.
    pub fn all_neighbors(p: Point, width: u32, height: u32) -> DirectionSet {
        let mut mask = 0b111_11_111;
        if p.x == 0 {
            mask &= const { !(NW.mask | W.mask | SW.mask) };
        }
        if p.x >= width - 1 {
            mask &= const { !(NE.mask | E.mask | SE.mask) };
        }
        if p.y == 0 {
            mask &= const { !(NW.mask | N.mask | NE.mask) };
        }
        if p.y >= height - 1 {
            mask &= const { !(SW.mask | S.mask | SE.mask) };
        }
        DirectionSet(mask)
    }

    /// Creates an empty [`DirectionSet`].
    pub fn new() -> DirectionSet {
        DirectionSet::default()
    }

    /// Returns the number of directions currently in the set (`0..=8`).
    pub fn len(&self) -> u8 {
        self.0.count_ones() as u8
    }

    /// Returns `true` if the set contains no directions.
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Returns `true` if the set contains direction `d`.
    pub fn contains(&self, d: Direction) -> bool {
        self.0 & d != 0
    }

    /// Inserts direction `d` into the set, returning `true` if it was not already present.
    pub fn insert(&mut self, d: Direction) -> bool {
        if self.contains(d) {
            false
        } else {
            self.0 |= d;
            true
        }
    }

    /// Removes direction `d` from the set, returning `true` if it was present.
    pub fn remove(&mut self, d: Direction) -> bool {
        if self.contains(d) {
            self.0 &= !d;
            true
        } else {
            false
        }
    }

    /// Returns a new [`DirectionSet`] containing all directions in `self` or `other`.
    pub fn union(&self, other: DirectionSet) -> DirectionSet {
        DirectionSet(self.0 | other.0)
    }

    /// Returns a new [`DirectionSet`] containing directions in `self` that are not in `other`.
    pub fn difference(&self, other: DirectionSet) -> DirectionSet {
        DirectionSet(self.0 & !other.0)
    }

    /// Returns a new [`DirectionSet`] containing only directions present in both `self` and `other`.
    pub fn intersection(&self, other: DirectionSet) -> DirectionSet {
        DirectionSet(self.0 & other.0)
    }

    /// Retains only the directions specified by the predicate `f`.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(Direction) -> bool,
    {
        self.0 = self.iter().filter(|&d| f(d)).fold(0u8, |acc, d| acc | d);
    }

    /// Returns an iterator over all [`Direction`]s in `self`.
    pub fn iter(&self) -> DirectionSetIterator {
        DirectionSetIterator(BinaryIterator(self.0))
    }
}

impl BitOr for DirectionSet {
    type Output = DirectionSet;

    fn bitor(self, rhs: DirectionSet) -> Self::Output {
        self.union(rhs)
    }
}

impl BitAnd for DirectionSet {
    type Output = DirectionSet;

    fn bitand(self, rhs: DirectionSet) -> Self::Output {
        self.intersection(rhs)
    }
}

impl Sub for DirectionSet {
    type Output = DirectionSet;

    fn sub(self, rhs: DirectionSet) -> Self::Output {
        self.difference(rhs)
    }
}

impl FromIterator<Direction> for DirectionSet {
    fn from_iter<I: IntoIterator<Item = Direction>>(iter: I) -> DirectionSet {
        DirectionSet(iter.into_iter().fold(0u8, |acc, d| acc | d))
    }
}

impl IntoIterator for DirectionSet {
    type Item = Direction;
    type IntoIter = DirectionSetIterator;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Display for DirectionSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        let mut first = true;
        for d in self.iter() {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
            first = false;
        }
        write!(f, "]")
    }
}

/// A single compass direction represented by its bitmask and `(dx, dy)` coordinate offset.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Direction {
    mask: u8,
    dx: i8,
    dy: i8,
}

impl Direction {
    /// Returns the opposite [`Direction`] (e.g., [`NW`] $\leftrightarrow$ [`SE`]).
    pub fn reverse(&self) -> Direction {
        Direction {
            mask: self.mask.reverse_bits(),
            dx: -self.dx,
            dy: -self.dy,
        }
    }
}

impl Display for Direction {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            DIRECTION_NAMES[self.mask.trailing_zeros() as usize]
        )
    }
}

impl Not for Direction {
    type Output = u8;

    fn not(self) -> Self::Output {
        !self.mask
    }
}

impl BitOr for Direction {
    type Output = u8;

    fn bitor(self, rhs: Direction) -> Self::Output {
        self.mask | rhs.mask
    }
}

impl BitOr<u8> for Direction {
    type Output = u8;

    fn bitor(self, rhs: u8) -> Self::Output {
        self.mask | rhs
    }
}

impl BitOr<Direction> for u8 {
    type Output = u8;

    fn bitor(self, rhs: Direction) -> Self::Output {
        self | rhs.mask
    }
}

impl BitOrAssign for Direction {
    fn bitor_assign(&mut self, rhs: Direction) {
        self.mask |= rhs.mask
    }
}

impl BitOrAssign<u8> for Direction {
    fn bitor_assign(&mut self, rhs: u8) {
        self.mask |= rhs
    }
}

impl BitOrAssign<Direction> for u8 {
    fn bitor_assign(&mut self, rhs: Direction) {
        *self |= rhs.mask
    }
}

impl BitAnd for Direction {
    type Output = u8;

    fn bitand(self, rhs: Direction) -> Self::Output {
        self.mask & rhs.mask
    }
}

impl BitAnd<u8> for Direction {
    type Output = u8;

    fn bitand(self, rhs: u8) -> Self::Output {
        self.mask & rhs
    }
}

impl BitAnd<Direction> for u8 {
    type Output = u8;

    fn bitand(self, rhs: Direction) -> Self::Output {
        self & rhs.mask
    }
}

/// An iterator over the [`Direction`]s contained in a [`DirectionSet`].
#[derive(Debug, Clone, Copy)]
pub struct DirectionSetIterator(BinaryIterator);

impl Iterator for DirectionSetIterator {
    type Item = Direction;

    fn next(&mut self) -> Option<Self::Item> {
        let b = self.0.next()?;
        Some(ALL_DIRECTIONS[b.trailing_zeros() as usize])
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl ExactSizeIterator for DirectionSetIterator {}

/// An iterator that returns all the `1` digits in a binary number.
/// Ex. the values 11 in binary is 0b1011; this iterator will generate the
/// values: [0b0001, 0b0010, 0b1000]
#[derive(Debug, Clone, Copy)]
struct BinaryIterator(u8);

impl Iterator for BinaryIterator {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            return None;
        }
        let t = self.0;
        self.0 &= t - 1;
        Some(self.0 ^ t)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.count_ones() as usize;
        (len, Some(len))
    }
}
impl ExactSizeIterator for BinaryIterator {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_iterator_iterates() {
        assert_eq!(
            BinaryIterator(0b1011001).collect::<Vec<u8>>(),
            vec![0b1, 0b1000, 0b10000, 0b1000000]
        );
        assert_eq!(BinaryIterator(0).collect::<Vec<u8>>(), vec![]);
    }
}

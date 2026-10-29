#![allow(clippy::unusual_byte_groupings)]

//! 2D grid coordinates, compass directions, and bitset-backed neighbor iteration.

use std::{
    fmt::{self, Display, Formatter},
    marker::PhantomData,
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Sub, SubAssign},
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
    const DIRECTION_OFFSETS: [(i8, i8); 8] = [
        // NW
        (-1, -1),
        // N
        (0, -1),
        // NE
        (1, -1),
        // W
        (-1, 0),
        // E
        (1, 0),
        // SW
        (-1, 1),
        // S
        (0, 1),
        // SE
        (1, 1),
    ];
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
    #[inline]
    pub const fn to_index(self, width: u32) -> usize {
        self.x as usize + self.y as usize * width as usize
    }

    /// Returns an iterator over the neighboring [`Point`]s specified by `directions`.
    pub fn neighbors(&self, directions: DirectionSet) -> impl Iterator<Item = Neighbor<Point>> {
        NeighborIterator {
            offset_values: &Self::DIRECTION_OFFSETS,
            apply_offset: |d, (dx, dy)| self.to_neighbor(d, dx, dy),
            iter: directions.iter(),
            _result: PhantomData,
        }
    }

    /// Returns the adjacent [`Point`] in direction `to`, paired with the reverse [`Direction`]
    /// pointing from that neighbor back to `self`.
    #[inline]
    pub fn neighbor(&self, to: Direction) -> Neighbor<Point> {
        let (dx, dy) = Self::DIRECTION_OFFSETS[to.mask().trailing_zeros() as usize];
        self.to_neighbor(to, dx, dy)
    }

    fn to_neighbor(&self, to: Direction, dx: i8, dy: i8) -> Neighbor<Point> {
        Neighbor {
            neighbor: Point {
                x: self.x.wrapping_add_signed(dx as i32),
                y: self.y.wrapping_add_signed(dy as i32),
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
pub(super) struct NeighborIterator<'a, O, R, F> {
    pub(super) offset_values: &'a [O],
    pub(super) apply_offset: F,
    pub(super) iter: DirectionSetIterator,
    pub(super) _result: PhantomData<R>,
}

/// Associates a neighboring item `T` (such as a [`Point`] or cell reference) with the
/// [`Direction`] pointing from that neighbor back to the origin cell.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Neighbor<T: Copy> {
    /// The neighboring value or coordinate.
    pub neighbor: T,
    /// The direction from `neighbor` back to the origin cell.
    pub from: Direction,
}

impl<'a, O, R, F> Iterator for NeighborIterator<'a, O, R, F>
where
    O: Copy,
    R: Copy,
    F: FnMut(Direction, O) -> R,
{
    type Item = R;

    fn next(&mut self) -> Option<Self::Item> {
        let d = self.iter.next()?;
        Some((self.apply_offset)(
            d,
            self.offset_values[d.mask().trailing_zeros() as usize],
        ))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
impl<'a, O: Copy, R: Copy, F: FnMut(Direction, O) -> R> ExactSizeIterator
    for NeighborIterator<'a, O, R, F>
{
}

/// Human-readable compass abbreviations ordered by bit position (`0..8`).
const DIRECTION_NAMES: [&str; 8] = ["NW", "N", "NE", "W", "E", "SW", "S", "SE"];

/// All 8 compass directions ordered by bitmask position (`NW`, `N`, `NE`, `W`, `E`, `SW`, `S`, `SE`).
///
/// Opposite directions use bit-reversed masks (`mask.reverse_bits()`), allowing O(1)
/// reversal via [`Direction::reverse`].
pub const ALL_DIRECTIONS: [Direction; 8] = [
    // NW
    Direction(0b000_00_001),
    // N
    Direction(0b000_00_010),
    // NE
    Direction(0b000_00_100),
    // W
    Direction(0b000_01_000),
    // E
    Direction(0b000_10_000),
    // SW
    Direction(0b001_00_000),
    // S
    Direction(0b010_00_000),
    // SE
    Direction(0b100_00_000),
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
    pub const ALL: DirectionSet = DirectionSet(0xff);

    /// Creates an empty [`DirectionSet`].
    pub fn new() -> DirectionSet {
        DirectionSet(0)
    }

    /// Given a point in a fixed size field, return the set of neighbors, taking
    /// edges into consideration.
    pub fn all_neighbors(p: Point, width: u32, height: u32) -> DirectionSet {
        let mut neighbors = Self::ALL;
        if p.x == 0 {
            neighbors.0 &= const { !(NW.0 | W.0 | SW.0) };
        }
        if p.x >= width - 1 {
            neighbors.0 &= const { !(NE.0 | E.0 | SE.0) };
        }
        if p.y == 0 {
            neighbors.0 &= const { !(NW.0 | N.0 | NE.0) };
        }
        if p.y >= height - 1 {
            neighbors.0 &= const { !(SW.0 | S.0 | SE.0) };
        }
        neighbors
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
    #[inline]
    pub fn contains(&self, d: Direction) -> bool {
        self.0 & d.0 != 0
    }

    /// Inserts direction `d` into the set, returning `true` if it was not already present.
    #[inline]
    pub fn insert(&mut self, d: Direction) -> bool {
        if self.contains(d) {
            false
        } else {
            self.0 |= d.0;
            true
        }
    }

    /// Removes direction `d` from the set, returning `true` if it was present.
    #[inline]
    pub fn remove(&mut self, d: Direction) -> bool {
        if self.contains(d) {
            self.0 &= !d.0;
            true
        } else {
            false
        }
    }

    /// Returns a new [`DirectionSet`] containing all directions in `self` or `other`.
    pub fn union(&self, other: DirectionSet) -> DirectionSet {
        *self | other
    }

    /// Returns a new [`DirectionSet`] containing directions in `self` that are not in `other`.
    pub fn difference(&self, other: DirectionSet) -> DirectionSet {
        *self - other
    }

    /// Returns a new [`DirectionSet`] containing only directions present in both `self` and `other`.
    pub fn intersection(&self, other: DirectionSet) -> DirectionSet {
        *self & other
    }

    /// Retains only the directions specified by the predicate `f`.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(Direction) -> bool,
    {
        self.0 = self
            .iter()
            .filter(|&d| f(d))
            .fold(DirectionSet::new(), |acc, d| acc | d)
            .0;
    }

    /// Returns an iterator over all [`Direction`]s in `self`.
    pub fn iter(&self) -> DirectionSetIterator {
        DirectionSetIterator(BinaryIterator(self.0))
    }

    pub fn iter_over<T: Copy>(&self, values: &[T]) -> impl Iterator<Item = T> {
        self.iter()
            .map(|d| values[d.mask().trailing_zeros() as usize])
    }
}

impl Not for DirectionSet {
    type Output = DirectionSet;

    fn not(self) -> Self::Output {
        DirectionSet(!self.0)
    }
}

impl BitOr for DirectionSet {
    type Output = DirectionSet;

    fn bitor(self, rhs: DirectionSet) -> Self::Output {
        DirectionSet(self.0 | rhs.0)
    }
}

impl BitOr<Direction> for DirectionSet {
    type Output = DirectionSet;

    fn bitor(self, rhs: Direction) -> Self::Output {
        DirectionSet(self.0 | rhs.0)
    }
}

impl BitOrAssign for DirectionSet {
    fn bitor_assign(&mut self, rhs: DirectionSet) {
        self.0 |= rhs.0;
    }
}

impl BitOrAssign<Direction> for DirectionSet {
    fn bitor_assign(&mut self, rhs: Direction) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for DirectionSet {
    type Output = DirectionSet;

    fn bitand(self, rhs: DirectionSet) -> Self::Output {
        DirectionSet(self.0 & rhs.0)
    }
}

impl BitAnd<Direction> for DirectionSet {
    type Output = DirectionSet;

    fn bitand(self, rhs: Direction) -> Self::Output {
        DirectionSet(self.0 & rhs.0)
    }
}

impl BitAndAssign for DirectionSet {
    fn bitand_assign(&mut self, rhs: DirectionSet) {
        self.0 &= rhs.0;
    }
}

impl BitAndAssign<Direction> for DirectionSet {
    fn bitand_assign(&mut self, rhs: Direction) {
        self.0 &= rhs.0;
    }
}

impl BitXor for DirectionSet {
    type Output = DirectionSet;

    fn bitxor(self, rhs: DirectionSet) -> Self::Output {
        DirectionSet(self.0 ^ rhs.0)
    }
}

impl BitXor<Direction> for DirectionSet {
    type Output = DirectionSet;

    fn bitxor(self, rhs: Direction) -> Self::Output {
        DirectionSet(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for DirectionSet {
    fn bitxor_assign(&mut self, rhs: DirectionSet) {
        self.0 ^= rhs.0;
    }
}

impl BitXorAssign<Direction> for DirectionSet {
    fn bitxor_assign(&mut self, rhs: Direction) {
        self.0 ^= rhs.0;
    }
}

impl Sub for DirectionSet {
    type Output = DirectionSet;

    fn sub(self, rhs: DirectionSet) -> Self::Output {
        DirectionSet(self.0 & !rhs.0)
    }
}

impl Sub<Direction> for DirectionSet {
    type Output = DirectionSet;

    fn sub(self, rhs: Direction) -> Self::Output {
        DirectionSet(self.0 & !rhs.0)
    }
}

impl SubAssign for DirectionSet {
    fn sub_assign(&mut self, rhs: DirectionSet) {
        self.0 &= !rhs.0;
    }
}

impl SubAssign<Direction> for DirectionSet {
    fn sub_assign(&mut self, rhs: Direction) {
        self.0 &= !rhs.0;
    }
}

impl From<Direction> for DirectionSet {
    fn from(d: Direction) -> Self {
        DirectionSet(d.0)
    }
}

impl FromIterator<Direction> for DirectionSet {
    fn from_iter<I: IntoIterator<Item = Direction>>(iter: I) -> DirectionSet {
        iter.into_iter().fold(DirectionSet::new(), |acc, d| acc | d)
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

/// A single compass direction represented by a bitmask
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Direction(u8);

impl Direction {
    /// Returns the single-bit mask (`1 << 0..8`) identifying this direction.
    #[inline]
    pub const fn mask(&self) -> u8 {
        self.0
    }

    /// Returns the opposite [`Direction`] (e.g., [`NW`] $\leftrightarrow$ [`SE`]).
    #[inline]
    pub const fn reverse(&self) -> Direction {
        Direction(self.0.reverse_bits())
    }
}

impl Display for Direction {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", DIRECTION_NAMES[self.0.trailing_zeros() as usize])
    }
}

impl Not for Direction {
    type Output = DirectionSet;

    fn not(self) -> Self::Output {
        DirectionSet(!self.0)
    }
}

impl BitOr for Direction {
    type Output = DirectionSet;

    fn bitor(self, rhs: Direction) -> Self::Output {
        DirectionSet(self.0 | rhs.0)
    }
}

impl BitOr<DirectionSet> for Direction {
    type Output = DirectionSet;

    fn bitor(self, rhs: DirectionSet) -> Self::Output {
        DirectionSet(self.0 | rhs.0)
    }
}

impl BitAnd for Direction {
    type Output = DirectionSet;

    fn bitand(self, rhs: Direction) -> Self::Output {
        DirectionSet(self.0 & rhs.0)
    }
}

impl BitAnd<DirectionSet> for Direction {
    type Output = DirectionSet;

    fn bitand(self, rhs: DirectionSet) -> Self::Output {
        DirectionSet(self.0 & rhs.0)
    }
}

impl BitXor for Direction {
    type Output = DirectionSet;

    fn bitxor(self, rhs: Direction) -> Self::Output {
        DirectionSet(self.0 ^ rhs.0)
    }
}

impl BitXor<DirectionSet> for Direction {
    type Output = DirectionSet;

    fn bitxor(self, rhs: DirectionSet) -> Self::Output {
        DirectionSet(self.0 ^ rhs.0)
    }
}

impl Sub for Direction {
    type Output = DirectionSet;

    fn sub(self, rhs: Direction) -> Self::Output {
        DirectionSet(self.0 & !rhs.0)
    }
}

impl Sub<DirectionSet> for Direction {
    type Output = DirectionSet;

    fn sub(self, rhs: DirectionSet) -> Self::Output {
        DirectionSet(self.0 & !rhs.0)
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

    #[inline]
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
        let iter = BinaryIterator(0b1011001);
        assert_eq!(iter.len(), 4);
        assert_eq!(
            iter.collect::<Vec<u8>>(),
            vec![0b1, 0b1000, 0b10000, 0b1000000]
        );
        assert_eq!(BinaryIterator(0).collect::<Vec<u8>>(), vec![]);
    }

    #[test]
    fn test_point_index_round_trip_and_display() {
        let width = 10;
        let p = Point { x: 4, y: 7 };
        let idx = p.to_index(width);
        assert_eq!(idx, 74);
        assert_eq!(Point::from_index(idx, width), p);
        assert_eq!(p.to_string(), "(4, 7)");
    }

    #[test]
    fn test_direction_reverse_and_display() {
        let expected_pairs = [
            (NW, SE, "NW", "SE"),
            (N, S, "N", "S"),
            (NE, SW, "NE", "SW"),
            (W, E, "W", "E"),
        ];
        for (d, rev, name, rev_name) in expected_pairs {
            assert_eq!(d.reverse(), rev);
            assert_eq!(rev.reverse(), d);
            assert_eq!(d.to_string(), name);
            assert_eq!(rev.to_string(), rev_name);
        }
    }

    #[test]
    fn test_point_neighbor() {
        let origin = Point { x: 5, y: 5 };
        for d in ALL_DIRECTIONS {
            let step = origin.neighbor(d);
            assert_eq!(step.from, d.reverse());
            let back = step.neighbor.neighbor(step.from);
            assert_eq!(back.neighbor, origin);
            assert_eq!(back.from, d);
        }
    }

    #[test]
    fn test_direction_set_operations() {
        let mut set = DirectionSet::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert_eq!(set.to_string(), "[]");

        assert!(set.insert(NW));
        assert!(!set.insert(NW));
        assert!(set.insert(E));
        assert!(set.insert(SE));
        assert!(!set.is_empty());
        assert_eq!(set.len(), 3);
        assert!(set.contains(NW));
        assert!(!set.contains(N));
        assert_eq!(set.to_string(), "[NW, E, SE]");

        let other: DirectionSet = E | S | SW;
        assert_eq!(set | other, NW | E | SW | S | SE);
        assert_eq!(set & other, !!E);
        assert_eq!(set - other, NW | SE);

        assert!(set.remove(E));
        assert!(!set.remove(E));
        assert_eq!(set.len(), 2);

        set.retain(|d| d == NW);
        assert_eq!(set.into_iter().collect::<Vec<_>>(), vec![NW]);
    }

    #[test]
    fn test_all_neighbors_3x3_and_degenerate_grids() {
        // 3x3 grid corners, edges, and center
        assert_eq!(
            DirectionSet::all_neighbors(Point { x: 0, y: 0 }, 3, 3),
            E | S | SE
        );
        assert_eq!(
            DirectionSet::all_neighbors(Point { x: 2, y: 0 }, 3, 3),
            W | SW | S
        );
        assert_eq!(
            DirectionSet::all_neighbors(Point { x: 0, y: 2 }, 3, 3),
            N | NE | E
        );
        assert_eq!(
            DirectionSet::all_neighbors(Point { x: 2, y: 2 }, 3, 3),
            NW | N | W
        );
        assert_eq!(
            DirectionSet::all_neighbors(Point { x: 1, y: 0 }, 3, 3),
            W | E | SW | S | SE
        );
        assert_eq!(
            DirectionSet::all_neighbors(Point { x: 1, y: 1 }, 3, 3).len(),
            8
        );

        // 1x1 grid has no valid neighbors
        assert!(DirectionSet::all_neighbors(Point { x: 0, y: 0 }, 1, 1).is_empty());

        // 1x3 vertical strip (middle cell only has N and S)
        assert_eq!(
            DirectionSet::all_neighbors(Point { x: 0, y: 1 }, 1, 3),
            N | S
        );
    }

    #[test]
    fn test_direction_and_direction_set_bitwise_operators() {
        let a: DirectionSet = NW | N | E;
        let b: DirectionSet = N | E | SE;

        // Not (!) on DirectionSet and Direction
        assert_eq!(!DirectionSet::new(), DirectionSet::ALL);
        assert_eq!(!DirectionSet::ALL, DirectionSet::new());
        assert_eq!(!a, NE | W | SW | S | SE);
        assert_eq!(!NW, DirectionSet::ALL - NW);

        // BitOr (|) across (DirectionSet, DirectionSet), (DirectionSet, Direction),
        // (Direction, DirectionSet), and (Direction, Direction)
        assert_eq!(a | b, NW | N | E | SE);
        assert_eq!(a.union(b), a | b);
        assert_eq!(a | S, NW | N | E | S);
        assert_eq!(S | a, NW | N | E | S);
        assert_eq!(NW | SE, [NW, SE].into_iter().collect());

        // BitAnd (&) across all 4 combinations
        assert_eq!(a & b, N | E);
        assert_eq!(a.intersection(b), a & b);
        assert_eq!(a & N, !!N);
        assert_eq!(a & S, DirectionSet::new());
        assert_eq!(N & a, !!N);
        assert_eq!(S & a, DirectionSet::new());
        assert_eq!(NW & NW, !!NW);
        assert_eq!(NW & SE, DirectionSet::new());

        // BitXor (^) across all 4 combinations
        assert_eq!(a ^ b, NW | SE);
        assert_eq!(a ^ N, NW | E);
        assert_eq!(a ^ S, NW | N | E | S);
        assert_eq!(N ^ a, NW | E);
        assert_eq!(S ^ a, NW | N | E | S);
        assert_eq!(NW ^ SE, NW | SE);
        assert_eq!(NW ^ NW, DirectionSet::new());

        // Sub (-) across all 4 combinations
        assert_eq!(a - b, !!NW);
        assert_eq!(a.difference(b), a - b);
        assert_eq!(a - N, NW | E);
        assert_eq!(a - S, a);
        assert_eq!(N - a, DirectionSet::new());
        assert_eq!(S - a, !!S);
        assert_eq!(NW - SE, !!NW);
        assert_eq!(NW - NW, DirectionSet::new());
    }

    #[test]
    fn test_direction_set_bitwise_assign_operators() {
        let mut s = DirectionSet::new();

        // BitOrAssign (|=) with Direction and DirectionSet
        s |= NW;
        assert_eq!(s, !!NW);
        s |= E | SE;
        assert_eq!(s, NW | E | SE);

        // BitXorAssign (^=) with Direction and DirectionSet
        s ^= E; // removes E
        assert_eq!(s, NW | SE);
        s ^= N | SE; // adds N, removes SE
        assert_eq!(s, NW | N);

        // BitAndAssign (&=) with DirectionSet and Direction
        let mask_set: DirectionSet = N | E | S;
        s &= mask_set;
        assert_eq!(s, !!N);
        s |= S;
        s &= S;
        assert_eq!(s, !!S);

        // SubAssign (-=) with Direction and DirectionSet
        s |= NW | N | E;
        assert_eq!(s, NW | N | E | S);
        s -= N;
        assert_eq!(s, NW | E | S);
        s -= E | S;
        assert_eq!(s, !!NW);
    }
}

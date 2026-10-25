use std::{
    fmt::{self, Display, Formatter},
    ops::{BitAnd, BitOr, BitOrAssign, Not, Sub},
};

#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq)]
pub struct Point {
    pub x: u32,
    pub y: u32,
}

impl Point {
    pub fn neighbors(&self, directions: DirectionSet) -> NeighborIterator {
        NeighborIterator {
            p: *self,
            iter: directions.iter(),
        }
    }

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

#[derive(Debug, Clone, Copy)]
pub struct NeighborIterator {
    p: Point,
    iter: DirectionSetIterator,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Neighbor<T> {
    pub neighbor: T,
    pub from: Direction,
}

impl<T> Neighbor<T> {
    pub fn map<F, U>(self, f: F) -> Neighbor<U>
    where
        F: Fn(T) -> U,
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
}

const DIRECTION_NAMES: [&str; 8] = ["NW", "N", "NE", "W", "E", "SW", "S", "SE"];

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

pub const NW: Direction = ALL_DIRECTIONS[0];
pub const N: Direction = ALL_DIRECTIONS[1];
pub const NE: Direction = ALL_DIRECTIONS[2];
pub const W: Direction = ALL_DIRECTIONS[3];
pub const E: Direction = ALL_DIRECTIONS[4];
pub const SW: Direction = ALL_DIRECTIONS[5];
pub const S: Direction = ALL_DIRECTIONS[6];
pub const SE: Direction = ALL_DIRECTIONS[7];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DirectionSet(u8);

impl DirectionSet {
    /// Given a point in a fixed size field, return the set of neighbors, taking
    /// edges into consideration.
    pub fn all_neighbors(p: Point, width: u32, height: u32) -> DirectionSet {
        // Create a number to identify where the cell is. The values look like:'
        //
        // +3 +0    +0 +6
        // --------------
        //  4  1 ..  1  7  -- +1
        //  3  0 ..  0  6  -- +0
        //  ..             -- +0
        //  3  0 ..  0  6  -- +0
        //  5  2 ..  2  8  -- +2
        let cell_state = if p.x == 0 {
            3
        } else if p.x >= width - 1 {
            6
        } else {
            0
        } + if p.y == 0 {
            1
        } else if p.y >= height - 1 {
            2
        } else {
            0
        };

        DirectionSet(match cell_state {
            0 => {
                // middle cells - all directions
                const { NW.mask | N.mask | NE.mask | W.mask | E.mask | SW.mask | S.mask | SE.mask }
            }
            // top row
            1 => const { W.mask | E.mask | SW.mask | S.mask | SE.mask },
            // bottom row
            2 => const { NW.mask | N.mask | NE.mask | W.mask | E.mask },
            // left col
            3 => const { N.mask | NE.mask | E.mask | S.mask | SE.mask },
            // NW corner
            4 => const { E.mask | S.mask | SE.mask },
            // SW corner
            5 => const { N.mask | NE.mask | E.mask },
            // right col
            6 => const { NW.mask | N.mask | W.mask | SW.mask | S.mask },
            // NE corner
            7 => const { W.mask | SW.mask | S.mask },
            // SE corner
            8 => const { NW.mask | N.mask | W.mask },
            _ => unreachable!(),
        })
    }

    pub fn new() -> DirectionSet {
        DirectionSet::default()
    }

    pub fn len(&self) -> u8 {
        self.0.count_ones() as u8
    }

    pub fn contains(&self, d: Direction) -> bool {
        self.0 & d != 0
    }

    pub fn insert(&mut self, d: Direction) -> bool {
        if self.contains(d) {
            false
        } else {
            self.0 |= d;
            true
        }
    }

    pub fn remove(&mut self, d: Direction) -> bool {
        if self.contains(d) {
            self.0 &= !d;
            true
        } else {
            false
        }
    }

    pub fn union(&self, other: DirectionSet) -> DirectionSet {
        DirectionSet(self.0 | other.0)
    }

    pub fn difference(&self, other: DirectionSet) -> DirectionSet {
        DirectionSet(self.0 & !other.0)
    }

    pub fn intersection(&self, other: DirectionSet) -> DirectionSet {
        DirectionSet(self.0 & other.0)
    }

    pub fn retain<F>(&mut self, f: F)
    where
        F: Fn(Direction) -> bool,
    {
        self.0 = self.iter().filter(|&d| f(d)).fold(0u8, |acc, d| acc | d);
    }

    pub fn retain_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(Direction) -> bool,
    {
        self.0 = self.iter().filter(|&d| f(d)).fold(0u8, |acc, d| acc | d);
    }

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

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Direction {
    mask: u8,
    dx: i8,
    dy: i8,
}

impl Direction {
    fn reverse(&self) -> Direction {
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

#[derive(Debug, Clone, Copy)]
pub struct DirectionSetIterator(BinaryIterator);

impl Iterator for DirectionSetIterator {
    type Item = Direction;

    fn next(&mut self) -> Option<Self::Item> {
        let b = self.0.next()?;
        Some(ALL_DIRECTIONS[b.trailing_zeros() as usize])
    }
}

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
}

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

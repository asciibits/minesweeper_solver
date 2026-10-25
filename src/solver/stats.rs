//! Combinatorial utilities for computing solution counts on cell neighborhoods.

/// Maximum size of the precomputed Pascal's triangle (`0..=8` neighbors).
const TRIANGLE_SIZE: usize = 9;

/// Compile-time lookup table of binomial coefficients $\binom{n}{k}$ for $0 \le n, k \le 8$.
const PASCAL_TRIANGLE: [[u64; TRIANGLE_SIZE]; TRIANGLE_SIZE] = pascal_triangle();

/// Computes Pascal's triangle up to [`TRIANGLE_SIZE`] at compile time.
const fn pascal_triangle() -> [[u64; TRIANGLE_SIZE]; TRIANGLE_SIZE] {
    let mut triangle = [[0u64; TRIANGLE_SIZE]; TRIANGLE_SIZE];
    let mut n = 0;
    while n < triangle.len() {
        triangle[n][0] = 1;
        let mut k = 1;
        while k < n {
            triangle[n][k] = triangle[n - 1][k - 1] + triangle[n - 1][k];
            k += 1;
        }
        triangle[n][n] = 1;

        n += 1;
    }
    triangle
}

/// Returns the binomial coefficient $\binom{n}{k}$ ("`n` choose `k`") for up to $n, k = 8$.
///
/// Returns `0` if `n < 0` or `k < 0` (or if `k > n` within bounds).
///
/// # Panics
///
/// Panics if `n >= 9` or `k >= 9`.
pub fn choose(n: i8, k: i8) -> u64 {
    if n < 0 || k < 0 {
        return 0;
    }
    PASCAL_TRIANGLE[n as usize][k as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_choose() {
        assert_eq!(choose(7, 3), 35);
    }
}

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
#[inline]
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
    fn test_choose_known_values() {
        assert_eq!(choose(0, 0), 1);
        assert_eq!(choose(7, 3), 35);
        let row_8: Vec<u64> = (0..=8).map(|k| choose(8, k)).collect();
        assert_eq!(row_8, vec![1, 8, 28, 56, 70, 56, 28, 8, 1]);
    }

    #[test]
    fn test_choose_symmetry_and_identities() {
        for n in 0..=8 {
            assert_eq!(choose(n, 0), 1);
            assert_eq!(choose(n, n), 1);
            for k in 0..=n {
                assert_eq!(choose(n, k), choose(n, n - k));
            }
        }
    }

    #[test]
    fn test_choose_negative_and_k_gt_n() {
        assert_eq!(choose(-1, 2), 0);
        assert_eq!(choose(4, -1), 0);
        assert_eq!(choose(-5, -5), 0);
        assert_eq!(choose(3, 5), 0);
        assert_eq!(choose(0, 8), 0);
    }

    #[test]
    #[should_panic]
    fn test_choose_out_of_bounds_panics() {
        let _ = choose(9, 1);
    }
}

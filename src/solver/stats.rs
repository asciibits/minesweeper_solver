const TRIANGLE_SIZE: usize = 9;
const PASCAL_TRIANGLE: [[u64; TRIANGLE_SIZE]; TRIANGLE_SIZE] = pascal_triangle();

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

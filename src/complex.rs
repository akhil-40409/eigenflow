/// A complex number z = re + i*im
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    /// |z|^2 = a^2 + b^2
    pub fn norm_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn norm(&self) -> f64 {
        self.norm_sq().sqrt()
    }

    pub fn conjugate(&self) -> Self {
        Self { re: self.re, im: -self.im }
    }
}

impl std::fmt::Display for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{:.4}+{:.4}i", self.re, self.im)
        } else {
            write!(f, "{:.4}{:.4}i", self.re, self.im)
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { re: self.re + rhs.re, im: self.im + rhs.im }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_sq() {
        let z = Complex::new(3.0, 4.0);
        assert_eq!(z.norm_sq(), 25.0);
        assert_eq!(z.norm(), 5.0);
    }

    #[test]
    fn test_conjugate() {
        let z = Complex::new(1.0, -2.0);
        assert_eq!(z.conjugate(), Complex::new(1.0, 2.0));
    }

    #[test]
    fn test_multiply() {
        // (1+i)(1-i) = 1 - i^2 = 2
        let a = Complex::new(1.0, 1.0);
        let b = Complex::new(1.0, -1.0);
        let result = a * b;
        assert!((result.re - 2.0).abs() < 1e-10);
        assert!(result.im.abs() < 1e-10);
    }
}

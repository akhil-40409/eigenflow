use crate::complex::Complex;

/// A single-qubit gate represented as a 2x2 complex matrix.
/// Applied as: [new_0]   [m00 m01] [amp_0]
///             [new_1] = [m10 m11] [amp_1]
pub struct Gate {
    pub matrix: [[Complex; 2]; 2],
}

impl Gate {
    pub fn new(matrix: [[Complex; 2]; 2]) -> Self {
        Self { matrix }
    }

    // Pauli-X: flips |0⟩ ↔ |1⟩ (quantum NOT gate)
    pub fn x() -> Self {
        let o = Complex::one();
        let z = Complex::zero();
        Self::new([[z, o], [o, z]])
    }

    // Pauli-Z: leaves |0⟩ alone, flips the phase of |1⟩ to -|1⟩
    pub fn z() -> Self {
        let o = Complex::one();
        let z = Complex::zero();
        let m = Complex::new(-1.0, 0.0);
        Self::new([[o, z], [z, m]])
    }

    // Pauli-Y: combination of X and Z rotations, introduces imaginary phase
    pub fn y() -> Self {
        let z = Complex::zero();
        let i = Complex::new(0.0, 1.0);
        let mi = Complex::new(0.0, -1.0);
        Self::new([[z, mi], [i, z]])
    }

    // Hadamard: puts a qubit into equal superposition of |0⟩ and |1⟩
    // H|0⟩ = (|0⟩ + |1⟩)/√2,  H|1⟩ = (|0⟩ - |1⟩)/√2
    pub fn h() -> Self {
        let s = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);
        let ms = Complex::new(-1.0 / 2.0_f64.sqrt(), 0.0);
        Self::new([[s, s], [s, ms]])
    }

    // S (phase gate): leaves |0⟩ alone, multiplies |1⟩ by i
    pub fn s() -> Self {
        let o = Complex::one();
        let z = Complex::zero();
        let i = Complex::new(0.0, 1.0);
        Self::new([[o, z], [z, i]])
    }
}

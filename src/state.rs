use crate::complex::Complex;
use crate::gate::Gate;

/// Quantum state for n qubits stored as 2^n complex amplitudes.
/// Basis states are ordered in binary: |00⟩=0, |01⟩=1, |10⟩=2, |11⟩=3, etc.
pub struct StateVector {
    pub num_qubits: usize,
    pub amplitudes: Vec<Complex>,
}

impl StateVector {
    /// Initialize to the all-zero computational basis state |00...0⟩
    pub fn new(num_qubits: usize) -> Self {
        assert!(num_qubits >= 1, "need at least 1 qubit");
        let dim = 1 << num_qubits; // 2^n
        let mut amplitudes = vec![Complex::zero(); dim];
        amplitudes[0] = Complex::one(); // |00...0⟩ has amplitude 1
        Self { num_qubits, amplitudes }
    }

    /// Build a state directly from a list of amplitudes (must be length 2^n and normalized)
    pub fn from_amplitudes(amplitudes: Vec<Complex>) -> Self {
        let dim = amplitudes.len();
        assert!(dim.is_power_of_two(), "amplitude count must be a power of 2");
        let num_qubits = dim.trailing_zeros() as usize;
        let sv = Self { num_qubits, amplitudes };
        assert!(sv.is_normalized(), "state must be normalized");
        sv
    }

    /// Probability of measuring each basis state: P(i) = |amplitude[i]|^2
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_sq()).collect()
    }

    /// A valid quantum state must satisfy: sum of |amplitude|^2 == 1
    pub fn is_normalized(&self) -> bool {
        let total: f64 = self.amplitudes.iter().map(|a| a.norm_sq()).sum();
        (total - 1.0).abs() < 1e-9
    }

    /// Apply a single-qubit gate to the qubit at `qubit` (0 = leftmost/most significant).
    ///
    /// Only pairs of amplitudes that differ in bit `qubit` are mixed — all others untouched.
    /// mask isolates which bit corresponds to this qubit in the index, then we iterate over
    /// every index where that bit is 0 (the |0⟩ half) and derive the |1⟩ partner via i | mask.
    pub fn apply_gate(&mut self, qubit: usize, gate: &Gate) {
        assert!(qubit < self.num_qubits, "qubit index out of range");
        let mask = 1 << (self.num_qubits - 1 - qubit);
        let mut new_amps = self.amplitudes.clone();

        for i in 0..self.amplitudes.len() {
            if i & mask == 0 {
                let j = i | mask;
                let a0 = self.amplitudes[i];
                let a1 = self.amplitudes[j];
                new_amps[i] = gate.matrix[0][0] * a0 + gate.matrix[0][1] * a1;
                new_amps[j] = gate.matrix[1][0] * a0 + gate.matrix[1][1] * a1;
            }
        }
        self.amplitudes = new_amps;
    }

    /// Print each basis state with its amplitude and probability
    pub fn print(&self) {
        println!("State ({} qubit{}):", self.num_qubits, if self.num_qubits == 1 { "" } else { "s" });
        for (i, amp) in self.amplitudes.iter().enumerate() {
            let prob = amp.norm_sq();
            if prob > 1e-12 {
                // Format index as binary string padded to num_qubits digits
                println!("  |{:0>width$b}⟩  amplitude: {}  probability: {:.4}",
                    i, amp, prob, width = self.num_qubits);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::Gate;

    #[test]
    fn test_initial_state_is_zero() {
        let sv = StateVector::new(1);
        assert_eq!(sv.amplitudes[0], Complex::one());
        assert_eq!(sv.amplitudes[1], Complex::zero());
    }

    #[test]
    fn test_dimension() {
        // n qubits => 2^n amplitudes
        assert_eq!(StateVector::new(1).amplitudes.len(), 2);
        assert_eq!(StateVector::new(3).amplitudes.len(), 8);
    }

    #[test]
    fn test_initial_state_is_normalized() {
        assert!(StateVector::new(2).is_normalized());
    }

    #[test]
    fn test_probabilities_sum_to_one() {
        let sv = StateVector::new(3);
        let total: f64 = sv.probabilities().iter().sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_superposition_state() {
        // |+⟩ = (|0⟩ + |1⟩) / sqrt(2) — equal probability of 0 and 1
        let s = 1.0_f64 / 2.0_f64.sqrt();
        let sv = StateVector::from_amplitudes(vec![
            Complex::new(s, 0.0),
            Complex::new(s, 0.0),
        ]);
        assert!(sv.is_normalized());
        let probs = sv.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-9);
        assert!((probs[1] - 0.5).abs() < 1e-9);
    }

    #[test]
    #[should_panic]
    fn test_unnormalized_state_panics() {
        StateVector::from_amplitudes(vec![
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0), // not normalized
        ]);
    }

    // --- Gate tests ---

    #[test]
    fn test_x_gate_flips_zero_to_one() {
        // X|0⟩ = |1⟩: amplitude moves from index 0 to index 1
        let mut sv = StateVector::new(1);
        sv.apply_gate(0, &Gate::x());
        assert!((sv.amplitudes[0].re).abs() < 1e-10);
        assert!((sv.amplitudes[1].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_x_gate_is_its_own_inverse() {
        // XX|0⟩ = |0⟩
        let mut sv = StateVector::new(1);
        sv.apply_gate(0, &Gate::x());
        sv.apply_gate(0, &Gate::x());
        assert!((sv.amplitudes[0].re - 1.0).abs() < 1e-10);
        assert!((sv.amplitudes[1].re).abs() < 1e-10);
    }

    #[test]
    fn test_h_gate_creates_superposition() {
        // H|0⟩ = (|0⟩ + |1⟩)/√2: both amplitudes become 1/√2 ≈ 0.7071
        let mut sv = StateVector::new(1);
        sv.apply_gate(0, &Gate::h());
        let s = 1.0 / 2.0_f64.sqrt();
        assert!((sv.amplitudes[0].re - s).abs() < 1e-10);
        assert!((sv.amplitudes[1].re - s).abs() < 1e-10);
        assert!(sv.is_normalized());
    }

    #[test]
    fn test_h_gate_is_its_own_inverse() {
        // HH|0⟩ = |0⟩ — Hadamard is self-inverse
        let mut sv = StateVector::new(1);
        sv.apply_gate(0, &Gate::h());
        sv.apply_gate(0, &Gate::h());
        assert!((sv.amplitudes[0].re - 1.0).abs() < 1e-10);
        assert!(sv.amplitudes[1].re.abs() < 1e-10);
    }

    #[test]
    fn test_z_gate_flips_phase_of_one() {
        // Z|1⟩ = -|1⟩: apply X first to get |1⟩, then Z should negate its amplitude
        let mut sv = StateVector::new(1);
        sv.apply_gate(0, &Gate::x()); // now |1⟩
        sv.apply_gate(0, &Gate::z()); // Z|1⟩ = -|1⟩
        assert!((sv.amplitudes[1].re + 1.0).abs() < 1e-10); // amplitude is -1
    }

    #[test]
    fn test_gate_on_second_qubit_of_two() {
        // Apply H to qubit 1 (rightmost) of |00⟩
        // Expected: (|00⟩ + |01⟩)/√2 — only the right qubit is in superposition
        let mut sv = StateVector::new(2);
        sv.apply_gate(1, &Gate::h());
        let s = 1.0 / 2.0_f64.sqrt();
        assert!((sv.amplitudes[0].re - s).abs() < 1e-10); // |00⟩
        assert!((sv.amplitudes[1].re - s).abs() < 1e-10); // |01⟩
        assert!(sv.amplitudes[2].re.abs() < 1e-10);       // |10⟩ untouched
        assert!(sv.amplitudes[3].re.abs() < 1e-10);       // |11⟩ untouched
        assert!(sv.is_normalized());
    }

    #[test]
    fn test_gate_on_first_qubit_of_two() {
        // Apply H to qubit 0 (leftmost) of |00⟩
        // Expected: (|00⟩ + |10⟩)/√2 — only the left qubit is in superposition
        let mut sv = StateVector::new(2);
        sv.apply_gate(0, &Gate::h());
        let s = 1.0 / 2.0_f64.sqrt();
        assert!((sv.amplitudes[0].re - s).abs() < 1e-10); // |00⟩
        assert!(sv.amplitudes[1].re.abs() < 1e-10);       // |01⟩ untouched
        assert!((sv.amplitudes[2].re - s).abs() < 1e-10); // |10⟩
        assert!(sv.amplitudes[3].re.abs() < 1e-10);       // |11⟩ untouched
        assert!(sv.is_normalized());
    }
}

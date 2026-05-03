use crate::gate::Gate;
use crate::state::StateVector;

/// Run Grover's search on `num_qubits` qubits looking for `target`.
/// Returns the final state — measure it to get the answer with high probability.
pub fn run(num_qubits: usize, target: usize) -> StateVector {
    assert!(target < (1 << num_qubits), "target index out of range");

    let n = 1 << num_qubits; // total number of basis states
    // floor, not round — overshooting by one iteration undoes the amplitude buildup
    let iterations = ((std::f64::consts::PI / 4.0) * (n as f64).sqrt()) as usize;

    let mut sv = StateVector::new(num_qubits);

    // Step 1: put all qubits into equal superposition
    for q in 0..num_qubits {
        sv.apply_gate(q, &Gate::h());
    }

    // Step 2: alternate oracle and diffusion
    for _ in 0..iterations {
        oracle(&mut sv, target);
        diffusion(&mut sv);
    }

    sv
}

/// Flip the phase of the target amplitude — marks it as "the answer".
/// Every other amplitude is left untouched.
fn oracle(sv: &mut StateVector, target: usize) {
    sv.amplitudes[target].re = -sv.amplitudes[target].re;
    sv.amplitudes[target].im = -sv.amplitudes[target].im;
}

/// Invert every amplitude about the mean: amp[i] = 2*mean - amp[i].
/// This amplifies whatever stands out (the negated target) and suppresses the rest.
fn diffusion(sv: &mut StateVector) {
    let n = sv.amplitudes.len() as f64;
    let mean_re: f64 = sv.amplitudes.iter().map(|a| a.re).sum::<f64>() / n;
    let mean_im: f64 = sv.amplitudes.iter().map(|a| a.im).sum::<f64>() / n;
    for amp in sv.amplitudes.iter_mut() {
        amp.re = 2.0 * mean_re - amp.re;
        amp.im = 2.0 * mean_im - amp.im;
    }
}

/// Print the probability of each basis state, sorted highest first.
pub fn print_result(sv: &StateVector) {
    let mut probs: Vec<(usize, f64)> = sv.amplitudes.iter()
        .enumerate()
        .map(|(i, a)| (i, a.norm_sq()))
        .collect();
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Result ({} qubits):", sv.num_qubits);
    for (i, prob) in &probs {
        if *prob > 1e-6 {
            println!("  |{:0>width$b}⟩  {:.4}", i, prob, width = sv.num_qubits);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grover_2qubits_finds_target_0() {
        let sv = run(2, 0);
        assert!((sv.amplitudes[0].norm_sq() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_grover_2qubits_finds_target_1() {
        let sv = run(2, 1);
        assert!((sv.amplitudes[1].norm_sq() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_grover_2qubits_finds_target_3() {
        let sv = run(2, 3);
        assert!((sv.amplitudes[3].norm_sq() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_grover_3qubits_finds_target() {
        // 3 qubits, N=8: 2 iterations give ~97% probability
        let target = 5;
        let sv = run(3, target);
        assert!(sv.amplitudes[target].norm_sq() > 0.9);
    }

    #[test]
    fn test_grover_state_is_normalized() {
        let sv = run(3, 2);
        let total: f64 = sv.amplitudes.iter().map(|a| a.norm_sq()).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_grover_2qubits_all_other_amplitudes_zero() {
        // For 2 qubits, Grover's is exact — all non-target amplitudes collapse to 0
        let target = 2;
        let sv = run(2, target);
        for (i, amp) in sv.amplitudes.iter().enumerate() {
            if i != target {
                assert!(amp.norm_sq() < 1e-9, "index {} should be zero", i);
            }
        }
    }
}

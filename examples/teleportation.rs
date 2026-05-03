// cargo run --example teleportation
//
// Quantum teleportation transfers a qubit's state from Alice to Bob
// using a shared Bell pair and two classical bits.
//
// No information travels faster than light — Alice must send her 2 measurement
// bits to Bob over a classical channel before Bob can reconstruct the state.
//
// Qubits:
//   0 — Alice's data qubit (the state to teleport)
//   1 — Alice's half of the shared Bell pair
//   2 — Bob's half of the shared Bell pair
//
// Protocol:
//   1. Prepare a Bell pair on qubits 1 and 2
//   2. Alice entangles her data qubit (0) with her Bell qubit (1)
//   3. Alice measures qubits 0 and 1 — gets 2 classical bits
//   4. Bob applies corrections to qubit 2 based on Alice's bits
//   5. Qubit 2 is now in the original state of qubit 0

use eigenflow::state::StateVector;
use eigenflow::gate::Gate;

fn teleport(description: &str, prepare_alice: impl Fn(&mut StateVector)) {
    println!("Teleporting: {}", description);

    let mut sv = StateVector::new(3);

    // Prepare Alice's data qubit (qubit 0) in the state we want to teleport
    prepare_alice(&mut sv);
    println!("  Alice's qubit before teleport:");
    print_qubit_probs(&sv, 0);

    // Step 1: Create Bell pair between qubits 1 and 2
    sv.apply_gate(1, &Gate::h());
    sv.apply_cnot(1, 2);

    // Step 2: Alice's Bell measurement — entangle data qubit with her Bell qubit
    sv.apply_cnot(0, 1);
    sv.apply_gate(0, &Gate::h());

    // Step 3: Alice measures — collapses to 2 classical bits
    let m0 = sv.measure_qubit(0);
    let m1 = sv.measure_qubit(1);
    println!("  Alice measured: qubit0={} qubit1={}", m0, m1);

    // Step 4: Bob applies corrections based on Alice's classical message
    if m1 == 1 { sv.apply_gate(2, &Gate::x()); }
    if m0 == 1 { sv.apply_gate(2, &Gate::z()); }

    println!("  Bob's qubit after correction:");
    print_qubit_probs(&sv, 2);
    println!();
}

// Print the |0⟩ and |1⟩ probabilities for a single qubit by summing
// over all basis states where that qubit is 0 or 1.
fn print_qubit_probs(sv: &StateVector, qubit: usize) {
    let mask = 1 << (sv.num_qubits - 1 - qubit);
    let p1: f64 = sv.amplitudes.iter().enumerate()
        .filter(|(i, _)| i & mask != 0)
        .map(|(_, a)| a.norm_sq())
        .sum();
    let p0 = 1.0 - p1;
    println!("    P(|0⟩) = {:.4}  P(|1⟩) = {:.4}", p0, p1);
}

fn main() {
    println!("=== Quantum Teleportation ===\n");

    // Teleport |0⟩ — trivial but confirms the protocol works
    teleport("|0⟩ (P(0)=1.0, P(1)=0.0)", |_sv| {
        // qubit 0 already |0⟩ at initialization
    });

    // Teleport |1⟩
    teleport("|1⟩ (P(0)=0.0, P(1)=1.0)", |sv| {
        sv.apply_gate(0, &Gate::x());
    });

    // Teleport |+⟩ = (|0⟩ + |1⟩)/√2 — the interesting case
    // Bob should end up with 50/50 probability
    teleport("|+⟩ (P(0)=0.5, P(1)=0.5)", |sv| {
        sv.apply_gate(0, &Gate::h());
    });

    // Teleport an arbitrary state: S·H|0⟩
    // H gives |+⟩, then S rotates phase giving (|0⟩ + i|1⟩)/√2
    // Both qubits have P=0.5 but with a phase difference
    teleport("S·H|0⟩ (P(0)=0.5, P(1)=0.5, phase differs)", |sv| {
        sv.apply_gate(0, &Gate::h());
        sv.apply_gate(0, &Gate::s());
    });

    println!("In all cases Bob's probabilities match Alice's original state.");
    println!("The state was transferred — not copied. Alice's qubit was destroyed by measurement.");
}

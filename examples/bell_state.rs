// cargo run --example bell_state
//
// The Bell state is the simplest entangled state.
// Two qubits are correlated: measuring one instantly determines the other,
// regardless of distance. You never get |01⟩ or |10⟩ — only |00⟩ or |11⟩.

use eigenflow::circuit::Circuit;
use eigenflow::state::StateVector;

fn main() {
    println!("=== Bell State ===\n");

    // Circuit: H on qubit 0, then CNOT
    let mut c = Circuit::new(2);
    c.h(0).cnot(0, 1);
    c.diagram();
    println!();

    // The resulting state
    let mut sv = StateVector::new(2);
    c.run(&mut sv);
    sv.print();

    // Sample it: only correlated outcomes appear
    println!("\nSampling 400 times:");
    let mut counts: Vec<(String, usize)> = c.sample(400).into_iter().collect();
    counts.sort();
    for (outcome, count) in &counts {
        let bar: String = "█".repeat(count / 5);
        println!("  |{}⟩  {:>3}  {}", outcome, count, bar);
    }
    println!("\n|01⟩ and |10⟩ never appear — the qubits are entangled.");

    // Partial measurement: collapsing one qubit collapses both
    println!("\nPartial measurement of qubit 0:");
    let mut sv = StateVector::new(2);
    c.run(&mut sv);
    let outcome = sv.measure_qubit(0);
    println!("  qubit 0 collapsed to |{}⟩", outcome);
    println!("  full state after:");
    sv.print();
}

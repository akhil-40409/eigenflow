// cargo run --example ghz
//
// The GHZ (Greenberger-Horne-Zeilinger) state is 3-qubit entanglement.
// All three qubits are perfectly correlated: you always get all-0 or all-1,
// never any mixed outcome. Measuring any one qubit collapses all three.

use eigenflow::circuit::Circuit;
use eigenflow::state::StateVector;

fn main() {
    println!("=== GHZ State ===\n");

    let mut c = Circuit::new(3);
    c.h(0).cnot(0, 1).cnot(0, 2);
    c.diagram();
    println!();

    let mut sv = StateVector::new(3);
    c.run(&mut sv);
    sv.print();

    println!("\nSampling 400 times (only |000⟩ and |111⟩ should appear):");
    let mut counts: Vec<(String, usize)> = c.sample(400).into_iter().collect();
    counts.sort();
    for (outcome, count) in &counts {
        let bar: String = "█".repeat(count / 5);
        println!("  |{}⟩  {:>3}  {}", outcome, count, bar);
    }

    // Measure just qubit 0 — it collapses qubits 1 and 2 as well
    println!("\nMeasuring only qubit 0:");
    let mut sv = StateVector::new(3);
    c.run(&mut sv);
    let q0 = sv.measure_qubit(0);
    println!("  qubit 0 = |{}⟩", q0);
    println!("  state after (all three qubits determined by one measurement):");
    sv.print();
}

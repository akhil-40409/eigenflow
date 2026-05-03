// cargo run --example superposition
//
// The simplest quantum phenomenon: a single qubit in superposition.
// Before measurement it is genuinely in both states at once.
// After measurement it collapses to one — and stays there.

use eigenflow::circuit::Circuit;
use eigenflow::state::StateVector;
use eigenflow::gate::Gate;

fn main() {
    println!("=== Superposition ===\n");

    // A qubit starts in |0⟩ — definite state, always measures 0
    println!("Before H gate (|0⟩):");
    let mut sv = StateVector::new(1);
    sv.print();

    // Hadamard puts it in equal superposition
    sv.apply_gate(0, &Gate::h());
    println!("\nAfter H gate (|+⟩):");
    sv.print();

    // Sampling shows the 50/50 split
    println!("\nSampling |+⟩ 500 times:");
    let mut c = Circuit::new(1);
    c.h(0);
    let mut counts: Vec<(String, usize)> = c.sample(500).into_iter().collect();
    counts.sort();
    for (outcome, count) in &counts {
        let bar: String = "█".repeat(count / 5);
        println!("  |{}⟩  {:>3}  {}", outcome, count, bar);
    }

    // Key point: measuring collapses the state — a second measurement always agrees
    println!("\nMeasuring twice in a row:");
    let mut sv = StateVector::new(1);
    sv.apply_gate(0, &Gate::h());
    let first  = sv.measure();
    let second = sv.measure(); // state already collapsed
    println!("  first:  {}", first);
    println!("  second: {} (always the same — state collapsed)", second);
}

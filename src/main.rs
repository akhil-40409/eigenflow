mod complex;
mod gate;
mod state;

use gate::Gate;
use state::StateVector;

fn main() {
    println!("=== Phase 2: Single-Qubit Gates ===\n");

    // X|0⟩ = |1⟩
    println!("X|0⟩:");
    let mut sv = StateVector::new(1);
    sv.apply_gate(0, &Gate::x());
    sv.print();

    // H|0⟩ = |+⟩
    println!("\nH|0⟩:");
    let mut sv = StateVector::new(1);
    sv.apply_gate(0, &Gate::h());
    sv.print();

    // H on qubit 0 of |00⟩ — only left qubit enters superposition
    println!("\nH on qubit 0 of |00⟩:");
    let mut sv = StateVector::new(2);
    sv.apply_gate(0, &Gate::h());
    sv.print();
}

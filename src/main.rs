mod circuit;
mod complex;
mod gate;
mod grover;
mod state;

use circuit::Circuit;
use state::StateVector;

fn main() {
    // --- Circuit diagrams ---

    println!("=== Bell State Circuit ===\n");
    let mut bell = Circuit::new(2);
    bell.h(0).cnot(0, 1);
    bell.diagram();

    println!("\n=== GHZ State Circuit (3 qubits) ===\n");
    let mut ghz = Circuit::new(3);
    ghz.h(0).cnot(0, 1).cnot(0, 2);
    ghz.diagram();

    println!("\n=== SWAP Circuit ===\n");
    let mut swap = Circuit::new(3);
    swap.h(0).h(1).swap(0, 2);
    swap.diagram();

    // --- Grover's search ---

    println!("\n=== Grover's Search: 2 qubits, target |10⟩ ===\n");
    let sv = grover::run(2, 2); // index 2 = |10⟩
    grover::print_result(&sv);

    println!("\n=== Grover's Search: 3 qubits, target |101⟩ ===\n");
    let sv = grover::run(3, 5); // index 5 = |101⟩
    grover::print_result(&sv);

    println!("\n=== Grover's Search: 4 qubits, target |1001⟩ ===\n");
    let sv = grover::run(4, 9); // index 9 = |1001⟩
    grover::print_result(&sv);

    // --- Bell state step by step ---
    println!("\n=== Bell State: step by step ===\n");
    let mut sv = StateVector::new(2);
    let mut c = Circuit::new(2);
    c.h(0).cnot(0, 1);
    c.run(&mut sv);
    sv.print();
}

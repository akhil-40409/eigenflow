// cargo run --example grover_search
//
// Grover's algorithm finds a marked item in an unsorted list quadratically faster
// than any classical algorithm. For N items it takes O(√N) steps vs O(N) classical.
//
// We search over 2^n basis states. The target's probability spikes toward 1.0
// after floor(π/4 * √N) iterations of oracle + diffusion.

use eigenflow::grover;

fn main() {
    println!("=== Grover's Search ===\n");

    let cases = [
        (2, 2,  "|10⟩"),
        (3, 5,  "|101⟩"),
        (4, 9,  "|1001⟩"),
        (5, 17, "|10001⟩"),
    ];

    for (num_qubits, target, label) in cases {
        let n = 1 << num_qubits;
        let iters = ((std::f64::consts::PI / 4.0) * (n as f64).sqrt()) as usize;

        let sv = grover::run(num_qubits, target);
        let p_target = sv.amplitudes[target].norm_sq();

        println!(
            "{} qubits ({} states)  target {}  {} iterations  P(target) = {:.4}",
            num_qubits, n, label, iters, p_target
        );
    }

    println!("\n--- Detailed result for 3 qubits, target |101⟩ ---\n");
    let sv = grover::run(3, 5);
    grover::print_result(&sv);

    println!("\nClassical search over 8 items: up to 8 guesses.");
    println!("Grover's search over 8 items: 2 iterations, ~94% success.");
}

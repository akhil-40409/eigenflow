"""
Ground truth verification of eigenflow's Grover implementation using PennyLane.

Eigenflow uses direct amplitude manipulation (inversion about the mean).
PennyLane uses a gate-based circuit. Both are mathematically equivalent
so probabilities should agree to floating point precision.

Install: pip install pennylane
"""

import math
import pennylane as qml


# ── PennyLane Grover implementation ─────────────────────────────────────────

def phase_flip(wires):
    """Phase flip on |1...1⟩ for any number of qubits.
    Uses a Hadamard sandwich around a multi-controlled X."""
    if len(wires) == 1:
        qml.PauliZ(wires=wires[0])
    else:
        qml.Hadamard(wires=wires[-1])
        qml.MultiControlledX(wires=wires)
        qml.Hadamard(wires=wires[-1])


def oracle(wires, target_bits):
    """Flip the phase of the target basis state.
    Works by mapping the target to |1...1⟩, applying MCZ, then unmapping."""
    for i, bit in enumerate(target_bits):
        if bit == 0:
            qml.PauliX(wires=wires[i])
    phase_flip(wires)
    for i, bit in enumerate(target_bits):
        if bit == 0:
            qml.PauliX(wires=wires[i])


def diffusion(wires):
    """Grover diffusion operator — gate-based inversion about the mean.
    Equivalent to: amp[i] = 2*mean - amp[i] for all i."""
    for w in wires:
        qml.Hadamard(wires=w)
    for w in wires:
        qml.PauliX(wires=w)
    phase_flip(wires)
    for w in wires:
        qml.PauliX(wires=w)
    for w in wires:
        qml.Hadamard(wires=w)


def run_grover(num_qubits, target):
    n_states = 2 ** num_qubits
    # floor matches eigenflow: (PI/4 * sqrt(N)) as usize
    iterations = int(math.pi / 4 * math.sqrt(n_states))
    wires = list(range(num_qubits))

    # qubit 0 = MSB, matching eigenflow's convention
    target_bits = [(target >> (num_qubits - 1 - i)) & 1 for i in range(num_qubits)]

    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit():
        for w in wires:
            qml.Hadamard(wires=w)
        for _ in range(iterations):
            oracle(wires, target_bits)
            diffusion(wires)
        return qml.probs(wires=wires)

    return circuit(), iterations


# ── Comparison ───────────────────────────────────────────────────────────────

# Exact values from eigenflow's cargo run output
EIGENFLOW_RESULTS = {
    (2, 0): 1.0000,
    (2, 1): 1.0000,
    (2, 2): 1.0000,
    (2, 3): 1.0000,
    (3, 5): 0.9453,
    (4, 9): 0.9613,
}

TEST_CASES = [
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 5),
    (4, 9),
]

if __name__ == "__main__":
    print("Grover's Algorithm — eigenflow vs PennyLane\n")
    print(f"{'Case':<28} {'Iters':<7} {'PennyLane':<12} {'eigenflow':<12} {'Match'}")
    print("─" * 70)

    all_pass = True
    for num_qubits, target in TEST_CASES:
        probs, iters = run_grover(num_qubits, target)
        pl_prob = probs[target]
        ef_prob = EIGENFLOW_RESULTS[(num_qubits, target)]
        match = abs(pl_prob - ef_prob) < 1e-3
        if not match:
            all_pass = False

        label = f"{num_qubits} qubits, target |{target:0{num_qubits}b}⟩"
        status = "✓" if match else "✗"
        print(f"{label:<28} {iters:<7} {pl_prob:<12.4f} {ef_prob:<12.4f} {status}")

    print()
    if all_pass:
        print("All cases match.")
    else:
        print("Some cases differ — check implementation.")

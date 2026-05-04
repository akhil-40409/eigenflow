# eigenflow

A lightweight quantum computer simulator built in Rust.

## How it works

A real quantum computer manipulates physical qubits whose states are governed by quantum mechanics. A simulator fakes this on classical hardware by tracking the math directly.

**State vector simulation** is the core idea: an n-qubit system is fully described by 2^n complex numbers called amplitudes. Each amplitude corresponds to one possible classical outcome (e.g. `|01⟩`, `|10⟩`), and its magnitude squared gives the probability of measuring that outcome. The state starts as `|00...0⟩` — all amplitude on the all-zeros basis state — and gates transform the amplitudes.

### Applying a gate

A single-qubit gate is a 2×2 complex matrix. To apply it to qubit `q` of an n-qubit state, you pair up every amplitude with its partner that differs only in bit `q`, then multiply each pair by the matrix. No other amplitudes are touched. A controlled gate (like CNOT) is the same idea but only acts on pairs where the control qubit's bit is 1.

### Measurement

Measuring collapses the probabilistic state to a single classical outcome. The simulator samples an index from the probability distribution (|amplitude|²), sets that amplitude to 1 and all others to 0. A partial measurement (single qubit) zeroes out inconsistent amplitudes and renormalizes the rest.

### Circuit execution

A `Circuit` is an ordered list of operations. `run()` applies them in sequence to a `StateVector`. `sample(shots)` runs the whole circuit from `|0⟩` repeatedly and counts outcomes, returning a histogram like `{"00": 512, "11": 488}`.

### Grover's algorithm

Grover's search finds a marked item in an unsorted list of N items in O(√N) steps instead of O(N). The simulator implements it as:
1. Hadamard all qubits → equal superposition over all N states
2. Repeat `⌊π/4 · √N⌋` times:
   - **Oracle**: negate the target amplitude (marks it)
   - **Diffusion**: invert all amplitudes about their mean (amplifies the marked one)

Each iteration steers probability toward the target. For 2 qubits (N=4) it's exact in one step; for 3 qubits (N=8) it reaches ~97% in two steps.

---

## Pros

- **Exact**: the full amplitude vector is always in memory, so every gate and measurement is numerically precise with no approximation.
- **Simple**: the core data structure is just a `Vec<Complex>` and gates are matrix multiplies over pairs of elements — easy to understand and extend.
- **Verifiable**: because the state is fully observable, you can inspect amplitudes directly in tests, which makes the simulator straightforward to validate.
- **No physical noise**: ideal for learning and algorithm prototyping where you want to isolate logic from hardware imperfections.

## Cons

- **Exponential memory**: storing 2^n amplitudes means memory doubles with every qubit. 30 qubits needs ~8 GB; 40 qubits is infeasible on a laptop.
- **Exponential time**: most gate operations touch O(2^n) amplitudes, so runtime also scales exponentially.
- **No parallelism**: the current implementation is single-threaded, leaving significant performance on the table for larger circuits.
- **No noise modeling**: real quantum hardware has decoherence, gate errors, and readout errors — none of which are simulated here.
- **Not sparse-aware**: circuits that only entangle a few qubits could be represented much more efficiently, but this implementation always allocates the full 2^n vector.

---

## Run

```
cargo run
```

## Test

```
cargo test
```

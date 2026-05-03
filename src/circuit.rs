use std::collections::HashMap;
use crate::gate::Gate;
use crate::state::StateVector;

// Each variant represents one operation in the circuit.
// An enum in Rust is a type that can be exactly one of its listed variants.
// Tuple variants like H(usize) store data inline — H(0) means "H gate on qubit 0".
// Struct variants like Cnot { control, target } use named fields for clarity.
pub enum Op {
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    S(usize),
    Cnot { control: usize, target: usize },
    Swap(usize, usize),
}

pub struct Circuit {
    pub num_qubits: usize,
    ops: Vec<Op>, // sequence of operations to apply in order
}

impl Circuit {
    pub fn new(num_qubits: usize) -> Self {
        Self { num_qubits, ops: Vec::new() }
    }

    // Each method pushes an Op onto the list and returns &mut Self so calls can be chained:
    // circuit.h(0).cnot(0, 1).h(1)
    // &mut Self = mutable borrow of the circuit, returned so the next call has something to chain on

    pub fn h(&mut self, qubit: usize) -> &mut Self {
        assert!(qubit < self.num_qubits, "qubit index out of range");
        self.ops.push(Op::H(qubit));
        self
    }

    pub fn x(&mut self, qubit: usize) -> &mut Self {
        assert!(qubit < self.num_qubits, "qubit index out of range");
        self.ops.push(Op::X(qubit));
        self
    }

    pub fn y(&mut self, qubit: usize) -> &mut Self {
        assert!(qubit < self.num_qubits, "qubit index out of range");
        self.ops.push(Op::Y(qubit));
        self
    }

    pub fn z(&mut self, qubit: usize) -> &mut Self {
        assert!(qubit < self.num_qubits, "qubit index out of range");
        self.ops.push(Op::Z(qubit));
        self
    }

    pub fn s(&mut self, qubit: usize) -> &mut Self {
        assert!(qubit < self.num_qubits, "qubit index out of range");
        self.ops.push(Op::S(qubit));
        self
    }

    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        assert!(control < self.num_qubits && target < self.num_qubits, "qubit index out of range");
        assert_ne!(control, target, "control and target must differ");
        self.ops.push(Op::Cnot { control, target });
        self
    }

    pub fn swap(&mut self, qubit_a: usize, qubit_b: usize) -> &mut Self {
        assert!(qubit_a < self.num_qubits && qubit_b < self.num_qubits, "qubit index out of range");
        assert_ne!(qubit_a, qubit_b, "swap qubits must differ");
        self.ops.push(Op::Swap(qubit_a, qubit_b));
        self
    }

    /// Execute every operation in order on the given state.
    /// `match` is Rust's pattern matching — it must cover every variant of the enum,
    /// so the compiler guarantees we never forget to handle a case.
    pub fn run(&self, state: &mut StateVector) {
        assert_eq!(state.num_qubits, self.num_qubits, "circuit and state qubit count must match");
        for op in &self.ops {
            match op {
                Op::H(q)                      => state.apply_gate(*q, &Gate::h()),
                Op::X(q)                      => state.apply_gate(*q, &Gate::x()),
                Op::Y(q)                      => state.apply_gate(*q, &Gate::y()),
                Op::Z(q)                      => state.apply_gate(*q, &Gate::z()),
                Op::S(q)                      => state.apply_gate(*q, &Gate::s()),
                Op::Cnot { control, target }  => state.apply_cnot(*control, *target),
                Op::Swap(a, b)                => state.apply_swap(*a, *b),
            }
        }
    }

    /// Print the circuit as a list of operations
    /// Run the circuit `shots` times from |0⟩, measure each run, and tally the results.
    /// Keys are the binary string of the outcome (e.g. "01", "11"), values are counts.
    /// This is how you observe a quantum circuit statistically.
    pub fn sample(&self, shots: usize) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for _ in 0..shots {
            let mut sv = StateVector::new(self.num_qubits);
            self.run(&mut sv);
            let outcome = sv.measure();
            let key = format!("{:0>width$b}", outcome, width = self.num_qubits);
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    pub fn print(&self) {
        println!("Circuit ({} qubit{}, {} ops):",
            self.num_qubits,
            if self.num_qubits == 1 { "" } else { "s" },
            self.ops.len()
        );
        for op in &self.ops {
            match op {
                Op::H(q)                     => println!("  H({})", q),
                Op::X(q)                     => println!("  X({})", q),
                Op::Y(q)                     => println!("  Y({})", q),
                Op::Z(q)                     => println!("  Z({})", q),
                Op::S(q)                     => println!("  S({})", q),
                Op::Cnot { control, target } => println!("  CNOT(control={}, target={})", control, target),
                Op::Swap(a, b)               => println!("  SWAP({}, {})", a, b),
            }
        }
    }

    /// Print the circuit as an ASCII diagram.
    /// Qubit wires run left to right. Vertical bars connect multi-qubit gates.
    pub fn diagram(&self) {
        let n = self.num_qubits;
        // display lines: even indices are qubit wires, odd are vertical connectors between them
        let num_lines = if n == 1 { 1 } else { 2 * n - 1 };
        let prefix_w = format!("q{}: ─", n - 1).len();

        let mut lines: Vec<String> = (0..num_lines).map(|i| {
            if i % 2 == 0 {
                format!("{:─<width$}", format!("q{}: ─", i / 2), width = prefix_w)
            } else {
                " ".repeat(prefix_w)
            }
        }).collect();

        for op in &self.ops {
            match op {
                Op::H(q)                     => draw_single(&mut lines, n, *q, "H"),
                Op::X(q)                     => draw_single(&mut lines, n, *q, "X"),
                Op::Y(q)                     => draw_single(&mut lines, n, *q, "Y"),
                Op::Z(q)                     => draw_single(&mut lines, n, *q, "Z"),
                Op::S(q)                     => draw_single(&mut lines, n, *q, "S"),
                Op::Cnot { control, target } => draw_cnot(&mut lines, n, *control, *target),
                Op::Swap(a, b)               => draw_swap(&mut lines, n, *a, *b),
            }
        }

        for line in &lines {
            println!("{}", line);
        }
    }
}

// Each op appends a fixed-width column to every display line.
// Qubit wires are at even indices; vertical connectors are at odd indices.

fn draw_single(lines: &mut Vec<String>, n: usize, qubit: usize, sym: &str) {
    for i in 0..(2 * n - 1) {
        if i == 2 * qubit   { lines[i].push_str(sym); lines[i].push_str("──"); }
        else if i % 2 == 0  { lines[i].push_str("───"); }
        else                 { lines[i].push_str("   "); }
    }
}

fn draw_cnot(lines: &mut Vec<String>, n: usize, control: usize, target: usize) {
    let lo = control.min(target);
    let hi = control.max(target);
    for i in 0..(2 * n - 1) {
        if i == 2 * control        { lines[i].push_str("●──"); }
        else if i == 2 * target    { lines[i].push_str("⊕──"); }
        else if i % 2 == 0         { lines[i].push_str("───"); }
        else {
            // connector row: draw vertical bar only between control and target
            if i / 2 >= lo && i / 2 < hi { lines[i].push_str("│  "); }
            else                          { lines[i].push_str("   "); }
        }
    }
}

fn draw_swap(lines: &mut Vec<String>, n: usize, a: usize, b: usize) {
    let lo = a.min(b);
    let hi = a.max(b);
    for i in 0..(2 * n - 1) {
        if i == 2 * a || i == 2 * b { lines[i].push_str("×──"); }
        else if i % 2 == 0           { lines[i].push_str("───"); }
        else {
            if i / 2 >= lo && i / 2 < hi { lines[i].push_str("│  "); }
            else                          { lines[i].push_str("   "); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::StateVector;

    #[test]
    fn test_empty_circuit_leaves_state_unchanged() {
        let mut sv = StateVector::new(2);
        let c = Circuit::new(2);
        c.run(&mut sv);
        assert!((sv.amplitudes[0].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bell_state_via_circuit() {
        // H on qubit 0, then CNOT(0,1) — same as the manual Phase 3 test
        let mut c = Circuit::new(2);
        c.h(0).cnot(0, 1);

        let mut sv = StateVector::new(2);
        c.run(&mut sv);

        let s = 1.0 / 2.0_f64.sqrt();
        assert!((sv.amplitudes[0].re - s).abs() < 1e-10); // |00⟩
        assert!(sv.amplitudes[1].re.abs() < 1e-10);        // |01⟩ = 0
        assert!(sv.amplitudes[2].re.abs() < 1e-10);        // |10⟩ = 0
        assert!((sv.amplitudes[3].re - s).abs() < 1e-10); // |11⟩
    }

    #[test]
    fn test_double_x_is_identity() {
        // XX|0⟩ = |0⟩ — applying X twice cancels out
        let mut c = Circuit::new(1);
        c.x(0).x(0);

        let mut sv = StateVector::new(1);
        c.run(&mut sv);
        assert!((sv.amplitudes[0].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_double_h_is_identity() {
        let mut c = Circuit::new(1);
        c.h(0).h(0);

        let mut sv = StateVector::new(1);
        c.run(&mut sv);
        assert!((sv.amplitudes[0].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ghz_state() {
        // GHZ state: H(0), CNOT(0,1), CNOT(0,2) → (|000⟩ + |111⟩)/√2
        // A 3-qubit entangled state — measuring any qubit collapses all three
        let mut c = Circuit::new(3);
        c.h(0).cnot(0, 1).cnot(0, 2);

        let mut sv = StateVector::new(3);
        c.run(&mut sv);

        let s = 1.0 / 2.0_f64.sqrt();
        assert!((sv.amplitudes[0].re - s).abs() < 1e-10); // |000⟩
        assert!((sv.amplitudes[7].re - s).abs() < 1e-10); // |111⟩ = index 7
        // all others zero
        for i in 1..7 {
            assert!(sv.amplitudes[i].re.abs() < 1e-10);
        }
    }

    #[test]
    fn test_circuit_matches_manual_application() {
        // Circuit and manual gate calls should produce identical states
        let mut c = Circuit::new(2);
        c.h(0).x(1).cnot(0, 1);

        let mut sv_circuit = StateVector::new(2);
        c.run(&mut sv_circuit);

        let mut sv_manual = StateVector::new(2);
        sv_manual.apply_gate(0, &Gate::h());
        sv_manual.apply_gate(1, &Gate::x());
        sv_manual.apply_cnot(0, 1);

        for i in 0..4 {
            assert!((sv_circuit.amplitudes[i].re - sv_manual.amplitudes[i].re).abs() < 1e-10);
            assert!((sv_circuit.amplitudes[i].im - sv_manual.amplitudes[i].im).abs() < 1e-10);
        }
    }

    #[test]
    #[should_panic]
    fn test_qubit_mismatch_panics() {
        let c = Circuit::new(2);
        let mut sv = StateVector::new(3); // wrong number of qubits
        c.run(&mut sv);
    }

    // --- Measurement tests ---

    #[test]
    fn test_measure_zero_state_always_gives_zero() {
        // |0⟩ has 100% probability of measuring 0 — no randomness here
        let mut sv = StateVector::new(1);
        let outcome = sv.measure();
        assert_eq!(outcome, 0);
        assert!((sv.amplitudes[0].re - 1.0).abs() < 1e-10); // collapsed to |0⟩
    }

    #[test]
    fn test_measure_one_state_always_gives_one() {
        let mut sv = StateVector::new(1);
        sv.apply_gate(0, &Gate::x()); // |1⟩
        let outcome = sv.measure();
        assert_eq!(outcome, 1);
        assert!((sv.amplitudes[1].re - 1.0).abs() < 1e-10); // collapsed to |1⟩
    }

    #[test]
    fn test_measure_collapses_superposition() {
        // After measuring |+⟩, the state must be fully collapsed (prob 1 on one outcome)
        let mut sv = StateVector::new(1);
        sv.apply_gate(0, &Gate::h()); // |+⟩
        sv.measure();
        let total_prob: f64 = sv.amplitudes.iter().map(|a| a.norm_sq()).sum();
        assert!((total_prob - 1.0).abs() < 1e-10);
        // one amplitude is 1, the other is 0
        let nonzero = sv.amplitudes.iter().filter(|a| a.norm_sq() > 0.5).count();
        assert_eq!(nonzero, 1);
    }

    #[test]
    fn test_bell_state_sample_only_gives_correlated_outcomes() {
        // Bell state can only produce |00⟩ or |11⟩, never |01⟩ or |10⟩
        let mut c = Circuit::new(2);
        c.h(0).cnot(0, 1);
        let counts = c.sample(200);
        assert!(!counts.contains_key("01"), "got |01⟩ from Bell state");
        assert!(!counts.contains_key("10"), "got |10⟩ from Bell state");
    }

    #[test]
    fn test_sample_plus_state_is_roughly_half_half() {
        // H|0⟩ should give 0 and 1 each about 50% of the time over many shots
        let mut c = Circuit::new(1);
        c.h(0);
        let counts = c.sample(1000);
        let zeros = *counts.get("0").unwrap_or(&0);
        let ones  = *counts.get("1").unwrap_or(&0);
        // allow wide margin — just checking it's not wildly skewed
        assert!(zeros > 350 && zeros < 650, "expected ~500 zeros, got {}", zeros);
        assert!(ones  > 350 && ones  < 650, "expected ~500 ones, got {}", ones);
    }

    #[test]
    fn test_measure_qubit_collapses_and_renormalizes() {
        // Measure qubit 0 of Bell state — whichever outcome, state must still be normalized
        // and the two qubits must agree (entanglement collapses both)
        let mut sv = StateVector::new(2);
        sv.apply_gate(0, &Gate::h());
        sv.apply_cnot(0, 1);
        let outcome = sv.measure_qubit(0);
        let total_prob: f64 = sv.amplitudes.iter().map(|a| a.norm_sq()).sum();
        assert!((total_prob - 1.0).abs() < 1e-9, "state not normalized after partial measure");
        // if qubit 0 collapsed to |0⟩, only |00⟩ should have amplitude; if |1⟩, only |11⟩
        if outcome == 0 {
            assert!((sv.amplitudes[0].re - 1.0).abs() < 1e-9);
        } else {
            assert!((sv.amplitudes[3].re - 1.0).abs() < 1e-9);
        }
    }
}

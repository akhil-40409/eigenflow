# QAOA (MaxCut)

Farhi, Goldstone, Gutmann — [arXiv:1411.4028](https://arxiv.org/abs/1411.4028)

Find a cut that maximizes edges across the partition. Encode the graph in
a cost Hamiltonian, mix with ΣX, optimize the angles.

## Circuit

```
H_C = ½ Σ_{(i,j)∈E} (Z_i Z_j − I)
H_M = Σ_i X_i

|0…0⟩ ─ H⊗n ─ [e^{-iγ H_C} e^{-iβ H_M}]^p ─ ⟨H_C⟩
```

Minimize ⟨H_C⟩. Sample for a bitstring. Cut size = edges with different bits.

## Run

```bash
python examples/train_maxcut.py
```

## Files

```
eigenflow/qaoa/circuit.py
eigenflow/qaoa/train.py
```

## Sharp bits

- PennyLane `qml.qaoa.cost_layer` / `mixer_layer` on Lightning + `@qjit`.
- Exact statevector for energy. Sampling uses shots (eager).
- Demo: [Intro to QAOA](https://pennylane.ai/demos/tutorial_qaoa_intro).

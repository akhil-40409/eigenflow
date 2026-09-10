# Data re-uploading

Pérez-Salinas et al. — [arXiv:1907.02085](https://arxiv.org/abs/1907.02085)

A net reuses `x` at every neuron. A circuit cannot clone a state. So we
re-upload the classical `x` every layer, interleaved with trainable rotations.

One qubit + enough layers ≈ universal classifier (UAT sense). Not a
quantum-advantage claim.

## Circuit

```
split        L_i = U(θ_i) U(x)
compressed   L_i = U(θ_i + ω_i ⊙ x)     # default; same expressivity, half depth

|0⟩ ─ L_1(x) ─ … ─ L_L(x) ─ ⟨Z⟩
```

`U = Rot`. Pad `x` to 3. Labels at Bloch poles. Loss = mean (1 − F_y)².

## Run

```bash
python examples/train_circle.py
python examples/train_circle.py --layers 6 --steps 120 --plot
```

Circle-in-a-square, radius √(2/π). First `@qjit` call compiles.

## Files

```
eigenflow/reuploading/circuit.py
eigenflow/reuploading/data.py
eigenflow/reuploading/train.py
```

## Sharp bits

- `lightning.qubit` + Catalyst. Compile QNode, then `jax.vmap`.
- Not hardware. Not faster than an MLP on a circle.

# QKAN (DARUAN)

Jiang et al. — [arXiv:2509.14026](https://arxiv.org/abs/2509.14026)

A KAN edge is a 1-qubit data-reuploading activation (DARUAN / QVAF) plus a
SiLU residual.

```
φ(x) = w_b SiLU(x) + ⟨Z⟩(U(x; θ))
```

## Circuit (per edge)

```
n_reps times:  RY(w0·x + w1)  RZ(w2)
measure ⟨Z⟩
```

Live PennyLane QNode, Catalyst `@qjit`, `jax.vmap` over batch and edges.

## Run

```bash
python examples/train_qkan.py
python examples/train_qkan.py --task sinc
```

Uses Feynman / special datasets from `eigenflow.datasets`.

## Files

```
eigenflow/qkan/model.py
eigenflow/qkan/train.py
```

## Sharp bits

- Many edges ⇒ many compiled 1-qubit calls. Start tiny (`[2, 4, 1]`).
- Not hardware. Not a quantum-advantage claim.

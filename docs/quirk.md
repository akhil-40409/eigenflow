# QuIRK

Sharma et al. — [arXiv:2510.08650](https://arxiv.org/abs/2510.08650)

KAN with single-qubit re-uploading edges. Encode with RY(x), train RZ/RX.
Rescale activations to [0, π] between layers. No SiLU residual (unlike QKAN).

## Circuit (per edge)

```
n_reps times:  RY(x)  RZ(θ0)  RX(θ1)
measure ⟨Z⟩
```

Optional dense head on the last layer.

## Run

```bash
python examples/train_quirk.py
python examples/train_quirk.py --task I.12.1
```

## Files

```
eigenflow/quirk/model.py
eigenflow/quirk/train.py
```

## Sharp bits

- Same stack as QKAN: Lightning + Catalyst + vmap.
- “Quantum-inspired”: factorizable 1-qubit units.

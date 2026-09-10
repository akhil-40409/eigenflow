# Stack

PennyLane writes the circuit. JAX does arrays and grads. Catalyst compiles
the hot path. Optax does Adam.

```
Optax
  └─ jax.grad / jax.vmap
       └─ @qjit QNode          # Catalyst binary
            └─ lightning.qubit # not default.qubit
```

## Rules

1. Compiled path uses `lightning.qubit`. Catalyst refuses `default.qubit`.
2. Compile the QNode. Batch + Adam stay in JAX (`vmap` / `grad`).
3. Do not wrap a batched `qml.for_loop` in `qml.grad` — Catalyst 0.15 dies.
4. `n_layers` / `n_reps` are Python ints so they unroll under `@qjit`.

## Helpers

`eigenflow.backends` — `make_device`, `make_qnode`, Catalyst device check.

## Install

```bash
pip install -e ".[dev]"
```

macOS needs Xcode CLT (`clang`). Let `pennylane-catalyst` pick the JAX pin.

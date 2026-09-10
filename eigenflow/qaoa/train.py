"""Train MaxCut QAOA. Compile the energy QNode; Adam in JAX."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import optax
import pennylane as qml

from eigenflow.backends import make_device
from eigenflow.qaoa.circuit import (
    Edge,
    cost_hamiltonian,
    init_params,
    make_circuit,
    mixer_hamiltonian,
    n_wires,
)


def cut_value(bitstring: Sequence[int], edges: Sequence[Edge]) -> int:
    """Number of edges crossing the cut defined by bitstring."""
    return sum(1 for i, j in edges if bitstring[i] != bitstring[j])


def train(
    edges: Sequence[Edge],
    *,
    n_layers: int = 2,
    steps: int = 80,
    lr: float = 0.2,
    seed: int = 0,
    log_every: int = 10,
) -> tuple[jnp.ndarray, list[dict]]:
    """Minimize ⟨H_C⟩. Returns (params, history). First step compiles."""
    circuit = qml.qjit(make_circuit(edges, n_layers))
    params = init_params(n_layers, seed)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    value_and_grad = jax.value_and_grad(lambda p: circuit(p))

    history: list[dict] = []
    for t in range(steps):
        energy, grads = value_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if t % log_every == 0 or t == steps - 1:
            row = {"step": t, "energy": float(energy)}
            history.append(row)
            print(f"step {t:3d}  ⟨H_C⟩ {row['energy']:.4f}")

    return params, history


def most_likely_bitstring(
    edges: Sequence[Edge],
    params: jnp.ndarray,
    n_layers: int,
    *,
    shots: int = 1024,
) -> tuple[tuple[int, ...], dict[tuple[int, ...], int]]:
    """Sample the optimized circuit (eager, shot-based)."""
    wires = n_wires(edges)
    cost_h = cost_hamiltonian(edges)
    mixer_h = mixer_hamiltonian(wires)
    dev = make_device("lightning.qubit", wires=wires)

    @qml.set_shots(shots)
    @qml.qnode(dev)
    def sample_circuit(p):
        for w in range(wires):
            qml.Hadamard(wires=w)
        for i in range(n_layers):
            qml.qaoa.cost_layer(p[0, i], cost_h)
            qml.qaoa.mixer_layer(p[1, i], mixer_h)
        return qml.sample(wires=range(wires))

    samples = sample_circuit(params)
    counts: dict[tuple[int, ...], int] = {}
    for row in samples:
        key = tuple(int(b) for b in row)
        counts[key] = counts.get(key, 0) + 1
    best = max(counts, key=counts.get)
    return best, counts

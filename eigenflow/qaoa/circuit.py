"""QAOA for MaxCut. PennyLane qaoa layers + Catalyst on lightning.

H_C = ½ Σ_{(i,j)∈E} (Z_i Z_j − I)
H_M = Σ_i X_i

Ansatz: H⊗n, then p layers of e^{-iγ H_C} e^{-iβ H_M}. Minimize ⟨H_C⟩.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import pennylane as qml

from eigenflow.backends import make_device

Edge = tuple[int, int]
Circuit = Callable[[jnp.ndarray], jnp.ndarray]


def n_wires(edges: Sequence[Edge]) -> int:
    return 1 + max(max(i, j) for i, j in edges)


def cost_hamiltonian(edges: Sequence[Edge]):
    """MaxCut cost Hamiltonian (PennyLane convention)."""
    ops = []
    for i, j in edges:
        ops.append(0.5 * (qml.PauliZ(i) @ qml.PauliZ(j)))
        ops.append(-0.5 * qml.Identity(i))
    return qml.sum(*ops) if len(ops) > 1 else ops[0]


def mixer_hamiltonian(wires: int):
    return qml.sum(*[qml.PauliX(w) for w in range(wires)])


def init_params(n_layers: int, seed: int = 0) -> jnp.ndarray:
    """params shape (2, p): row 0 = γ, row 1 = β."""
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, (2, n_layers), minval=0.0, maxval=jnp.pi)


def make_circuit(
    edges: Sequence[Edge],
    n_layers: int,
    *,
    device: str = "lightning.qubit",
) -> Circuit:
    """Build a Lightning QNode. ``n_layers`` is a Python int (unrolled)."""
    wires = n_wires(edges)
    cost_h = cost_hamiltonian(edges)
    mixer_h = mixer_hamiltonian(wires)
    dev = make_device(device, wires=wires)

    @qml.qnode(dev, interface="jax")
    def circuit(params):
        for w in range(wires):
            qml.Hadamard(wires=w)
        for i in range(n_layers):
            qml.qaoa.cost_layer(params[0, i], cost_h)
            qml.qaoa.mixer_layer(params[1, i], mixer_h)
        return qml.expval(cost_h)

    return circuit

"""Single-qubit data re-uploading (Pérez-Salinas et al., arXiv:1907.02085).

Two layer types, same measurement:

  split       L_i = U(θ_i) U(x)              paper eq. (1)–(4)
  compressed  L_i = U(θ_i + ω_i ⊙ x)         paper §2, “A proposal for data encoding”

U is a general SU(2) rotation (qml.Rot). x is padded to 3 angles.
Binary labels live at the poles: class 0 → |0⟩, class 1 → |1⟩.
Fidelity to |0⟩/|1⟩ is (1 ± ⟨Z⟩)/2. No Hermitian observables — Catalyst-safe.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pennylane as qml

DEVICE = qml.device("lightning.qubit", wires=1)

Layer = str  # "compressed" | "split"
Circuit = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def init_params(n_layers: int, layer: Layer = "compressed", seed: int = 0) -> jnp.ndarray:
    """θ ~ U(0, 2π). For compressed layers, ω ~ U(0, 1)."""
    key = jax.random.PRNGKey(seed)
    if layer == "split":
        return jax.random.uniform(key, (n_layers, 3), minval=0.0, maxval=2 * jnp.pi)
    if layer != "compressed":
        raise ValueError(f"unknown layer {layer!r}")
    k_th, k_w = jax.random.split(key)
    theta = jax.random.uniform(k_th, (n_layers, 3), minval=0.0, maxval=2 * jnp.pi)
    omega = jax.random.uniform(k_w, (n_layers, 3), minval=0.0, maxval=1.0)
    return jnp.stack([theta, omega], axis=1)


def make_circuit(n_layers: int, layer: Layer = "compressed") -> Circuit:
    """Build a Lightning QNode. n_layers is a Python int (unrolled under qjit)."""

    if layer == "compressed":

        @qml.qnode(DEVICE)
        def circuit(params, x):
            for i in range(n_layers):
                th, w = params[i, 0], params[i, 1]
                qml.Rot(th[0] + w[0] * x[0], th[1] + w[1] * x[1], th[2] + w[2] * x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        return circuit

    if layer == "split":

        @qml.qnode(DEVICE)
        def circuit(params, x):
            for i in range(n_layers):
                qml.Rot(x[0], x[1], x[2], wires=0)
                qml.Rot(params[i, 0], params[i, 1], params[i, 2], wires=0)
            return qml.expval(qml.PauliZ(0))

        return circuit

    raise ValueError(f"unknown layer {layer!r}")


def fidelity(z: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """⟨label|ψ⟩² for y ∈ {0,1}. F_0 = (1+Z)/2, F_1 = (1−Z)/2."""
    return 0.5 * (1.0 + (1.0 - 2.0 * y) * z)


def fidelity_loss(circuit: Circuit, params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Paper χ²_f: mean (1 − F)² over the batch. Compiled with for_loop on the batch."""
    n = X.shape[0]

    def body(i, acc):
        z = circuit(params, X[i])
        err = 1.0 - fidelity(z, y[i])
        return acc + err * err

    return qml.for_loop(0, n, 1)(body)(0.0) / n


def predict(circuit: Circuit, params: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    """Class 0 if ⟨Z⟩ ≥ 0 (closer to |0⟩), else class 1."""
    n = X.shape[0]

    def body(i, acc):
        return acc.at[i].set(circuit(params, X[i]))

    z = qml.for_loop(0, n, 1)(body)(jnp.zeros((n,)))
    return (z < 0.0).astype(jnp.int32)

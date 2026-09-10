"""QKAN: KAN edges are 1-qubit DARUAN activations.

φ(x) = w_b SiLU(x) + ⟨Z⟩(U(x; θ))

Per edge, n_reps times: RY(w0·x + w1), RZ(w2). Live PennyLane QNode,
Catalyst @qjit, jax.vmap over batch / edges.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import pennylane as qml

from eigenflow.backends import make_device

Circuit = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def make_daruan_circuit(n_reps: int, *, device: str = "lightning.qubit") -> Circuit:
    """1-qubit DARUAN edge. weights shape (n_reps, 3)."""
    dev = make_device(device, wires=1)

    @qml.qnode(dev, interface="jax")
    def circuit(x, weights):
        for i in range(n_reps):
            w = weights[i]
            qml.RY(w[0] * x + w[1], wires=0)
            qml.RZ(w[2], wires=0)
        return qml.expval(qml.PauliZ(0))

    return circuit


def daruan_expval(circuit: Circuit, x: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return circuit(x, weights)


class QKAN:
    """QVAF-edge KAN. ``params = model.init(key); y = model.apply(params, x)``."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        n_reps: int = 2,
        *,
        device: str = "lightning.qubit",
        qjit: bool = True,
        squeeze: bool = True,
    ):
        self.layer_sizes = list(layer_sizes)
        self.n_reps = n_reps
        self.device = device
        self.qjit = qjit
        self.squeeze = squeeze
        edge = make_daruan_circuit(n_reps, device=device)
        self._edge = qml.qjit(edge) if qjit else edge

    def init(self, key: jax.Array) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
        keys = jax.random.split(key, len(self.layer_sizes) - 1)
        params = []
        for i, k in enumerate(keys):
            in_f, out_f = self.layer_sizes[i], self.layer_sizes[i + 1]
            k1, k2 = jax.random.split(k)
            limit = jnp.sqrt(6.0 / (in_f + out_f))
            w_base = jax.random.uniform(k1, (in_f, out_f), minval=-limit, maxval=limit)
            w_q = jax.random.normal(k2, (in_f, out_f, self.n_reps, 3)) * 0.1
            params.append((w_base, w_q))
        return params

    def _layer(self, x: jnp.ndarray, params: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        w_base, w_q = params
        single = x.ndim == 1
        xb = x if not single else x[None, :]
        base = jax.nn.silu(xb) @ w_base

        # q[b, i, o] = edge(x[b, i], w_q[i, o])
        edge = self._edge
        q = jax.vmap(
            lambda x_row: jax.vmap(
                lambda xi, w_in: jax.vmap(lambda w: edge(xi, w))(w_in),
                in_axes=(0, 0),
            )(x_row, w_q),
            in_axes=0,
        )(xb)
        out = base + jnp.sum(q, axis=1)
        return out[0] if single else out

    def apply(
        self,
        params: list[tuple[jnp.ndarray, jnp.ndarray]],
        X: jnp.ndarray,
        *,
        squeeze: bool | None = None,
    ) -> jnp.ndarray:
        h = X
        for layer_params in params:
            h = self._layer(h, layer_params)
        sq = self.squeeze if squeeze is None else squeeze
        if sq and h.ndim > 0 and h.shape[-1] == 1:
            return jnp.squeeze(h, axis=-1)
        return h

    def __call__(self, params, X, *, squeeze: bool | None = None):
        return self.apply(params, X, squeeze=squeeze)

"""QuIRK edges: RY(x), RZ(θ0), RX(θ1). Rescale to [0, π] between layers.

Live 1-qubit PennyLane QNode + Catalyst. No SiLU residual (unlike QKAN).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import pennylane as qml

from eigenflow.backends import make_device

Circuit = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def make_quirk_circuit(n_reps: int, *, device: str = "lightning.qubit") -> Circuit:
    """weights shape (n_reps, 2)."""
    dev = make_device(device, wires=1)

    @qml.qnode(dev, interface="jax")
    def circuit(x, weights):
        for i in range(n_reps):
            w = weights[i]
            qml.RY(x, wires=0)
            qml.RZ(w[0], wires=0)
            qml.RX(w[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    return circuit


def _rescale_to_pi(h: jnp.ndarray) -> jnp.ndarray:
    h_min = jnp.min(h, axis=-1, keepdims=True)
    h_max = jnp.max(h, axis=-1, keepdims=True)
    return (h - h_min) / (h_max - h_min + 1e-8) * jnp.pi


class QuIRK:
    """``params = model.init(key); y = model.apply(params, x)``."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        n_reps: int = 2,
        *,
        device: str = "lightning.qubit",
        qjit: bool = True,
        use_dense_head: bool = True,
        squeeze: bool = True,
    ):
        self.layer_sizes = list(layer_sizes)
        self.n_reps = n_reps
        self.device = device
        self.qjit = qjit
        self.use_dense_head = use_dense_head
        self.squeeze = squeeze
        edge = make_quirk_circuit(n_reps, device=device)
        self._edge = qml.qjit(edge) if qjit else edge

    def init(self, key: jax.Array) -> dict:
        keys = jax.random.split(key, len(self.layer_sizes))
        edges = []
        for i in range(len(self.layer_sizes) - 1):
            in_f, out_f = self.layer_sizes[i], self.layer_sizes[i + 1]
            w = jax.random.normal(keys[i], (in_f, out_f, self.n_reps, 2)) * 0.1
            edges.append(w)
        params: dict = {"edges": edges}
        if self.use_dense_head:
            out_dim = self.layer_sizes[-1]
            limit = jnp.sqrt(6.0 / (out_dim + 1))
            params["w_head"] = jax.random.uniform(
                keys[-1], (out_dim, 1), minval=-limit, maxval=limit
            )
            params["b_head"] = jnp.zeros((1,))
        return params

    def _layer(self, x: jnp.ndarray, w_edges: jnp.ndarray) -> jnp.ndarray:
        single = x.ndim == 1
        xb = x if not single else x[None, :]
        edge = self._edge
        q = jax.vmap(
            lambda x_row: jax.vmap(
                lambda xi, w_in: jax.vmap(lambda w: edge(xi, w))(w_in),
                in_axes=(0, 0),
            )(x_row, w_edges),
            in_axes=0,
        )(xb)
        nodes = jnp.sum(q, axis=1)
        return nodes[0] if single else nodes

    def apply(self, params: dict, X: jnp.ndarray, *, squeeze: bool | None = None) -> jnp.ndarray:
        h = X
        edges = params["edges"]
        for i, w in enumerate(edges):
            h = self._layer(h, w)
            if i < len(edges) - 1:
                h = _rescale_to_pi(h)

        if self.use_dense_head:
            single = h.ndim == 1
            hb = h if not single else h[None, :]
            out = hb @ params["w_head"] + params["b_head"]
            if single:
                out = out[0]
        else:
            out = h

        sq = self.squeeze if squeeze is None else squeeze
        if sq and out.ndim > 0 and out.shape[-1] == 1:
            out = jnp.squeeze(out, axis=-1)
        return out

    def __call__(self, params, X, *, squeeze: bool | None = None):
        return self.apply(params, X, squeeze=squeeze)

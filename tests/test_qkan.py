"""QKAN / DARUAN checks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pennylane as qml

from eigenflow.qkan import QKAN, make_daruan_circuit
from eigenflow.qkan.train import train


def test_daruan_zero_is_plus_z():
    """Zero angles: RY(0) RZ(0) is identity → ⟨Z⟩ = 1."""
    circuit = make_daruan_circuit(2, device="lightning.qubit")
    z = circuit(jnp.array(0.0), jnp.zeros((2, 3)))
    assert jnp.isclose(z, 1.0, atol=1e-5)


def test_daruan_qjit_matches_eager():
    circuit = make_daruan_circuit(2)
    x = jnp.array(0.4)
    w = jnp.array([[0.1, 0.2, 0.3], [0.05, -0.1, 0.2]])
    eager = circuit(x, w)
    compiled = qml.qjit(circuit)(x, w)
    assert jnp.isclose(eager, compiled, atol=1e-5)


def test_qkan_shapes_and_train():
    model = QKAN([2, 3, 1], n_reps=1, qjit=True)
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (16, 2))
    y = X[:, 0] + X[:, 1]
    params, hist = train(model, X, y, steps=6, lr=0.05, seed=1, log_every=5)
    assert hist[-1]["loss"] <= hist[0]["loss"] + 1e-3
    pred = model.apply(params, X[:4])
    assert pred.shape == (4,)

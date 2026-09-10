"""QuIRK checks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pennylane as qml

from eigenflow.quirk import QuIRK, make_quirk_circuit
from eigenflow.quirk.train import train


def test_quirk_zero_x_identityish():
    """x=0, zero weights: RY(0) RZ(0) RX(0) → ⟨Z⟩ = 1."""
    circuit = make_quirk_circuit(1)
    z = circuit(jnp.array(0.0), jnp.zeros((1, 2)))
    assert jnp.isclose(z, 1.0, atol=1e-5)


def test_quirk_qjit_matches_eager():
    circuit = make_quirk_circuit(2)
    x = jnp.array(0.7)
    w = jnp.array([[0.2, -0.1], [0.05, 0.3]])
    assert jnp.isclose(circuit(x, w), qml.qjit(circuit)(x, w), atol=1e-5)


def test_quirk_shapes_and_train():
    model = QuIRK([1, 3, 1], n_reps=1, qjit=True)
    key = jax.random.PRNGKey(2)
    X = jax.random.uniform(key, (20, 1), minval=-1.0, maxval=1.0)
    y = jnp.sin(X[:, 0])
    params, hist = train(model, X, y, steps=6, lr=0.05, seed=2, log_every=5)
    assert hist[-1]["loss"] <= hist[0]["loss"] + 1e-3
    assert model.apply(params, X[:3]).shape == (3,)

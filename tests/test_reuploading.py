"""Small checks. First qjit call compiles — expect ~10–30s on a cold run."""

from __future__ import annotations

import jax.numpy as jnp
import pennylane as qml

from eigenflow.reuploading import (
    accuracy,
    circle,
    fidelity,
    fidelity_loss,
    init_params,
    make_circuit,
    pad3,
    predict,
)


def test_pad3_and_circle_shapes():
    X, y = circle(32, seed=1)
    assert X.shape == (32, 3)
    assert y.shape == (32,)
    assert jnp.all((y == 0) | (y == 1))
    assert pad3(X).shape == (32, 3)


def test_zero_circuit_stays_in_zero():
    """All-zero rotations are identity. |0⟩ → ⟨Z⟩ = 1, F_0 = 1, F_1 = 0."""
    circuit = make_circuit(n_layers=2, layer="compressed")
    params = jnp.zeros((2, 2, 3))
    z = circuit(params, jnp.zeros(3))
    assert jnp.isclose(z, 1.0, atol=1e-6)
    assert jnp.isclose(fidelity(z, 0.0), 1.0, atol=1e-6)
    assert jnp.isclose(fidelity(z, 1.0), 0.0, atol=1e-6)


def test_split_zero_also_identity():
    circuit = make_circuit(n_layers=1, layer="split")
    z = circuit(jnp.zeros((1, 3)), jnp.zeros(3))
    assert jnp.isclose(z, 1.0, atol=1e-6)


def test_loss_zero_when_all_class_zero_and_identity():
    circuit = make_circuit(n_layers=1, layer="compressed")
    params = jnp.zeros((1, 2, 3))
    X = jnp.zeros((4, 3))
    y = jnp.zeros(4)
    loss = fidelity_loss(circuit, params, X, y)
    assert jnp.isclose(loss, 0.0, atol=1e-6)


def test_predict_identity_is_class_zero():
    circuit = make_circuit(n_layers=1, layer="compressed")
    params = jnp.zeros((1, 2, 3))
    X = jnp.zeros((5, 3))
    yhat = predict(circuit, params, X)
    assert jnp.all(yhat == 0)
    assert jnp.isclose(accuracy(jnp.zeros(5), yhat), 1.0)


def test_qjit_loss_matches_eager():
    circuit = make_circuit(n_layers=2, layer="compressed")
    params = init_params(2, "compressed", seed=2)
    X, y = circle(8, seed=3)

    eager = fidelity_loss(circuit, params, X, y)

    @qml.qjit
    def compiled(p, xx, yy):
        return fidelity_loss(circuit, p, xx, yy)

    assert jnp.isclose(eager, compiled(params, X, y), atol=1e-5)


def test_init_shapes():
    assert init_params(4, "compressed").shape == (4, 2, 3)
    assert init_params(4, "split").shape == (4, 3)


def test_train_drops_loss():
    from eigenflow.reuploading import train

    X, y = circle(40, seed=4)
    params, hist = train(X, y, n_layers=2, steps=8, lr=0.4, seed=4, log_every=7)
    assert hist[-1]["loss"] < hist[0]["loss"]
    assert params.shape == (2, 2, 3)

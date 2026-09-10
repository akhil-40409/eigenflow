"""QAOA MaxCut checks."""

from __future__ import annotations

import jax.numpy as jnp
import pennylane as qml

from eigenflow.qaoa import cut_value, init_params, make_circuit, train


def test_two_node_identity_energy():
    """p=0 effectively: only Hadamards → ⟨H_C⟩ = −0.5 for one edge? 

    With p layers of zero angles, cost/mixer are identity → equal superposition.
    For one edge, ⟨Z0 Z1⟩ = 0 so ⟨H_C⟩ = −0.5.
    """
    edges = [(0, 1)]
    circuit = make_circuit(edges, n_layers=1)
    params = jnp.zeros((2, 1))
    energy = circuit(params)
    assert jnp.isclose(energy, -0.5, atol=1e-5)


def test_cut_value():
    edges = [(0, 1), (1, 2), (0, 2)]
    assert cut_value((0, 1, 0), edges) == 2
    assert cut_value((0, 0, 0), edges) == 0


def test_qjit_matches_eager():
    edges = [(0, 1), (1, 2)]
    circuit = make_circuit(edges, n_layers=1)
    params = init_params(1, seed=3)
    eager = circuit(params)
    compiled = qml.qjit(circuit)(params)
    assert jnp.isclose(eager, compiled, atol=1e-5)


def test_train_drops_energy():
    edges = [(0, 1), (0, 2), (1, 2)]
    params, hist = train(edges, n_layers=1, steps=12, lr=0.3, seed=0, log_every=11)
    assert hist[-1]["energy"] < hist[0]["energy"]
    assert params.shape == (2, 1)

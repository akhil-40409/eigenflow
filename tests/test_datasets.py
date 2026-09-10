"""Dataset smoke tests."""

from __future__ import annotations

import jax

from eigenflow.datasets import FEYNMAN_EQUATIONS, SPECIAL_FUNCTIONS, make_dataset


def test_feynman_i12_1_shapes():
    data = make_dataset("I.12.1", jax.random.PRNGKey(0), n_samples=40)
    assert data["n_features"] == 2
    assert data["x_train"].shape[1] == 2
    assert data["y_train"].ndim == 1
    assert "I.12.1" in FEYNMAN_EQUATIONS


def test_special_sinc_shapes():
    data = make_dataset("sinc", jax.random.PRNGKey(1), n_samples=40)
    assert data["n_features"] == 1
    assert "sinc" in SPECIAL_FUNCTIONS


def test_unknown_task_raises():
    try:
        make_dataset("not_a_task", jax.random.PRNGKey(0), n_samples=4)
        assert False, "expected ValueError"
    except ValueError:
        pass

"""Circle-in-a-square, the paper/PennyLane sanity task.

Radius √(2/π) so the disk has area 1 inside [-1, 1]² — classes are balanced
in expectation. Features are padded to 3 so they fit one SU(2) rotation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def pad3(X: jnp.ndarray) -> jnp.ndarray:
    """Pad 2-D points to (n, 3) with a trailing zero."""
    if X.shape[-1] == 3:
        return X
    zeros = jnp.zeros(X.shape[:-1] + (3 - X.shape[-1],), dtype=X.dtype)
    return jnp.concatenate([X, zeros], axis=-1)


def circle(n: int, seed: int = 0, radius: float | None = None) -> tuple[jnp.ndarray, jnp.ndarray]:
    """n uniform points in [-1, 1]². Label 1 inside the disk, 0 outside."""
    if radius is None:
        radius = float(jnp.sqrt(2.0 / jnp.pi))
    key = jax.random.PRNGKey(seed)
    X = jax.random.uniform(key, (n, 2), minval=-1.0, maxval=1.0)
    y = (jnp.linalg.norm(X, axis=1) < radius).astype(jnp.float32)
    return pad3(X), y


def accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((y_true.astype(jnp.int32) == y_pred.astype(jnp.int32)).astype(jnp.float32))

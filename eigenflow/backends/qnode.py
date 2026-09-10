"""PennyLane device + optional Catalyst @qjit.

Lightning only for the compiled path. default.qubit is fine for debugging.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pennylane as qml

CATALYST_DEVICES = frozenset(
    {
        "lightning.qubit",
        "lightning.kokkos",
        "lightning.gpu",
        "lightning.amdgpu",
        "null.qubit",
    }
)


def make_device(name: str = "lightning.qubit", wires: int = 1, **kwargs: Any):
    return qml.device(name, wires=wires, **kwargs)


def require_catalyst_device(device_name: str) -> None:
    base = device_name.split(",")[0].strip()
    if base not in CATALYST_DEVICES and not base.startswith("lightning."):
        raise ValueError(
            f"qjit needs a Catalyst device (e.g. lightning.qubit), got {device_name!r}"
        )


def maybe_qjit(fn: Callable, *, qjit: bool = True) -> Callable:
    if not qjit:
        return fn
    return qml.qjit(fn)


def make_qnode(
    circuit_fn: Callable,
    *,
    wires: int = 1,
    device: str = "lightning.qubit",
    qjit: bool = True,
    interface: str = "jax",
    diff_method: str | None = None,
    device_kwargs: dict | None = None,
) -> Callable:
    """QNode on ``device``, optionally Catalyst-compiled."""
    if qjit:
        require_catalyst_device(device)
    dev = make_device(device, wires=wires, **(device_kwargs or {}))
    kwargs: dict[str, Any] = {"interface": interface}
    if diff_method is not None:
        kwargs["diff_method"] = diff_method
    qnode = qml.QNode(circuit_fn, dev, **kwargs)
    return qml.qjit(qnode) if qjit else qnode

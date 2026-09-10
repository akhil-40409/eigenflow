"""QKAN with DARUAN / QVAF edges (Jiang et al.). Live 1-qubit PennyLane + Catalyst."""

from eigenflow.qkan.model import QKAN, daruan_expval, make_daruan_circuit
from eigenflow.qkan.train import fit_dataset, train

__all__ = ["QKAN", "daruan_expval", "fit_dataset", "make_daruan_circuit", "train"]

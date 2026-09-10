"""QuIRK: Quantum-Inspired Re-uploading KAN (Sharma et al.)."""

from eigenflow.quirk.model import QuIRK, make_quirk_circuit
from eigenflow.quirk.train import fit_dataset, train

__all__ = ["QuIRK", "fit_dataset", "make_quirk_circuit", "train"]

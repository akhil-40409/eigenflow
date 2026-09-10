# eigenflow

A lab of quantum ML paradigms I have studied.
Not a framework. Each model is one paper, one folder, one compiled circuit.

[akhil-40409/eigenflow](https://github.com/akhil-40409/eigenflow)

**Stack:** PennyLane + JAX + Catalyst `@qjit` on `lightning.qubit`.
Read [`docs/stack.md`](docs/stack.md), then the paradigm note, then the circuit file.

## Models

| paradigm | paper | note |
|---|---|---|
| data re-uploading | [arXiv:1907.02085](https://arxiv.org/abs/1907.02085) | [docs/data-reuploading.md](docs/data-reuploading.md) |
| QAOA (MaxCut) | [arXiv:1411.4028](https://arxiv.org/abs/1411.4028) | [docs/qaoa.md](docs/qaoa.md) |
| QKAN / DARUAN | [arXiv:2509.14026](https://arxiv.org/abs/2509.14026) | [docs/qkan.md](docs/qkan.md) |
| QuIRK | [arXiv:2510.08650](https://arxiv.org/abs/2510.08650) | [docs/quirk.md](docs/quirk.md) |

Shared data: AI Feynman equations + special functions (`eigenflow.datasets`).

## Setup

Python 3.11–3.13. Catalyst needs `lightning.qubit`, not `default.qubit`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,plot]"
```

macOS: Xcode CLT (`clang`). Let `pennylane-catalyst` pick the JAX pin.

## Run

```bash
python examples/train_circle.py
python examples/train_maxcut.py
python examples/train_qkan.py --task I.12.1
python examples/train_quirk.py --task sinc
pytest
```

First `@qjit` call compiles. That is the slow part.

## Layout

```
docs/                      # short notes
eigenflow/backends/        # device + qjit
eigenflow/datasets/        # Feynman + special
eigenflow/reuploading/     # QNN
eigenflow/qaoa/
eigenflow/qkan/
eigenflow/quirk/
examples/
```

#!/usr/bin/env python3
"""Train QKAN on a Feynman / special-function task."""

from __future__ import annotations

import argparse

from eigenflow.qkan.train import fit_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="I.12.1")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--samples", type=int, default=120)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--no-qjit", action="store_true")
    args = p.parse_args()

    model, params, hist = fit_dataset(
        args.task,
        hidden=(4,),
        n_reps=args.reps,
        n_samples=args.samples,
        steps=args.steps,
        qjit=not args.no_qjit,
    )
    print(f"layers {model.layer_sizes}  final mse {hist[-1]['loss']:.4f}")


if __name__ == "__main__":
    main()

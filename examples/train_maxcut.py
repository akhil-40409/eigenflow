#!/usr/bin/env python3
"""MaxCut QAOA on a small graph."""

from __future__ import annotations

import argparse

from eigenflow.qaoa import cut_value, most_likely_bitstring, train

# 5-node graph (same topology as the Spring lab notebook)
EDGES = [(0, 1), (0, 2), (0, 4), (1, 2), (2, 3), (3, 4)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.25)
    args = p.parse_args()

    params, hist = train(EDGES, n_layers=args.layers, steps=args.steps, lr=args.lr)
    bits, _ = most_likely_bitstring(EDGES, params, args.layers, shots=512)
    print(f"best sample {bits}  cut {cut_value(bits, EDGES)} / {len(EDGES)}")
    print(f"final ⟨H_C⟩ {hist[-1]['energy']:.4f}")


if __name__ == "__main__":
    main()

"""Experiment 31: exact reduction with a common score component.

Section 5.6; Supplementary Section S13.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from hevi_hitl import flood as fl
from hevi_hitl import social as so
from hevi_hitl.moments import MomentSolver
from hevi_hitl.scaled import FastSolver

RESULTS = ROOT / "results"
LEVELS_EUR = np.array([1.0, 3.0, 6.0]) * 1e6
SEVERITY_SCALE = np.array([0.2, 0.5, 1.0])


def main():
    grid = so.GRID
    cfg = [so.SocialConfig(**dict(zip(grid, v))) for v in product(*grid.values())][10]
    m = so.build_social_flood(cfg)
    n = len(m.prior)
    w_econ = fl.weights_of(fl.FloodConfig(preference_prior=cfg.preference_prior))[m.indices[:, 5], 0]
    eur = SEVERITY_SCALE[m.indices[:, 0], None] * LEVELS_EUR[None, :]          # n x 3, EUR
    B = eur / fl.SCALES[0] * w_econ[:, None]                                  # score units
    prior = np.repeat(m.prior, 3) / 3
    loss2d = np.repeat(m.loss[:, 0, :], 3, axis=0) + B.reshape(-1)[:, None]
    ext = replace(m, prior=prior, likelihood=np.repeat(m.likelihood, 3, axis=0),
                  loss=np.broadcast_to(loss2d[:, None, :], (3 * n,) + m.loss.shape[1:]), features=np.zeros(1),
                  indices=np.column_stack((np.repeat(m.indices, 3, axis=0), np.tile(np.arange(3), n))))
    base = MomentSolver(m)
    v_base = base.solve()[0]
    extended = MomentSolver(ext)
    v_ext = extended.solve()[0]
    expected_B = float(prior @ B.reshape(-1))
    reference = FastSolver(m)
    reference.solve()
    identical = all(extended.decision(h) == reference.memo[h][1] for h in reference.memo)
    by_answer = {}
    for y, p in ((1, m.likelihood[:, 0]), (0, 1 - m.likelihood[:, 0])):
        mass = np.repeat(m.prior * p, 3) / 3
        by_answer[f"depth_flag_{y}"] = dict(probability=float(mass.sum()),
                                            expected_loss_eur=float(mass @ eur.reshape(-1) / mass.sum()))
    out = dict(configurations=3 * n, classes=n, value_extended=v_ext, value_base=v_base, expected_common=expected_B,
               identity_error=float(v_ext - (expected_B + v_base)), identical_decisions=bool(identical),
               histories_checked=len(reference.memo), expected_common_eur=float(prior @ eur.reshape(-1)),
               by_answer=by_answer)
    assert abs(out["identity_error"]) < 1e-12 and identical
    (RESULTS / "common_component.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

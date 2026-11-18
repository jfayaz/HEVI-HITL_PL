"""Exact solution by answer-subset moments (Section 4 and Supplementary Section S8)."""
from __future__ import annotations

from time import perf_counter

import numpy as np


class MomentSolver:
    """Exact (or immediate-commitment) solution of a model with time-invariant scores ``loss2d``."""

    def __init__(self, model, method: str = "optimal", block: int = 64, column_block: int | None = None):
        self.m = model
        self.method = method
        self.prior = np.asarray(model.prior, dtype=float)
        self.lik = np.asarray(model.likelihood, dtype=float)
        self.loss = np.ascontiguousarray(model.loss2d if getattr(model, "loss2d", None) is not None
                                         else model.loss[:, 0, :])
        self.feasible = np.asarray(model.feasible, dtype=bool)
        self.durations = np.asarray(model.durations, dtype=int)
        self.costs = np.asarray(model.costs, dtype=float)
        self.deadline = int(model.config.deadline)
        self.Q = self.lik.shape[1]
        self.block = int(block)
        self.column_block = column_block
        self.root = (-1,) * self.Q
        self.timings = {}

    def _moments(self):
        """``G[mask]`` (|A|) and ``g[mask]`` for every query set that fits; other rows stay zero."""
        Q = self.Q
        masks = np.arange(1 << Q)
        bits = ((masks[:, None] >> (Q - 1 - np.arange(Q))) & 1).astype(bool)
        fits = bits @ self.durations <= self.deadline
        G = np.zeros((1 << Q, self.loss.shape[1]))
        g = np.zeros(1 << Q)
        todo = np.flatnonzero(fits)
        for start in range(0, len(todo), self.block):
            chunk = todo[start:start + self.block]
            V = np.repeat(self.prior[None, :], len(chunk), axis=0)
            for row, mask in enumerate(chunk):
                for q in np.flatnonzero(bits[mask]):
                    V[row] *= self.lik[:, q]
            G[chunk] = V @ self.loss
            g[chunk] = V.sum(axis=1)
        return G, g

    @staticmethod
    def _transform(M, Q):
        """Map moments indexed by query sets to values indexed by histories, axis by axis."""
        shape_tail = M.shape[1:]
        T = M.reshape((2,) * Q + shape_tail)
        for q in range(Q):
            out_shape = T.shape[:q] + (3,) + T.shape[q + 1:]
            out = np.empty(out_shape)
            a = np.take(T, 0, axis=q)
            b = np.take(T, 1, axis=q)
            idx = [slice(None)] * len(out_shape)
            idx[q] = 0
            out[tuple(idx)] = a
            idx[q] = 1
            out[tuple(idx)] = a - b
            idx[q] = 2
            out[tuple(idx)] = b
            T = out
        return T.reshape((3 ** Q,) + shape_tail)

    def solve(self):
        start = perf_counter()
        Q = self.Q
        G, g = self._moments()
        self.timings["moments"] = perf_counter() - start
        tick = perf_counter()
        n_hist = 3 ** Q
        digits = np.zeros((n_hist, Q), dtype=np.int8)
        rest = np.arange(n_hist)
        for q in range(Q - 1, -1, -1):
            digits[:, q] = rest % 3
            rest //= 3
        asked = digits > 0
        times = asked @ self.durations
        valid = times <= self.deadline
        mass = self._transform(g, Q)
        # commitment values: blockwise over portfolios, keep the first minimiser among feasible ones
        A = self.loss.shape[1]
        # portfolios per transform block: at most about 256 MB of history values at a time
        cb = (max(1, min(A, int(2 ** 25 // n_hist))) if self.column_block is None else int(self.column_block))
        stop = np.full(n_hist, np.inf)
        action = np.full(n_hist, -1, dtype=np.int64)
        t_clip = np.minimum(times, self.deadline)
        for c0 in range(0, A, cb):
            cols = slice(c0, min(A, c0 + cb))
            V = self._transform(G[:, cols], Q)
            V = np.where(self.feasible[t_clip][:, cols], V, np.inf)
            j = np.argmin(V, axis=1)
            best = V[np.arange(n_hist), j]
            better = best < stop  # strict: earlier blocks win ties, as np.argmin over all columns
            stop[better] = best[better]
            action[better] = j[better] + c0
            del V
        self.timings["transform"] = perf_counter() - tick
        tick = perf_counter()
        W = stop.copy()
        decision = np.full(n_hist, -1, dtype=np.int64)  # -1: commit, q: query q
        if self.method != "immediate":
            level = asked.sum(axis=1)
            powers = 3 ** (Q - 1 - np.arange(Q))
            for L in range(Q - 1, -1, -1):
                rows = np.flatnonzero((level == L) & valid)
                if len(rows) == 0:
                    continue
                best = W[rows].copy()
                dec = np.full(len(rows), -1, dtype=np.int64)
                for q in range(Q):
                    ok = (~asked[rows, q]) & (times[rows] + self.durations[q] <= self.deadline)
                    if not ok.any():
                        continue
                    r = rows[ok]
                    cand = mass[r] * self.costs[q] + W[r + powers[q]] + W[r + 2 * powers[q]]
                    upd = cand < best[ok] - 1e-15
                    sel = np.flatnonzero(ok)[upd]
                    best[sel] = cand[upd]
                    dec[sel] = q
                W[rows] = best
                decision[rows] = dec
        self.timings["recursion"] = perf_counter() - tick
        self.timings["total"] = perf_counter() - start
        self.W, self.decision_array, self.action, self.powers = W, decision, action, 3 ** (Q - 1 - np.arange(Q))
        self.histories = int(valid.sum())
        return float(W[0]), self.decision(self.root)

    def index(self, h) -> int:
        return int(sum((0 if y == -1 else 1 if y == 0 else 2) * p for y, p in zip(h, self.powers)))

    def decision(self, h, mass=None):
        i = self.index(h)
        q = int(self.decision_array[i])
        return ("query", q) if q >= 0 else ("act", int(self.action[i]))

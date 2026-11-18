"""Signature partitions and prior-weighted aggregation of configurations (Section 3)."""

from dataclasses import replace
from heapq import heappush, heappop
import numpy as np


def signature_hierarchy(model, class_counts):
    """Nested deterministic partitions based on contrasts and response laws."""
    n = len(model.prior)
    requested = sorted(set(min(n, int(k)) for k in class_counts))
    endpoints = model.loss[:, model.feasible]
    contrasts = endpoints - model.loss[:, 0, 0, None]
    response_scale = max(float(np.ptp(contrasts, axis=1).max()), 1e-12)
    signature = np.column_stack((contrasts, response_scale * model.likelihood))
    heap = []
    leaves = {}
    serial = 0

    def insert(members):
        """Register a leaf and queue it for splitting along its widest coordinate."""
        nonlocal serial
        uid = serial
        serial += 1
        leaves[uid] = members
        if len(members) > 1:
            ranges = np.ptp(signature[members], axis=0)
            coordinate = int(np.argmax(ranges))
            priority = float(model.prior[members].sum() * ranges[coordinate])
            heappush(heap, (-priority, uid, coordinate))

    insert(np.arange(n))
    out = {}
    for k in requested:
        while len(leaves) < k:
            _, uid, coordinate = heappop(heap)
            members = leaves.pop(uid)
            order = np.lexsort((members, signature[members, coordinate]))
            mm = members[order]
            cut = len(mm) // 2
            insert(mm[:cut])
            insert(mm[cut:])
        inverse = np.empty(n, dtype=int)
        for j, (_, members) in enumerate(sorted(leaves.items())):
            inverse[members] = j
        out[k] = inverse
    return out


def aggregate_partition(model, inverse):
    """Prior-conditional means; preserve the original public graph exactly."""
    inverse = np.asarray(inverse, dtype=int)
    nc = int(inverse.max()) + 1
    mass = np.bincount(inverse, weights=model.prior, minlength=nc)
    if inverse.shape != (len(model.prior),) or not np.all(mass > 0):
        raise ValueError("Invalid partition")

    def average(array):
        """Prior-conditional class mean of a per-state array."""
        shape = (-1,) + (1,) * (array.ndim - 1)
        out = np.zeros((nc,) + array.shape[1:])
        np.add.at(out, inverse, array * model.prior.reshape(shape))
        return out / mass.reshape(shape)

    return replace(
        model,
        prior=mass,
        likelihood=average(model.likelihood),
        loss=average(model.loss),
        features=average(model.features),
        indices=np.arange(nc)[:, None],
    )

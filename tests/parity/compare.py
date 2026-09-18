"""Compare embedding matrices against a reference.

Bit-identity is not the bar: two correct implementations using different BLAS
kernels cannot produce identical floats. What matters is whether the difference
is large enough to change retrieval.

Real prose and degenerate inputs are scored separately. A handful of pathological
strings can drag whole-corpus cosine_min to 0.84 while every real sentence agrees
to seven decimals, and reporting one number hides which of those you have.
"""

import numpy as np

COSINE_FLOOR = 0.9999
RECALL_FLOOR = 0.99


def recall_at_k(reference, candidate, k=10):
    """Fraction of each row's top-k neighbours that survive the switch.

    This is the metric a vector database actually cares about.
    """
    def neighbours(matrix):
        sim = matrix @ matrix.T
        np.fill_diagonal(sim, -np.inf)
        return np.argsort(-sim, axis=1)[:, :k]

    a, b = neighbours(reference), neighbours(candidate)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)]))


def metrics(reference, candidate, rows):
    a, b = reference[rows], candidate[rows]
    diff = np.abs(a - b)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = np.sum(a * b, axis=1) / np.where(norms == 0, 1, norms)
    return {
        "n": int(len(rows)),
        "bit_identical_rows": int(np.sum(np.all(a == b, axis=1))),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "cosine_min": float(cosine.min()) if cosine.size else 1.0,
        "cosine_mean": float(cosine.mean()) if cosine.size else 1.0,
    }, cosine


def verdict(m):
    if m["bit_identical_rows"] == m["n"]:
        return "identical"
    if m["cosine_min"] >= 1 - 1e-6 and m["max_abs_diff"] < 1e-4:
        return "numerically-equivalent"
    if m["cosine_min"] >= COSINE_FLOOR and m.get("recall_at_10", 0) >= RECALL_FLOOR:
        return "retrieval-equivalent"
    if m.get("recall_at_10", 0) >= 0.90:
        return "close-but-drifting"
    return "DIVERGENT"


def compare(reference, candidate, groups):
    everything = np.arange(len(groups))
    out, cosine = metrics(reference, candidate, everything)
    out["dim"] = int(reference.shape[1])
    out["recall_at_10"] = recall_at_k(reference, candidate)
    out["verdict"] = verdict(out)

    for group in sorted(set(groups)):
        rows = np.array([i for i, g in enumerate(groups) if g == group])
        if rows.size == 0:
            continue
        sub, _ = metrics(reference, candidate, rows)
        out[f"cosine_min_{group}"] = sub["cosine_min"]
        out[f"max_abs_diff_{group}"] = sub["max_abs_diff"]
        if group == "real":
            # Real prose carries the verdict; edge cases are reported beside it
            # rather than being allowed to dominate.
            sub["recall_at_10"] = recall_at_k(reference[rows], candidate[rows])
            out["verdict_real"] = verdict(sub)
            out["recall_at_10_real"] = sub["recall_at_10"]

    worst = np.argsort(cosine)[:5]
    out["worst_rows"] = [{"row": int(i), "cosine": float(cosine[i])} for i in worst]
    return out

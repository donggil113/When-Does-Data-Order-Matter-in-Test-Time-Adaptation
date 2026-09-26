"""Minimal adaptation-subspace API and a separate cost ledger for source / meta-training.

An ``AdaptationSubspace`` restricts test-time updates to span(P): theta = theta0 + P z, and plain SGD on
theta becomes projected SGD, theta <- theta - eta * P P^T g.

Fitters (all consume *source/meta* data only; the runner checks ID disjointness from test splits):
  * ``full_space``            -- identity (baseline);
  * ``random_subspace``       -- seeded random k-dim subspace (dimension-matched control);
  * ``gradient_pca_subspace`` -- top-k directions of the uncentred second moment of adaptation
                                 gradients (a common low-dimensional-adaptation heuristic);
  * ``order_aware_subspace``  -- candidate: greedy choice of k directions from a pool, trading the
                                 first-order supervised usefulness of the adaptation step against
                                 the local second-order *output-level* order-difference term
                                 J P P^T (H_b P P^T g_a - H_a P P^T g_b), estimated on meta batch pairs.

The candidate is a minimal heuristic built directly on the two-step expansion in ``quadratic.py``; it
is not claimed to be optimal, and whether it helps beyond small-step / norm-matched controls is the
open empirical question (see docs/research_question.md).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from . import linalg as la
from .linalg import Matrix, Vector
from .replay import sha256_json


@dataclass
class CostRecord:
    phase: str  # e.g. "source_training", "meta_training:order_aware", "test_time:<method>/<order>"
    split: str
    n_samples: int
    sample_ids_sha256: str
    uses_labels: bool
    gradient_evals: int = 0
    hvp_evals: int = 0
    jvp_evals: int = 0
    wall_seconds: float = 0.0
    notes: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class CostLedger:
    """Keeps source/meta-training cost apart from test-time cost."""

    records: List[CostRecord] = field(default_factory=list)

    def add(self, rec: CostRecord) -> None:
        self.records.append(rec)

    def by_prefix(self, prefix: str) -> List[CostRecord]:
        return [r for r in self.records if r.phase.startswith(prefix)]

    def totals(self, prefix: str) -> dict:
        rs = self.by_prefix(prefix)
        return {"n_records": len(rs),
                "gradient_evals": sum(r.gradient_evals for r in rs),
                "hvp_evals": sum(r.hvp_evals for r in rs),
                "jvp_evals": sum(r.jvp_evals for r in rs),
                "wall_seconds": sum(r.wall_seconds for r in rs),
                "any_labels": any(r.uses_labels for r in rs)}

    def as_dict(self) -> dict:
        return {"records": [r.as_dict() for r in self.records],
                "totals": {p: self.totals(p) for p in ("source_training", "meta_training", "tuning",
                                                       "calibration", "test_time")}}


@dataclass
class AdaptationSubspace:
    name: str
    basis: Matrix  # k orthonormal rows of length ambient_dim (the columns of P)
    ambient_dim: int
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for i, bi in enumerate(self.basis):
            if len(bi) != self.ambient_dim:
                raise ValueError("basis vector has wrong length")
            for j, bj in enumerate(self.basis[: i + 1]):
                target = 1.0 if i == j else 0.0
                if abs(la.dot(bi, bj) - target) > 1e-8:
                    raise ValueError("basis must be orthonormal")

    @property
    def dim(self) -> int:
        return len(self.basis)

    def coords(self, v: Sequence[float]) -> Vector:
        """P^T v."""
        return la.matvec(self.basis, v)

    def lift(self, z: Sequence[float]) -> Vector:
        """P z."""
        return la.rmatvec(self.basis, z) if self.basis else la.zeros(self.ambient_dim)

    def project(self, v: Sequence[float]) -> Vector:
        """P P^T v."""
        return self.lift(self.coords(v))

    def fingerprint(self) -> str:
        return sha256_json({"name": self.name, "basis": [[round(x, 12) for x in b] for b in self.basis]})


def full_space(ambient_dim: int) -> AdaptationSubspace:
    return AdaptationSubspace("full", la.eye(ambient_dim), ambient_dim, {"fitter": "full_space"})


def random_subspace(ambient_dim: int, k: int, seed: int) -> AdaptationSubspace:
    rng = random.Random(seed)
    basis = la.orthonormalize(la.random_matrix(rng, k, ambient_dim))
    if len(basis) != k:
        raise RuntimeError("random basis degenerate")
    return AdaptationSubspace(f"random_k{k}_s{seed}", basis, ambient_dim,
                              {"fitter": "random_subspace", "k": k, "seed": seed})


def top_eigenvectors(sym: Matrix, k: int, iters: int = 500, seed: int = 0) -> List[Vector]:
    """Power iteration with deflation for a small symmetric PSD matrix."""
    n = len(sym)
    rng = random.Random(seed)
    vecs: List[Vector] = []
    mat = [list(r) for r in sym]
    for _ in range(min(k, n)):
        v = la.orthonormalize([la.random_vector(rng, n)])[0]
        for _ in range(iters):
            w = la.matvec(mat, v)
            for u in vecs:  # keep orthogonal to found vectors
                w = la.axpy(-la.dot(w, u), u, w)
            nw = la.norm(w)
            if nw < 1e-14:
                break
            v = la.scale(w, 1.0 / nw)
        lam = la.dot(v, la.matvec(mat, v))
        vecs.append(v)
        mat = la.mat_add(mat, la.mat_scale(la.outer(v, v), -lam))
    return la.orthonormalize(vecs)


def complete_basis(partial: Sequence[Vector], ambient_dim: int) -> Matrix:
    """Extend orthonormal vectors to a full orthonormal basis using coordinate axes."""
    return la.orthonormalize(list(partial) + la.eye(ambient_dim))


def gradient_pca_subspace(grads: Sequence[Vector], k: int, seed: int = 0) -> AdaptationSubspace:
    n = len(grads[0])
    second = [[0.0] * n for _ in range(n)]
    for g in grads:
        second = la.mat_add(second, la.outer(g, g))
    second = la.mat_scale(second, 1.0 / len(grads))
    basis = top_eigenvectors(second, k, seed=seed)
    return AdaptationSubspace(f"gradpca_k{k}", basis[:k], n,
                              {"fitter": "gradient_pca_subspace", "k": k, "n_grads": len(grads)})


@dataclass
class MetaBatchStats:
    """Quantities at theta0 for one meta batch, used by ``order_aware_subspace``."""

    g_adapt: Vector  # gradient of the unsupervised adaptation objective (e.g. entropy)
    g_sup: Vector  # gradient of the supervised loss (needs source/meta labels)
    h_pool: List[Vector]  # H_adapt p_j for every pool direction p_j


def order_aware_subspace(pool: Matrix, batches: Sequence[MetaBatchStats], output_jvp_pool: Sequence[Vector],
                         k: int, lr: float, lam: float, min_improvement: float = 1e-15,
                         normalized: bool = False) -> AdaptationSubspace:
    """Greedy subset (at most k directions) of the orthonormal ``pool`` maximising

        score(S) = lr * gain(S) - lam * lr^2 * sqrt(order(S))

    gain(S)  = mean_a < P_S g_sup_a, P_S g_adapt_a >            (first-order supervised-loss decrease
                                                               per unit step of adaptation SGD)
    order(S) = mean_{a != b} || J P_S (H_b P_S g_a - H_a P_S g_b) ||^2
               with P_S the projector onto S, H the adaptation-objective Hessian and J the output
               Jacobian at theta0 (``output_jvp_pool[j] = J p_j``, concatenated outputs).

    With ``normalized=True`` the score is scale-free,

        score(S) = gain(S) / G_ref - lam * sqrt(order(S) / order(pool)),

    where G_ref = sum_j max(0, gain({p_j})) is the largest gain any subset of the pool can reach (gain is
    additive over orthonormal pool directions; the whole pool's gain can be negative). The first term is
    the fraction of attainable first-order usefulness retained, the second lam times the retained fraction
    of the (root) order term of the whole pool; ``lr`` drops out. If no direction has positive gain the
    result is empty; if the pool has no order term the penalty is zero.

    A direction is added only if it raises the score by more than ``min_improvement`` over the current
    set (the empty set scores 0), so the result can have fewer than k directions -- possibly none, which
    is equivalent to not adapting and is reported as such rather than padded with useless directions.
    """
    m = len(pool)
    if not batches or len(batches) < 2:
        raise ValueError("need at least two meta batches for pairwise order terms")
    coef_a = [[la.dot(p, b.g_adapt) for p in pool] for b in batches]  # p_j^T g_adapt
    coef_s = [[la.dot(p, b.g_sup) for p in pool] for b in batches]
    # pool-coordinates of H_b p_j restricted to the pool: M[b][i][j] = p_i^T H_b p_j
    hcoef = [[[la.dot(pool[i], b.h_pool[j]) for j in range(m)] for i in range(m)] for b in batches]

    def score(sel: Sequence[int]) -> Dict[str, float]:
        gain = sum(sum(coef_s[a][j] * coef_a[a][j] for j in sel) for a in range(len(batches))) / len(batches)
        tot, npairs = 0.0, 0
        for a in range(len(batches)):
            for b in range(len(batches)):
                if a == b:
                    continue
                # z-coordinates (within S) of P_S H_b P_S g_a - P_S H_a P_S g_b
                diff = []
                for i in sel:
                    hb_ga = sum(hcoef[b][i][j] * coef_a[a][j] for j in sel)
                    ha_gb = sum(hcoef[a][i][j] * coef_a[b][j] for j in sel)
                    diff.append(hb_ga - ha_gb)
                out = None
                for coeff, i in zip(diff, sel):
                    term = la.scale(output_jvp_pool[i], coeff)
                    out = term if out is None else la.add(out, term)
                tot += la.dot(out, out) if out is not None else 0.0
                npairs += 1
        order = tot / npairs
        return {"gain": gain, "order": order, "score": lr * gain - lam * lr * lr * math.sqrt(order)}

    ref = score(list(range(m)))
    g_ref = sum(max(0.0, score([j])["gain"]) for j in range(m))
    if normalized:
        raw = score
        o_ref = ref["order"]

        def score(sel: Sequence[int]) -> Dict[str, float]:  # noqa: F811 - deliberate rebinding
            r = raw(sel)
            gain_frac = r["gain"] / g_ref if g_ref > 0 else (0.0 if r["gain"] == 0 else -math.inf)
            pen = math.sqrt(r["order"] / o_ref) if o_ref > 0 else 0.0
            return {**r, "score": gain_frac - lam * pen}

    chosen: List[int] = []
    trace = []
    current = 0.0
    stopped = "reached_k"
    for _ in range(min(k, m)):
        best, best_s = None, None
        for j in range(m):
            if j in chosen:
                continue
            s = score(chosen + [j])
            if best_s is None or s["score"] > best_s["score"]:
                best, best_s = j, s
        if best_s["score"] <= current + min_improvement:
            stopped = "no_improving_direction"
            trace.append({"rejected": best, **best_s})
            break
        chosen.append(best)
        current = best_s["score"]
        trace.append({"picked": best, **best_s})
    basis = [list(pool[j]) for j in chosen]
    return AdaptationSubspace(f"orderaware_k{len(chosen)}of{k}_lam{lam:g}", basis, len(pool[0]),
                              {"fitter": "order_aware_subspace", "k_max": k, "k": len(chosen), "lam": lam,
                               "lr": lr, "normalized": normalized, "pool_gain": ref["gain"],
                               "attainable_gain": g_ref, "pool_order": ref["order"],
                               "chosen_pool_indices": chosen, "greedy_trace": trace,
                               "stopped": stopped, "n_meta_batches": len(batches)})



__all__ = ["CostRecord", "CostLedger", "AdaptationSubspace", "full_space", "random_subspace",
           "gradient_pca_subspace", "order_aware_subspace", "MetaBatchStats", "complete_basis",
           "top_eigenvectors"]

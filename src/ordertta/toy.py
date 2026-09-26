"""Synthetic shifted-Gaussian data and a linear softmax model with Tent-like affine adaptation.

This is a CPU stand-in used to exercise the pipeline, not evidence about real networks.

Model: logits = W z + b with z = gamma * n(x) + beta, where n(.) standardises features using either
fixed source statistics (``norm="source"``) or the current batch statistics (``norm="batch"``,
Tent-style). (W, b) are trained on labelled source data and then frozen; test-time adaptation
updates theta = [gamma (d), beta (d)] -- the analogue of Tent's BN affine parameters.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import linalg as la
from .linalg import Matrix, Vector
from .replay import Sample, sha256_json

EPS = 1e-5


# ----------------------------------------------------------------------------------------- data

@dataclass(frozen=True)
class DomainShift:
    name: str
    scale: Tuple[float, ...]
    shift: Tuple[float, ...]
    noise: float = 0.0

    def apply(self, rng: random.Random, x: Sequence[float]) -> Tuple[float, ...]:
        return tuple(s * xi + t + (rng.gauss(0.0, self.noise) if self.noise else 0.0)
                     for xi, s, t in zip(x, self.scale, self.shift))


def make_class_means(rng: random.Random, n_classes: int, dim: int, sep: float) -> Matrix:
    return [la.random_vector(rng, dim, sep) for _ in range(n_classes)]


def make_domains(rng: random.Random, prefix: str, n: int, dim: int, scale_range: Tuple[float, float],
                 shift_sd: float, noise: float) -> List[DomainShift]:
    out = []
    for k in range(n):
        scale = tuple(rng.uniform(*scale_range) for _ in range(dim))
        shift = tuple(rng.gauss(0.0, shift_sd) for _ in range(dim))
        out.append(DomainShift(f"{prefix}{k}", scale, shift, noise))
    return out


def sample_labelled(rng: random.Random, split: str, domain: Optional[DomainShift], n_per_class: int,
                    means: Matrix, sigma: float) -> List[Tuple[Sample, int]]:
    out = []
    dname = domain.name if domain else "source"
    idx = 0
    for y, mu in enumerate(means):
        for _ in range(n_per_class):
            x = tuple(m + rng.gauss(0.0, sigma) for m in mu)
            if domain is not None:
                x = domain.apply(rng, x)
            out.append((Sample(f"{split}-{dname}-{idx:05d}", x, dname), y))
            idx += 1
    return out


@dataclass
class SyntheticWorld:
    """All splits of one synthetic experiment, generated from a single data seed."""

    splits: Dict[str, List[Tuple[Sample, int]]]
    domains: Dict[str, List[DomainShift]]
    means: Matrix
    config: dict

    def samples(self, split: str) -> List[Sample]:
        return [s for s, _ in self.splits[split]]

    def labels(self, split: str) -> Dict[str, int]:
        return {s.id: y for s, y in self.splits[split]}

    def ids(self, split: str) -> List[str]:
        return [s.id for s, _ in self.splits[split]]

    def fingerprint(self) -> str:
        return sha256_json({k: [[s.id, [round(v, 12) for v in s.x], s.domain, y] for s, y in rows]
                            for k, rows in sorted(self.splits.items())})


def build_world(cfg: dict) -> SyntheticWorld:
    """cfg keys: seed, dim, n_classes, class_sep, sigma, n_test_domains, n_val_domains, n_meta_domains,
    scale_range, shift_sd, domain_noise, n_per_class: {split: int}."""
    rng = random.Random(cfg["seed"])
    dim, n_classes = cfg["dim"], cfg["n_classes"]
    means = make_class_means(rng, n_classes, dim, cfg["class_sep"])
    sr = tuple(cfg["scale_range"])
    test_domains = make_domains(rng, "T", cfg["n_test_domains"], dim, sr, cfg["shift_sd"], cfg["domain_noise"])
    val_domains = make_domains(rng, "V", cfg["n_val_domains"], dim, sr, cfg["shift_sd"], cfg["domain_noise"])
    meta_domains = make_domains(rng, "M", cfg["n_meta_domains"], dim, sr, cfg["shift_sd"], cfg["domain_noise"])
    npc = cfg["n_per_class"]
    sigma = cfg["sigma"]
    splits: Dict[str, List[Tuple[Sample, int]]] = {}
    splits["source_train"] = sample_labelled(rng, "src", None, npc["source_train"], means, sigma)
    splits["meta_train"] = [r for d in meta_domains
                            for r in sample_labelled(rng, "meta", d, npc["meta_train"], means, sigma)]
    splits["development"] = [r for d in val_domains
                             for r in sample_labelled(rng, "dev", d, npc["development"], means, sigma)]
    splits["dev_holdout"] = [r for d in val_domains
                             for r in sample_labelled(rng, "devho", d, npc["dev_holdout"], means, sigma)]
    splits["calibration"] = [r for d in val_domains
                             for r in sample_labelled(rng, "cal", d, npc["calibration"], means, sigma)]
    splits["test_stream"] = [r for d in test_domains
                             for r in sample_labelled(rng, "test", d, npc["test_stream"], means, sigma)]
    splits["test_holdout"] = [r for d in test_domains
                              for r in sample_labelled(rng, "hold", d, npc["test_holdout"], means, sigma)]
    return SyntheticWorld(splits, {"test": test_domains, "val": val_domains, "meta": meta_domains},
                          means, dict(cfg))


# ---------------------------------------------------------------------------------------- model

def softmax(s: Sequence[float]) -> Vector:
    m = max(s)
    e = [math.exp(v - m) for v in s]
    z = math.fsum(e)
    return [v / z for v in e]


def entropy(p: Sequence[float]) -> float:
    return -math.fsum(pi * math.log(pi) for pi in p if pi > 0.0)


@dataclass
class LinearTTAModel:
    W: Matrix  # C x d, frozen after source training
    b: Vector  # C
    mean: Vector  # source feature mean
    std: Vector  # source feature std
    norm: str = "source"  # "source" | "batch"
    grad_evals: int = field(default=0, compare=False)

    @property
    def dim(self) -> int:
        return len(self.mean)

    @property
    def theta_dim(self) -> int:
        return 2 * self.dim

    def initial_theta(self) -> Vector:
        return [1.0] * self.dim + [0.0] * self.dim

    def fingerprint(self) -> str:
        r = lambda v: [round(x, 12) for x in v]  # noqa: E731
        return sha256_json({"W": [r(row) for row in self.W], "b": r(self.b), "mean": r(self.mean),
                            "std": r(self.std), "norm": self.norm})

    # -- forward -------------------------------------------------------------------------------
    def normalize(self, xs: Sequence[Sequence[float]]) -> List[Vector]:
        if self.norm == "source":
            mu, sd = self.mean, self.std
        elif self.norm == "batch":
            n = len(xs)
            mu = [math.fsum(x[i] for x in xs) / n for i in range(self.dim)]
            sd = [math.sqrt(math.fsum((x[i] - mu[i]) ** 2 for x in xs) / n + EPS) for i in range(self.dim)]
        else:
            raise ValueError(f"unknown norm {self.norm!r}")
        return [[(x[i] - mu[i]) / sd[i] for i in range(self.dim)] for x in xs]

    def _z(self, xhat: Sequence[float], theta: Sequence[float]) -> Vector:
        d = self.dim
        return [theta[i] * xhat[i] + theta[d + i] for i in range(d)]

    def logits(self, xs: Sequence[Sequence[float]], theta: Sequence[float]) -> List[Vector]:
        return [la.add(la.matvec(self.W, self._z(xh, theta)), self.b) for xh in self.normalize(xs)]

    def probs(self, xs: Sequence[Sequence[float]], theta: Sequence[float]) -> List[Vector]:
        return [softmax(s) for s in self.logits(xs, theta)]

    def predict(self, xs: Sequence[Sequence[float]], theta: Sequence[float]) -> List[int]:
        return [max(range(len(p)), key=p.__getitem__) for p in self.probs(xs, theta)]

    # -- objectives ----------------------------------------------------------------------------
    def _backprop(self, xhat: Sequence[float], ds: Sequence[float]) -> Vector:
        dz = la.rmatvec(self.W, ds)
        return [dz[i] * xhat[i] for i in range(self.dim)] + dz

    def entropy_and_grad(self, xs: Sequence[Sequence[float]], theta: Sequence[float]) -> Tuple[float, Vector]:
        """Mean prediction entropy over the batch and its gradient w.r.t. theta (Tent objective)."""
        self.grad_evals += 1
        xh_all = self.normalize(xs)
        total, g = 0.0, la.zeros(self.theta_dim)
        for xh in xh_all:
            p = softmax(la.add(la.matvec(self.W, self._z(xh, theta)), self.b))
            h = entropy(p)
            ds = [-pc * ((math.log(pc) if pc > 0 else 0.0) + h) for pc in p]
            total += h
            g = la.add(g, self._backprop(xh, ds))
        n = len(xh_all)
        return total / n, la.scale(g, 1.0 / n)

    def ce_and_grad(self, xs: Sequence[Sequence[float]], ys: Sequence[int],
                    theta: Sequence[float]) -> Tuple[float, Vector]:
        """Mean cross-entropy and its gradient w.r.t. theta. Needs labels: source/meta data only."""
        self.grad_evals += 1
        xh_all = self.normalize(xs)
        total, g = 0.0, la.zeros(self.theta_dim)
        for xh, y in zip(xh_all, ys):
            p = softmax(la.add(la.matvec(self.W, self._z(xh, theta)), self.b))
            total += -math.log(max(p[y], 1e-300))
            ds = [pc - (1.0 if c == y else 0.0) for c, pc in enumerate(p)]
            g = la.add(g, self._backprop(xh, ds))
        n = len(xh_all)
        return total / n, la.scale(g, 1.0 / n)

    def entropy_hvp(self, xs: Sequence[Sequence[float]], theta: Sequence[float], v: Sequence[float],
                    eps: float = 1e-4) -> Vector:
        """Hessian-vector product of the mean entropy by central differences of the analytic gradient."""
        nv = la.norm(v)
        if nv == 0.0:
            return la.zeros(self.theta_dim)
        h = eps / nv
        _, gp = self.entropy_and_grad(xs, la.axpy(h, v, theta))
        _, gm = self.entropy_and_grad(xs, la.axpy(-h, v, theta))
        return la.scale(la.sub(gp, gm), 1.0 / (2.0 * h))


def train_source_model(rows: Sequence[Tuple[Sample, int]], n_classes: int, epochs: int, lr: float,
                       norm: str, seed: int) -> Tuple[LinearTTAModel, dict]:
    """Full-batch gradient descent on cross-entropy for (W, b); returns model and a cost record."""
    rng = random.Random(seed)
    xs = [s.x for s, _ in rows]
    ys = [y for _, y in rows]
    d = len(xs[0])
    n = len(xs)
    mean = [math.fsum(x[i] for x in xs) / n for i in range(d)]
    std = [math.sqrt(math.fsum((x[i] - mean[i]) ** 2 for x in xs) / n + EPS) for i in range(d)]
    xh = [[(x[i] - mean[i]) / std[i] for i in range(d)] for x in xs]
    W = la.random_matrix(rng, n_classes, d, 0.01)
    b = la.zeros(n_classes)
    for _ in range(epochs):
        gW = [[0.0] * d for _ in range(n_classes)]
        gb = la.zeros(n_classes)
        for x, y in zip(xh, ys):
            p = softmax(la.add(la.matvec(W, x), b))
            for c in range(n_classes):
                r = p[c] - (1.0 if c == y else 0.0)
                gb[c] += r
                for i in range(d):
                    gW[c][i] += r * x[i]
        W = [[W[c][i] - lr * gW[c][i] / n for i in range(d)] for c in range(n_classes)]
        b = [b[c] - lr * gb[c] / n for c in range(n_classes)]
    model = LinearTTAModel(W, b, mean, std, norm)
    src_model = LinearTTAModel(W, b, mean, std, "source")
    preds = src_model.predict(xs, src_model.initial_theta())
    src_err = sum(int(p != y) for p, y in zip(preds, ys)) / n
    cost = {"split": "source_train", "n_samples": n, "uses_labels": True, "epochs": epochs,
            "full_batch_gradient_evals": epochs, "train_error_source_norm": src_err}
    return model, cost


__all__ = ["DomainShift", "SyntheticWorld", "build_world", "LinearTTAModel", "train_source_model",
           "softmax", "entropy"]

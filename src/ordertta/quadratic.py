"""Order dependence of plain SGD on smooth losses: exact differences vs. the local second-order term.

Scope (deliberately narrow):
  * plain SGD, constant step size, no momentum, no Adam state, no weight decay;
  * a small number of steps from a common starting point;
  * smooth losses whose Hessian is available (exactly, or analytically).

Known expansion (Taylor; see docs/prior_art.md, e.g. Reptile/Nichol et al. 2018 and
Smith et al. 2021): for losses L_1..L_K applied in the order pi,

    theta_K = theta - eta * sum_k g_k + eta^2 * sum_{i<j} H_{pi_j} g_{pi_i} + O(eta^3),

with all g, H evaluated at the common starting point theta. For two steps,

    theta_ab - theta_ba = eta^2 (H_b g_a - H_a g_b) + O(eta^3),

and for quadratics the two-step identity is exact (gradients are affine). Nothing in
this module says anything about adaptive optimisers or long-horizon stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Protocol, Sequence

from . import linalg as la
from .linalg import Matrix, Vector

SUPPORTED_OPTIMIZERS = ("sgd",)


class SmoothLoss(Protocol):
    def value(self, theta: Sequence[float]) -> float: ...

    def grad(self, theta: Sequence[float]) -> Vector: ...

    def hess(self, theta: Sequence[float]) -> Matrix: ...


@dataclass
class Quadratic:
    """L(theta) = 0.5 (theta - c)^T A (theta - c), A symmetric."""

    A: Matrix
    c: Vector

    def value(self, theta: Sequence[float]) -> float:
        r = la.sub(theta, self.c)
        return 0.5 * la.dot(r, la.matvec(self.A, r))

    def grad(self, theta: Sequence[float]) -> Vector:
        return la.matvec(self.A, la.sub(theta, self.c))

    def hess(self, theta: Sequence[float]) -> Matrix:
        return [list(r) for r in self.A]


@dataclass
class QuarticPerturbedQuadratic:
    """L(theta) = 0.5 r^T A r + (kappa / 4) sum_i r_i^4, r = theta - c. Smooth but not quadratic."""

    A: Matrix
    c: Vector
    kappa: float

    def value(self, theta: Sequence[float]) -> float:
        r = la.sub(theta, self.c)
        return 0.5 * la.dot(r, la.matvec(self.A, r)) + 0.25 * self.kappa * sum(x ** 4 for x in r)

    def grad(self, theta: Sequence[float]) -> Vector:
        r = la.sub(theta, self.c)
        return la.add(la.matvec(self.A, r), [self.kappa * x ** 3 for x in r])

    def hess(self, theta: Sequence[float]) -> Matrix:
        r = la.sub(theta, self.c)
        h = [list(row) for row in self.A]
        for i, x in enumerate(r):
            h[i][i] += 3.0 * self.kappa * x * x
        return h


def _check_optimizer(optimizer: str) -> None:
    if optimizer not in SUPPORTED_OPTIMIZERS:
        raise NotImplementedError(
            f"optimizer={optimizer!r} is out of scope: the second-order order-difference analysis "
            "here only covers plain SGD (no momentum/Adam state)."
        )


def sgd_path(theta0: Sequence[float], losses: Sequence[SmoothLoss], lr: float,
             optimizer: str = "sgd") -> List[Vector]:
    """Apply one plain-SGD step per loss, in the given order. Returns [theta_0, ..., theta_K]."""
    _check_optimizer(optimizer)
    path = [list(theta0)]
    theta = list(theta0)
    for loss in losses:
        theta = la.axpy(-lr, loss.grad(theta), theta)
        path.append(theta)
    return path


def sgd_final(theta0: Sequence[float], losses: Sequence[SmoothLoss], lr: float,
              optimizer: str = "sgd") -> Vector:
    return sgd_path(theta0, losses, lr, optimizer)[-1]


def two_step_order_difference(theta0: Sequence[float], loss_a: SmoothLoss, loss_b: SmoothLoss,
                              lr: float) -> Vector:
    """Exact theta_ab - theta_ba (a then b, minus b then a)."""
    return la.sub(sgd_final(theta0, [loss_a, loss_b], lr), sgd_final(theta0, [loss_b, loss_a], lr))


def two_step_second_order_term(theta0: Sequence[float], loss_a: SmoothLoss, loss_b: SmoothLoss,
                               lr: float) -> Vector:
    """Local prediction eta^2 (H_b g_a - H_a g_b), all evaluated at theta0."""
    ga, gb = loss_a.grad(theta0), loss_b.grad(theta0)
    ha, hb = loss_a.hess(theta0), loss_b.hess(theta0)
    return la.scale(la.sub(la.matvec(hb, ga), la.matvec(ha, gb)), lr * lr)


def ordered_second_order_sum(theta0: Sequence[float], losses: Sequence[SmoothLoss],
                             order: Sequence[int]) -> Vector:
    """S(pi) = sum_{i<j} H_{pi_j} g_{pi_i} at theta0 (without the eta^2 factor)."""
    grads = {k: losses[k].grad(theta0) for k in set(order)}
    hess = {k: losses[k].hess(theta0) for k in set(order)}
    total = la.zeros(len(theta0))
    prefix = la.zeros(len(theta0))  # sum of g_{pi_i} for i < j
    for j, k in enumerate(order):
        if j > 0:
            total = la.add(total, la.matvec(hess[k], prefix))
        prefix = la.add(prefix, grads[k])
    return total


def multi_step_order_difference(theta0: Sequence[float], losses: Sequence[SmoothLoss],
                                order_1: Sequence[int], order_2: Sequence[int], lr: float) -> Vector:
    """Exact theta(order_1) - theta(order_2)."""
    if sorted(order_1) != sorted(order_2):
        raise ValueError("orders must be permutations of the same multiset of loss indices")
    t1 = sgd_final(theta0, [losses[k] for k in order_1], lr)
    t2 = sgd_final(theta0, [losses[k] for k in order_2], lr)
    return la.sub(t1, t2)


def multi_step_second_order_prediction(theta0: Sequence[float], losses: Sequence[SmoothLoss],
                                       order_1: Sequence[int], order_2: Sequence[int],
                                       lr: float) -> Vector:
    """eta^2 [S(order_1) - S(order_2)]; the first-order terms cancel for permutations."""
    if sorted(order_1) != sorted(order_2):
        raise ValueError("orders must be permutations of the same multiset of loss indices")
    d = la.sub(ordered_second_order_sum(theta0, losses, order_1),
               ordered_second_order_sum(theta0, losses, order_2))
    return la.scale(d, lr * lr)


def output_level_prediction(output_grad: Sequence[float], param_difference: Sequence[float]) -> float:
    """First-order output difference f(theta_1) - f(theta_2) ~= grad f(theta0)^T (theta_1 - theta_2)."""
    return la.dot(output_grad, param_difference)


def projected_losses(losses: Sequence[SmoothLoss], basis: Matrix, anchor: Sequence[float]) -> List["ProjectedLoss"]:
    """Re-express losses in subspace coordinates z, theta = anchor + P z (P rows = basis vectors)."""
    return [ProjectedLoss(l, basis, list(anchor)) for l in losses]


@dataclass
class ProjectedLoss:
    """L(anchor + P z) where ``basis`` holds the columns of P as rows (orthonormal)."""

    base: SmoothLoss
    basis: Matrix
    anchor: Vector

    def lift(self, z: Sequence[float]) -> Vector:
        return la.add(self.anchor, la.rmatvec(self.basis, z))

    def value(self, z: Sequence[float]) -> float:
        return self.base.value(self.lift(z))

    def grad(self, z: Sequence[float]) -> Vector:
        return la.matvec(self.basis, self.base.grad(self.lift(z)))

    def hess(self, z: Sequence[float]) -> Matrix:
        h = self.base.hess(self.lift(z))
        # P^T H P with P columns = basis rows
        hp = [la.matvec(h, b) for b in self.basis]  # rows: H p_j
        return [[la.dot(bi, hpj) for hpj in hp] for bi in self.basis]


def residual_order_estimate(residual_fn: Callable[[float], float], lrs: Sequence[float]) -> List[float]:
    """Empirical convergence orders log(r(eta_k)/r(eta_{k+1})) / log(eta_k/eta_{k+1})."""
    import math

    res = [residual_fn(lr) for lr in lrs]
    out = []
    for (l1, r1), (l2, r2) in zip(zip(lrs, res), zip(lrs[1:], res[1:])):
        if r1 <= 0 or r2 <= 0:
            out.append(float("inf"))
        else:
            out.append(math.log(r1 / r2) / math.log(l1 / l2))
    return out


def make_random_quadratic(rng, dim: int, min_eig: float = 0.2, max_eig: float = 2.0,
                          center_sd: float = 1.0) -> Quadratic:
    return Quadratic(la.random_spd(rng, dim, min_eig, max_eig), la.random_vector(rng, dim, center_sd))


def make_random_quartic(rng, dim: int, kappa: float = 0.5) -> QuarticPerturbedQuadratic:
    q = make_random_quadratic(rng, dim)
    return QuarticPerturbedQuadratic(q.A, q.c, kappa)



__all__ = [
    "Quadratic", "QuarticPerturbedQuadratic", "ProjectedLoss", "sgd_path", "sgd_final",
    "two_step_order_difference", "two_step_second_order_term", "ordered_second_order_sum",
    "multi_step_order_difference", "multi_step_second_order_prediction", "output_level_prediction",
    "projected_losses", "residual_order_estimate", "make_random_quadratic", "make_random_quartic",
    "SUPPORTED_OPTIMIZERS",
]

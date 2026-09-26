"""Test-time adapters (label-free) for the toy model, plus the control constructions.

Controls wired here:
  * ``NoAdapt``                         -- consumes the stream, never updates (order-invariant by construction);
  * small learning rate                 -- the baseline adapter with lr scaled by a fixed factor;
  * update-norm matched, ``per_step``    -- full-space adapter whose step t has exactly the candidate's
                                           step-t update norm on the same order (uses only the candidate's
                                           unsupervised update norms, never labels);
  * update-norm matched, ``calibrated_global`` -- full-space adapter whose lr multiplier is bisected on the
                                           *calibration* split so that its mean terminal displacement equals
                                           the candidate's there (no test-split information).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from . import linalg as la
from .linalg import Vector
from .replay import Batch
from .subspace import AdaptationSubspace
from .toy import LinearTTAModel


@dataclass
class MethodCard:
    """What a comparison arm may access and what it adapts. Filled per run and written to the summary."""

    name: str
    kind: str
    uses_test_labels: bool = False
    uses_source_labels_at_test_time: bool = False
    subspace_fit_uses_source_labels: bool = False
    history: str = "online; state carried across all steps of one order; no reset"
    extra_supervision: str = "none"
    adapted_params: str = "gamma,beta (feature-wise affine)"
    n_adapted_params: int = 0
    effective_dim: int = 0
    lr: Optional[float] = None
    control_of: Optional[str] = None
    notes: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class StepRecord:
    step: int
    n: int
    objective: float
    grad_norm: float
    update_norm: float
    displacement: float
    target_norm: Optional[float] = None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class Adapter:
    card: MethodCard

    def __init__(self, model: LinearTTAModel):
        self.model = model
        self.theta0: Vector = model.initial_theta()
        self.theta: Vector = list(self.theta0)
        self.trace: List[StepRecord] = []

    def predict_proba(self, xs: Sequence[Sequence[float]]) -> List[Vector]:
        return self.model.probs(xs, self.theta)

    def displacement(self) -> float:
        return la.norm(la.sub(self.theta, self.theta0))

    def path_length(self) -> float:
        return math.fsum(r.update_norm for r in self.trace)

    def total_grad_evals(self) -> int:
        """Gradient evaluations charged to this arm (including any reference computation)."""
        return self.model.grad_evals

    def step(self, batch: Batch) -> StepRecord:  # pragma: no cover - interface
        raise NotImplementedError


class FutureTraceAccessError(RuntimeError):
    """A norm-matched control asked for a reference step other than the current one."""


class CausalNormReference:
    """Reveals the target arm's update norm for step t only while step t is being taken.

    A fresh replica of the target adapter (same start, lr and subspace) is advanced on the same batch
    at the same time as the control, so the control can only see norms from steps <= t. Its gradient
    evaluations are charged to the control.
    """

    def __init__(self, reference: Adapter):
        self.reference = reference
        self._next = 0

    def norm_for(self, batch: Batch) -> float:
        if batch.step != self._next:
            raise FutureTraceAccessError(f"requested reference step {batch.step}, current step is {self._next}")
        rec = self.reference.step(batch)
        self._next += 1
        return rec.update_norm

    @property
    def grad_evals(self) -> int:
        return self.reference.model.grad_evals


class NoAdapt(Adapter):
    def __init__(self, model: LinearTTAModel, name: str = "no_adapt"):
        super().__init__(model)
        self.card = MethodCard(name, "no_adapt", n_adapted_params=0, effective_dim=0,
                               history="none (parameters never change)")

    def step(self, batch: Batch) -> StepRecord:
        rec = StepRecord(batch.step, len(batch.ids), float("nan"), 0.0, 0.0, 0.0)
        self.trace.append(rec)
        return rec


class EntropySGD(Adapter):
    """Tent-like entropy minimisation with plain SGD, optionally inside a subspace and/or norm-matched."""

    def __init__(self, model: LinearTTAModel, name: str, lr: float,
                 subspace: Optional[AdaptationSubspace] = None,
                 step_norms: Optional[Sequence[float]] = None,
                 control_of: Optional[str] = None, subspace_fit_uses_source_labels: bool = False,
                 norm_reference: Optional[CausalNormReference] = None):
        super().__init__(model)
        if lr < 0 or not math.isfinite(lr):
            raise ValueError("lr must be finite and non-negative")
        if step_norms is not None and norm_reference is not None:
            raise ValueError("use either a precomputed norm schedule or a causal reference, not both")
        self.lr = lr
        self.subspace = subspace
        self.step_norms = list(step_norms) if step_norms is not None else None
        self.norm_reference = norm_reference
        dim = subspace.dim if subspace is not None else model.theta_dim
        self.card = MethodCard(name, "entropy_sgd", n_adapted_params=model.theta_dim, effective_dim=dim,
                               lr=lr, control_of=control_of,
                               subspace_fit_uses_source_labels=subspace_fit_uses_source_labels,
                               notes=(f"subspace={subspace.name}" if subspace else "full space")
                               + ("; per-step update norms copied from target" if step_norms is not None else "")
                               + ("; per-step update norms from a lockstep (causal) replica of the target"
                                  if norm_reference is not None else ""))

    def total_grad_evals(self) -> int:
        extra = self.norm_reference.grad_evals if self.norm_reference is not None else 0
        return self.model.grad_evals + extra

    def step(self, batch: Batch) -> StepRecord:
        obj, g = self.model.entropy_and_grad(batch.xs, self.theta)
        u = self.subspace.project(g) if self.subspace is not None else g
        upd = la.scale(u, -self.lr)
        target = None
        if self.norm_reference is not None:
            target = self.norm_reference.norm_for(batch)
        elif self.step_norms is not None:
            if batch.step >= len(self.step_norms):
                raise IndexError("norm schedule shorter than the stream")
            target = self.step_norms[batch.step]
        if target is not None:
            n = la.norm(upd)
            upd = la.scale(upd, target / n) if n > 0 else la.zeros(len(upd))
        self.theta = la.add(self.theta, upd)
        rec = StepRecord(batch.step, len(batch.ids), obj, la.norm(g), la.norm(upd), self.displacement(), target)
        self.trace.append(rec)
        return rec


def small_lr(base_lr: float, factor: float) -> float:
    if not 0 < factor < 1:
        raise ValueError("small-lr control factor must be in (0, 1)")
    return base_lr * factor


def per_step_norm_schedule(candidate_trace: Sequence[StepRecord]) -> List[float]:
    return [r.update_norm for r in candidate_trace]


@dataclass
class CalibrationResult:
    lr: float
    achieved: float
    target: float
    iterations: List[Dict[str, float]] = field(default_factory=list)

    @property
    def relative_mismatch(self) -> float:
        if self.target > 0:
            return abs(self.achieved - self.target) / self.target
        return 0.0 if self.achieved == 0 else float("inf")  # target never moved: lr = 0 matches exactly


def calibrate_lr_to_displacement(mean_displacement_at: Callable[[float], float], target: float,
                                 lr_lo: float, lr_hi: float, iters: int = 30,
                                 rel_tol: float = 1e-3) -> CalibrationResult:
    """Log-space bisection for lr such that mean terminal displacement ~= target.

    Assumes displacement is increasing in lr over [lr_lo, lr_hi]; the achieved mismatch is reported,
    never hidden, so a non-monotone case shows up as a large ``relative_mismatch``.
    """
    if target <= 0:
        return CalibrationResult(0.0, 0.0, target, [])
    log_lo, log_hi = math.log(lr_lo), math.log(lr_hi)
    hist: List[Dict[str, float]] = []
    best = None
    for _ in range(iters):
        mid = 0.5 * (log_lo + log_hi)
        lr = math.exp(mid)
        disp = mean_displacement_at(lr)
        hist.append({"lr": lr, "mean_displacement": disp})
        if best is None or abs(disp - target) < abs(best[1] - target):
            best = (lr, disp)
        if abs(disp - target) <= rel_tol * target:
            break
        if disp < target:
            log_lo = mid
        else:
            log_hi = mid
    return CalibrationResult(best[0], best[1], target, hist)


__all__ = ["MethodCard", "StepRecord", "Adapter", "NoAdapt", "EntropySGD", "small_lr", "CausalNormReference",
           "FutureTraceAccessError",
           "per_step_norm_schedule", "CalibrationResult", "calibrate_lr_to_displacement"]

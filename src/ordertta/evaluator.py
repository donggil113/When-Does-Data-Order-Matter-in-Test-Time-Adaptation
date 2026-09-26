"""Terminal common-holdout evaluation.

After one order of the stream has been replayed *completely* (``ReplayLoader.verify_complete``), the
final parameters are scored on one holdout set that is (i) the same for every order and method and
(ii) disjoint from every adaptation / fitting / tuning split. Holdout labels are read only here.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence

from .linalg import Vector
from .replay import (IntegrityError, LabelVault, ReplayLoader, Sample, assert_disjoint, sha256_json)


class NonTerminalEvaluationError(RuntimeError):
    """Holdout evaluation requested before the stream was fully and exactly replayed."""


@dataclass
class TerminalResult:
    error: float
    per_domain_error: Dict[str, float]
    predictions: List[int]
    probs: List[Vector]
    holdout_fingerprint: str

    def as_dict(self, with_outputs: bool = False) -> dict:
        d = {"error": self.error, "per_domain_error": self.per_domain_error,
             "holdout_fingerprint": self.holdout_fingerprint}
        if with_outputs:
            d["predictions"] = self.predictions
        return d


class TerminalHoldoutEvaluator:
    READER = "terminal_evaluator"

    def __init__(self, holdout: Sequence[Sample], labels: Mapping[str, int],
                 other_splits: Mapping[str, Sequence[str]], eval_batch_size: int):
        ids = [s.id for s in holdout]
        if len(set(ids)) != len(ids):
            raise IntegrityError("holdout contains duplicate IDs")
        assert_disjoint({"holdout": ids, **other_splits})
        self._holdout = list(holdout)
        self._vault = LabelVault({i: labels[i] for i in ids}, allowed_readers=[self.READER])
        self.eval_batch_size = eval_batch_size
        self.fingerprint = sha256_json({"ids": ids, "x": [[round(v, 12) for v in s.x] for s in holdout],
                                        "labels": [labels[i] for i in ids],
                                        "eval_batch_size": eval_batch_size})

    @property
    def label_reads(self) -> int:
        return self._vault.count()

    def evaluate(self, predict_proba: Callable[[Sequence[Sequence[float]]], List[Vector]],
                 loader: ReplayLoader) -> TerminalResult:
        if not loader.completed:
            raise NonTerminalEvaluationError(
                f"order {loader.order.name}: call loader.verify_complete() before terminal evaluation")
        probs: List[Vector] = []
        # Fixed, order-independent batching: matters only for batch-statistics normalisation.
        for start in range(0, len(self._holdout), self.eval_batch_size):
            chunk = self._holdout[start:start + self.eval_batch_size]
            probs.extend(predict_proba([s.x for s in chunk]))
        preds = [max(range(len(p)), key=p.__getitem__) for p in probs]
        wrong = defaultdict(int)
        count = defaultdict(int)
        for s, yhat in zip(self._holdout, preds):
            y = self._vault.read(self.READER, s.id)
            count[s.domain] += 1
            wrong[s.domain] += int(yhat != y)
        total = sum(count.values())
        per_dom = {d: wrong[d] / count[d] for d in sorted(count)}
        return TerminalResult(sum(wrong.values()) / total, per_dom, preds, probs, self.fingerprint)


class RegimeHoldoutEvaluator:
    """Current-regime holdout for drift streams.

    At the end of a contiguous single-domain block the runner may score the *current* parameters on the
    holdout samples of that block's domain. The holdout is disjoint from every other split, scoring never
    changes parameters, and every label read is counted. Unlike ``TerminalHoldoutEvaluator`` this is a
    mid-stream measurement by design; it must only be called at block boundaries (enforced by the runner).
    """

    READER = "regime_evaluator"

    def __init__(self, holdout: Sequence[Sample], labels: Mapping[str, int],
                 other_splits: Mapping[str, Sequence[str]], eval_batch_size: int):
        ids = [s.id for s in holdout]
        if len(set(ids)) != len(ids):
            raise IntegrityError("holdout contains duplicate IDs")
        assert_disjoint({"holdout": ids, **other_splits})
        self._by_domain: Dict[str, List[Sample]] = defaultdict(list)
        for smp in holdout:
            self._by_domain[smp.domain].append(smp)
        self._vault = LabelVault({i: labels[i] for i in ids}, allowed_readers=[self.READER])
        self.eval_batch_size = eval_batch_size
        self.fingerprint = sha256_json({"ids": ids, "labels": [labels[i] for i in ids],
                                        "eval_batch_size": eval_batch_size, "kind": "regime"})

    @property
    def label_reads(self) -> int:
        return self._vault.count()

    def domains(self) -> List[str]:
        return sorted(self._by_domain)

    def evaluate(self, predict_proba: Callable[[Sequence[Sequence[float]]], List[Vector]], domain: str) -> float:
        rows = self._by_domain[domain]
        if not rows:
            raise IntegrityError(f"no holdout samples for domain {domain!r}")
        wrong = 0
        for start in range(0, len(rows), self.eval_batch_size):
            chunk = rows[start:start + self.eval_batch_size]
            for s, p in zip(chunk, predict_proba([r.x for r in chunk])):
                wrong += int(max(range(len(p)), key=p.__getitem__) != self._vault.read(self.READER, s.id))
        return wrong / len(rows)


def class_distribution(predictions: Sequence[int], n_classes: int) -> Dict[str, object]:
    """Predicted-class histogram and collapse diagnostics (max share, normalised entropy)."""
    n = len(predictions)
    counts = [0] * n_classes
    for p in predictions:
        counts[p] += 1
    shares = [c / n for c in counts] if n else [0.0] * n_classes
    ent = -sum(q * math.log(q) for q in shares if q > 0)
    return {"shares": shares, "max_share": max(shares) if n else 0.0,
            "normalized_entropy": ent / math.log(n_classes) if n_classes > 1 else 0.0,
            "n_classes_predicted": sum(1 for c in counts if c > 0)}


def pairwise_output_difference(results: Sequence[TerminalResult]) -> Dict[str, float]:
    """Output-level order difference among terminal models: mean pairwise prediction-disagreement rate
    and mean pairwise total-variation distance of predictive distributions on the common holdout."""
    k = len(results)
    if k < 2:
        return {"mean_pairwise_disagreement": 0.0, "mean_pairwise_tv": 0.0, "n_pairs": 0}
    dis, tv, npairs = 0.0, 0.0, 0
    for i in range(k):
        for j in range(i + 1, k):
            a, b = results[i], results[j]
            if a.holdout_fingerprint != b.holdout_fingerprint:
                raise IntegrityError("results were computed on different holdouts")
            n = len(a.predictions)
            dis += sum(int(x != y) for x, y in zip(a.predictions, b.predictions)) / n
            tv += sum(0.5 * sum(abs(p - q) for p, q in zip(pa, pb)) for pa, pb in zip(a.probs, b.probs)) / n
            npairs += 1
    return {"mean_pairwise_disagreement": dis / npairs, "mean_pairwise_tv": tv / npairs, "n_pairs": npairs}


__all__ = ["NonTerminalEvaluationError", "TerminalResult", "TerminalHoldoutEvaluator", "RegimeHoldoutEvaluator",
           "class_distribution", "pairwise_output_difference"]

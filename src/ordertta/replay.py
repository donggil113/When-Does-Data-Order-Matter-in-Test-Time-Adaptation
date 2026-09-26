"""Replay one fixed multiset of sample IDs in several fixed orders, with integrity and label-access checks.

Design constraints enforced here:
  * every order is a permutation of the *same* multiset of IDs (duplicates / missing / unknown IDs
    are detected before and after replay);
  * adaptation batches carry IDs and inputs only -- no labels and no domain tags;
  * labels live in a ``LabelVault``; the only stream-label reader is a ``PrequentialLabelOracle`` that
    refuses labels of samples whose prediction has not been committed yet (future-label access),
    and logs every access so that label-free methods can be audited to have made zero accesses.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple


class IntegrityError(RuntimeError):
    """Order / replay does not match the declared multiset."""


class FutureLabelAccessError(RuntimeError):
    """A label was requested before the corresponding prediction was committed."""


class LabelAccessDenied(RuntimeError):
    """A label was requested by a component that is not allowed to read labels."""


@dataclass(frozen=True)
class Sample:
    """Unlabelled sample as seen by adaptation code. ``domain`` is experimenter metadata only."""

    id: str
    x: Tuple[float, ...]
    domain: str


@dataclass(frozen=True)
class Batch:
    """What an adaptation method receives: step index, sample IDs and inputs. No labels, no domains."""

    step: int
    ids: Tuple[str, ...]
    xs: Tuple[Tuple[float, ...], ...]


@dataclass(frozen=True)
class OrderSpec:
    name: str
    family: str
    seed: Optional[int]
    ids: Tuple[str, ...]

    def fingerprint(self) -> str:
        return sha256_json({"family": self.family, "seed": self.seed, "ids": list(self.ids)})


@dataclass
class IntegrityReport:
    duplicates: Dict[str, int] = field(default_factory=dict)  # id -> extra occurrences
    missing: Dict[str, int] = field(default_factory=dict)  # id -> missing occurrences
    unknown: Dict[str, int] = field(default_factory=dict)  # id not in declared multiset

    @property
    def ok(self) -> bool:
        return not (self.duplicates or self.missing or self.unknown)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "duplicates": self.duplicates, "missing": self.missing,
                "unknown": self.unknown}


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compare_multisets(declared: Sequence[str], observed: Sequence[str]) -> IntegrityReport:
    want, got = Counter(declared), Counter(observed)
    rep = IntegrityReport()
    for k, n in got.items():
        if k not in want:
            rep.unknown[k] = n
        elif n > want[k]:
            rep.duplicates[k] = n - want[k]
    for k, n in want.items():
        if got.get(k, 0) < n:
            rep.missing[k] = n - got.get(k, 0)
    return rep


def check_order_integrity(declared: Sequence[str], order: OrderSpec) -> IntegrityReport:
    return compare_multisets(declared, order.ids)


def assert_disjoint(named_id_sets: Mapping[str, Sequence[str]]) -> None:
    """Raise IntegrityError if any two named splits share a sample ID."""
    names = list(named_id_sets)
    sets = {n: set(named_id_sets[n]) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = sets[a] & sets[b]
            if both:
                ex = sorted(both)[:5]
                raise IntegrityError(f"splits {a!r} and {b!r} share {len(both)} IDs, e.g. {ex}")


def make_orders(multiset: Sequence[str], family: str, seeds: Sequence[int],
                domain_of: Optional[Mapping[str, str]] = None,
                domain_sequences: Optional[Sequence[Sequence[str]]] = None) -> List[OrderSpec]:
    """Deterministically build fixed orders of one multiset.

    family:
      ``identity``            -- the multiset as given (one order; seeds ignored beyond the first);
      ``uniform_permutation`` -- stationary mixture: a seeded uniform shuffle per seed;
      ``domain_blocked``      -- drift: contiguous domain blocks; the block sequence is taken from
                                 ``domain_sequences`` (cycled) or seeded-shuffled, and samples are
                                 shuffled within each block with the same seed.
    """
    orders: List[OrderSpec] = []
    base = list(multiset)
    if family == "identity":
        return [OrderSpec("identity", family, None, tuple(base))]
    if family == "uniform_permutation":
        for s in seeds:
            ids = list(base)
            random.Random(s).shuffle(ids)
            orders.append(OrderSpec(f"perm_s{s}", family, s, tuple(ids)))
        return orders
    if family == "domain_blocked":
        if domain_of is None:
            raise ValueError("domain_blocked orders need domain_of")
        blocks: Dict[str, List[str]] = {}
        for i in base:
            blocks.setdefault(domain_of[i], []).append(i)
        domains = sorted(blocks)
        for n, s in enumerate(seeds):
            rng = random.Random(s)
            if domain_sequences:
                seq = list(domain_sequences[n % len(domain_sequences)])
                if sorted(seq) != domains:
                    raise ValueError(f"domain sequence {seq} is not a permutation of {domains}")
            else:
                seq = list(domains)
                rng.shuffle(seq)
            ids: List[str] = []
            for d in seq:
                blk = list(blocks[d])
                rng.shuffle(blk)
                ids.extend(blk)
            orders.append(OrderSpec(f"blocked_s{s}_{'-'.join(seq)}", family, s, tuple(ids)))
        return orders
    raise ValueError(f"unknown order family {family!r}")


class ReplayLoader:
    """Yield label-free batches of one fixed order; verify the multiset was replayed exactly once."""

    def __init__(self, samples: Mapping[str, Sample], declared_multiset: Sequence[str], order: OrderSpec,
                 batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        rep = check_order_integrity(declared_multiset, order)
        if not rep.ok:
            raise IntegrityError(f"order {order.name} does not match the declared multiset: {rep.as_dict()}")
        missing_payload = [i for i in set(order.ids) if i not in samples]
        if missing_payload:
            raise IntegrityError(f"no sample payload for IDs {sorted(missing_payload)[:5]}")
        self._samples = samples
        self._declared = list(declared_multiset)
        self.order = order
        self.batch_size = batch_size
        self._emitted: List[str] = []
        self.completed = False
        self._started = False

    @property
    def num_batches(self) -> int:
        n = len(self.order.ids)
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Batch]:
        if self._started:
            raise IntegrityError("a ReplayLoader replays its order once; build a new loader to replay again")
        self._started = True
        ids = self.order.ids
        for step, start in enumerate(range(0, len(ids), self.batch_size)):
            chunk = ids[start:start + self.batch_size]
            self._emitted.extend(chunk)
            yield Batch(step, tuple(chunk), tuple(self._samples[i].x for i in chunk))

    def emitted(self) -> List[str]:
        return list(self._emitted)

    def verify_complete(self) -> IntegrityReport:
        """Must be called after replay; raises if anything was duplicated, missing or unknown."""
        rep = compare_multisets(self._declared, self._emitted)
        if not rep.ok:
            raise IntegrityError(f"replay of {self.order.name} incomplete/inconsistent: {rep.as_dict()}")
        self.completed = True
        return rep


class LabelVault:
    """Holds labels. Readers must be registered explicitly; every read is logged."""

    def __init__(self, labels: Mapping[str, int], allowed_readers: Sequence[str]):
        self._labels = dict(labels)
        self._allowed = set(allowed_readers)
        self.access_log: List[Tuple[str, str]] = []  # (reader, id)

    def read(self, reader: str, sample_id: str) -> int:
        if reader not in self._allowed:
            raise LabelAccessDenied(f"reader {reader!r} may not read labels")
        self.access_log.append((reader, sample_id))
        return self._labels[sample_id]

    def ids(self) -> List[str]:
        return list(self._labels)

    def count(self, reader: Optional[str] = None) -> int:
        return sum(1 for r, _ in self.access_log if reader is None or r == reader)

    def fingerprint(self, ids: Sequence[str]) -> str:
        return sha256_json([[i, self._labels[i]] for i in ids])


class PrequentialLabelOracle:
    """Online (predict-then-score) access to stream labels.

    A label for a stream position may be read only after the prediction for that position has been
    committed via ``commit``. Anything else raises ``FutureLabelAccessError``. Used for scoring only;
    label-free methods never receive this object.
    """

    READER = "prequential_scorer"

    def __init__(self, vault: LabelVault, order: OrderSpec):
        self._vault = vault
        self._order = order
        self._committed: Counter = Counter()
        self._read: Counter = Counter()
        self._allowed_occurrences: Counter = Counter(order.ids)
        self.violations: List[str] = []

    def commit(self, ids: Sequence[str]) -> None:
        for i in ids:
            self._committed[i] += 1
            if self._committed[i] > self._allowed_occurrences[i]:
                raise IntegrityError(f"prediction for {i} committed more often than it occurs in the order")

    def label(self, sample_id: str) -> int:
        if self._read[sample_id] >= self._committed[sample_id]:
            self.violations.append(sample_id)
            raise FutureLabelAccessError(
                f"label of {sample_id} requested before its prediction was committed")
        self._read[sample_id] += 1
        return self._vault.read(self.READER, sample_id)


__all__ = [
    "IntegrityError", "FutureLabelAccessError", "LabelAccessDenied", "Sample", "Batch", "OrderSpec",
    "IntegrityReport", "compare_multisets", "check_order_integrity", "assert_disjoint", "make_orders",
    "ReplayLoader", "LabelVault", "PrequentialLabelOracle", "sha256_json",
]

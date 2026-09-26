"""Fixed-order replay of one multiset: duplicates, omissions, unknown IDs and future-label access."""

import dataclasses
import unittest
from collections import Counter

from ordertta.replay import (Batch, FutureLabelAccessError, IntegrityError, LabelAccessDenied, LabelVault,
                             OrderSpec, PrequentialLabelOracle, ReplayLoader, Sample, assert_disjoint,
                             compare_multisets, make_orders)


def _world(n=30, domains=("A", "B", "C")):
    samples = {}
    for i in range(n):
        d = domains[i % len(domains)]
        samples[f"s{i:03d}"] = Sample(f"s{i:03d}", (float(i), float(i % 7)), d)
    labels = {k: int(k[1:]) % 3 for k in samples}
    return samples, labels


class Orders(unittest.TestCase):
    def test_uniform_orders_are_deterministic_permutations(self):
        samples, _ = _world()
        ids = list(samples)
        o1 = make_orders(ids, "uniform_permutation", [1, 2, 3])
        o2 = make_orders(ids, "uniform_permutation", [1, 2, 3])
        self.assertEqual([o.ids for o in o1], [o.ids for o in o2])
        self.assertEqual(len({o.ids for o in o1}), 3)
        for o in o1:
            self.assertEqual(Counter(o.ids), Counter(ids))
            self.assertEqual(o.fingerprint(), make_orders(ids, "uniform_permutation", [o.seed])[0].fingerprint())

    def test_domain_blocked_orders_are_contiguous(self):
        samples, _ = _world()
        dom = {k: s.domain for k, s in samples.items()}
        for o in make_orders(list(samples), "domain_blocked", [5, 6, 7, 8], dom):
            seq = [dom[i] for i in o.ids]
            changes = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
            self.assertEqual(changes, 2)  # three blocks
            self.assertEqual(Counter(o.ids), Counter(list(samples)))

    def test_explicit_domain_sequences_validated(self):
        samples, _ = _world()
        dom = {k: s.domain for k, s in samples.items()}
        o = make_orders(list(samples), "domain_blocked", [0], dom, [["C", "A", "B"]])[0]
        self.assertEqual(dom[o.ids[0]], "C")
        self.assertEqual(dom[o.ids[-1]], "B")
        with self.assertRaises(ValueError):
            make_orders(list(samples), "domain_blocked", [0], dom, [["C", "A"]])

    def test_repeated_ids_in_multiset_supported(self):
        samples, _ = _world(3)
        declared = ["s000", "s000", "s001", "s002"]
        order = OrderSpec("x", "explicit", None, ("s001", "s000", "s002", "s000"))
        loader = ReplayLoader(samples, declared, order, batch_size=3)
        emitted = [i for b in loader for i in b.ids]
        self.assertEqual(Counter(emitted), Counter(declared))
        self.assertTrue(loader.verify_complete().ok)


class Integrity(unittest.TestCase):
    def setUp(self):
        self.samples, self.labels = _world(6)
        self.declared = list(self.samples)

    def _order(self, ids):
        return OrderSpec("bad", "explicit", None, tuple(ids))

    def test_duplicate_detected(self):
        ids = self.declared[:-1] + [self.declared[0]]
        rep = compare_multisets(self.declared, ids)
        self.assertEqual(rep.duplicates, {self.declared[0]: 1})
        self.assertEqual(rep.missing, {self.declared[-1]: 1})
        with self.assertRaises(IntegrityError):
            ReplayLoader(self.samples, self.declared, self._order(ids), 2)

    def test_missing_detected(self):
        with self.assertRaises(IntegrityError):
            ReplayLoader(self.samples, self.declared, self._order(self.declared[:-1]), 2)

    def test_unknown_detected(self):
        rep = compare_multisets(self.declared, self.declared + ["zzz"])
        self.assertEqual(rep.unknown, {"zzz": 1})
        with self.assertRaises(IntegrityError):
            ReplayLoader(self.samples, self.declared, self._order(self.declared + ["zzz"]), 2)

    def test_partial_consumption_fails_verification(self):
        loader = ReplayLoader(self.samples, self.declared, self._order(self.declared), 2)
        it = iter(loader)
        next(it)
        with self.assertRaises(IntegrityError):
            loader.verify_complete()
        self.assertFalse(loader.completed)

    def test_single_replay_only(self):
        loader = ReplayLoader(self.samples, self.declared, self._order(self.declared), 4)
        list(loader)
        with self.assertRaises(IntegrityError):
            list(loader)

    def test_last_partial_batch_kept(self):
        loader = ReplayLoader(self.samples, self.declared, self._order(self.declared), 4)
        sizes = [len(b.ids) for b in loader]
        self.assertEqual(sizes, [4, 2])
        loader.verify_complete()

    def test_disjoint_splits(self):
        assert_disjoint({"a": ["1", "2"], "b": ["3"]})
        with self.assertRaises(IntegrityError):
            assert_disjoint({"a": ["1", "2"], "b": ["2", "3"]})


class LabelAccess(unittest.TestCase):
    def setUp(self):
        self.samples, self.labels = _world(8)
        self.declared = list(self.samples)
        self.order = OrderSpec("o", "explicit", None, tuple(self.declared))

    def test_batches_expose_no_labels_or_domains(self):
        self.assertEqual({f.name for f in dataclasses.fields(Batch)}, {"step", "ids", "xs"})
        self.assertNotIn("label", {f.name for f in dataclasses.fields(Sample)})

    def test_vault_denies_unregistered_reader(self):
        vault = LabelVault(self.labels, allowed_readers=["scorer"])
        with self.assertRaises(LabelAccessDenied):
            vault.read("adapter", self.declared[0])
        self.assertEqual(vault.count(), 0)

    def test_prequential_oracle_blocks_future_labels(self):
        vault = LabelVault(self.labels, allowed_readers=[PrequentialLabelOracle.READER])
        oracle = PrequentialLabelOracle(vault, self.order)
        loader = ReplayLoader(self.samples, self.declared, self.order, batch_size=2)
        batches = list(loader)
        # a cheating component peeking at the next batch before predicting it
        with self.assertRaises(FutureLabelAccessError):
            oracle.label(batches[1].ids[0])
        oracle.commit(batches[0].ids)
        for i in batches[0].ids:
            oracle.label(i)
        with self.assertRaises(FutureLabelAccessError):  # a second read needs a second commit
            oracle.label(batches[0].ids[0])
        with self.assertRaises(FutureLabelAccessError):
            oracle.label(batches[1].ids[1])
        self.assertEqual(len(oracle.violations), 3)
        self.assertEqual(vault.count(), 2)

    def test_overcommit_detected(self):
        vault = LabelVault(self.labels, allowed_readers=[PrequentialLabelOracle.READER])
        oracle = PrequentialLabelOracle(vault, self.order)
        oracle.commit([self.declared[0]])
        with self.assertRaises(IntegrityError):
            oracle.commit([self.declared[0]])


if __name__ == "__main__":
    unittest.main()

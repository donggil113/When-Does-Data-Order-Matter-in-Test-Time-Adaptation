"""Adaptation-subspace API, fitters and the separate cost ledger."""

import random
import unittest

from ordertta import linalg as la
from ordertta.quadratic import Quadratic, projected_losses, two_step_second_order_term
from ordertta.subspace import (AdaptationSubspace, CostLedger, CostRecord, MetaBatchStats, full_space,
                               gradient_pca_subspace, order_aware_subspace, random_subspace)


class API(unittest.TestCase):
    def test_rejects_non_orthonormal_basis(self):
        with self.assertRaises(ValueError):
            AdaptationSubspace("bad", [[1.0, 1.0, 0.0]], 3)
        with self.assertRaises(ValueError):
            AdaptationSubspace("bad", [[1.0, 0.0], [1.0, 0.0]], 2)

    def test_projection_is_idempotent_and_lift_coords_roundtrip(self):
        sub = random_subspace(6, 3, seed=4)
        v = la.random_vector(random.Random(0), 6)
        p = sub.project(v)
        self.assertLess(la.max_abs(la.sub(sub.project(p), p)), 1e-12)
        z = [0.3, -1.0, 2.0]
        self.assertLess(la.max_abs(la.sub(sub.coords(sub.lift(z)), z)), 1e-12)
        self.assertEqual(full_space(4).project([1.0, 2.0, 3.0, 4.0]), [1.0, 2.0, 3.0, 4.0])

    def test_empty_subspace_means_no_movement(self):
        sub = AdaptationSubspace("empty", [], 3)
        self.assertEqual(sub.project([1.0, 2.0, 3.0]), [0.0, 0.0, 0.0])

    def test_random_subspace_is_seeded(self):
        self.assertEqual(random_subspace(5, 2, 1).fingerprint(), random_subspace(5, 2, 1).fingerprint())
        self.assertNotEqual(random_subspace(5, 2, 1).fingerprint(), random_subspace(5, 2, 2).fingerprint())

    def test_gradient_pca_recovers_dominant_direction(self):
        rng = random.Random(0)
        u = la.orthonormalize([[1.0, 2.0, 0.0, -1.0]])[0]
        grads = [la.add(la.scale(u, rng.gauss(0, 5.0)), la.random_vector(rng, 4, 0.1)) for _ in range(200)]
        sub = gradient_pca_subspace(grads, 1)
        self.assertGreater(abs(la.dot(sub.basis[0], u)), 0.999)


def _stats_for_quadratics(losses, theta, pool, sup_grads):
    out = []
    for q, gs in zip(losses, sup_grads):
        h = q.hess(theta)
        out.append(MetaBatchStats(q.grad(theta), gs, [la.matvec(h, p) for p in pool]))
    return out


class OrderAwareFitter(unittest.TestCase):
    """Axis 0: useful but order-sensitive (non-commuting curvature); axis 1: useful and commuting; axis 2: useless."""

    def setUp(self):
        self.theta = [0.0, 0.0, 0.0]
        self.losses = [Quadratic([[2.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], [1.0, 1.0, 0.0]),
                       Quadratic([[0.2, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], [-3.0, 1.0, 0.0])]
        self.pool = la.eye(3)
        self.sup = [q.grad(self.theta) for q in self.losses]  # supervised loss aligned with adaptation loss
        self.stats = _stats_for_quadratics(self.losses, self.theta, self.pool, self.sup)
        self.jvp = [list(p) for p in self.pool]  # identity output map

    def test_lambda_zero_picks_gain(self):
        sub = order_aware_subspace(self.pool, self.stats, self.jvp, k=1, lr=0.5, lam=0.0)
        self.assertEqual(sub.provenance["chosen_pool_indices"], [0])  # axis 0 has the largest gain

    def test_large_lambda_avoids_order_sensitive_direction(self):
        sub = order_aware_subspace(self.pool, self.stats, self.jvp, k=2, lr=0.5, lam=50.0)
        self.assertEqual(sub.provenance["chosen_pool_indices"], [1])
        self.assertEqual(sub.provenance["stopped"], "no_improving_direction")

    def test_returns_empty_when_nothing_helps(self):
        # adaptation steps that increase the supervised loss along every pool direction
        harmful = [la.scale(g, -1.0) for g in self.sup]
        stats = _stats_for_quadratics(self.losses, self.theta, self.pool, harmful)
        sub = order_aware_subspace(self.pool, stats, self.jvp, k=2, lr=0.5, lam=0.0)
        self.assertEqual(sub.dim, 0)
        self.assertEqual(sub.provenance["stopped"], "no_improving_direction")

    def test_commuting_direction_survives_any_lambda(self):
        sub = order_aware_subspace(self.pool, self.stats, self.jvp, k=2, lr=0.5, lam=1e9)
        self.assertEqual(sub.provenance["chosen_pool_indices"], [1])

    def test_chosen_subspace_has_smaller_local_order_term(self):
        """Numerical fixture: second-order output-level order term in the chosen subspace vs. full space."""
        sub = order_aware_subspace(self.pool, self.stats, self.jvp, k=2, lr=0.5, lam=50.0)
        a, b = self.losses
        full = la.norm(two_step_second_order_term(self.theta, a, b, 0.5))
        pa, pb = projected_losses([a, b], sub.basis, self.theta)
        restricted = la.norm(sub.lift(two_step_second_order_term([0.0] * sub.dim, pa, pb, 0.5)))
        self.assertGreater(full, 0.1)
        self.assertLess(restricted, 1e-12)

    def test_normalized_score_is_scale_free(self):
        big = [MetaBatchStats(la.scale(st.g_adapt, 10.0), la.scale(st.g_sup, 10.0), st.h_pool) for st in self.stats]
        s1 = order_aware_subspace(self.pool, self.stats, self.jvp, k=3, lr=0.5, lam=1.0, normalized=True)
        s2 = order_aware_subspace(self.pool, big, [la.scale(j, 3.0) for j in self.jvp], k=3, lr=7.0, lam=1.0,
                                  normalized=True)
        self.assertEqual(s1.provenance["chosen_pool_indices"], s2.provenance["chosen_pool_indices"])
        # lam = 0: the score is the retained gain fraction; the useless axis 2 adds nothing, so it is not picked
        s0 = order_aware_subspace(self.pool, self.stats, self.jvp, k=3, lr=0.5, lam=0.0, normalized=True)
        self.assertEqual(sorted(s0.provenance["chosen_pool_indices"]), [0, 1])
        picked = [t for t in s0.provenance["greedy_trace"] if "picked" in t]
        self.assertAlmostEqual(picked[-1]["score"], 1.0, places=12)

    def test_needs_two_batches(self):
        with self.assertRaises(ValueError):
            order_aware_subspace(self.pool, self.stats[:1], self.jvp, k=1, lr=0.5, lam=0.0)


class Ledger(unittest.TestCase):
    def test_meta_and_test_costs_are_kept_apart(self):
        led = CostLedger()
        led.add(CostRecord("meta_training:oa", "meta_train", 10, "h", True, gradient_evals=5, hvp_evals=7))
        led.add(CostRecord("test_time:tent/o1", "test_stream", 20, "h2", False, gradient_evals=3))
        led.add(CostRecord("test_time:tent/o2", "test_stream", 20, "h2", False, gradient_evals=4))
        self.assertEqual(led.totals("meta_training")["hvp_evals"], 7)
        self.assertTrue(led.totals("meta_training")["any_labels"])
        self.assertEqual(led.totals("test_time")["gradient_evals"], 7)
        self.assertFalse(led.totals("test_time")["any_labels"])
        self.assertIn("source_training", led.as_dict()["totals"])


if __name__ == "__main__":
    unittest.main()

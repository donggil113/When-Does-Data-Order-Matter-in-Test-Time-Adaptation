"""Plain-SGD order difference vs. the local second-order term (numerical fixtures, not proofs)."""

import random
import unittest

from ordertta import linalg as la
from ordertta.quadratic import (Quadratic, make_random_quadratic, make_random_quartic,
                                multi_step_order_difference, multi_step_second_order_prediction,
                                output_level_prediction, projected_losses, residual_order_estimate, sgd_final,
                                sgd_path, two_step_order_difference, two_step_second_order_term)


class TwoStepQuadratic(unittest.TestCase):
    def test_exact_identity_for_quadratics(self):
        for seed in range(20):
            rng = random.Random(seed)
            dim = rng.randint(2, 6)
            a, b = make_random_quadratic(rng, dim), make_random_quadratic(rng, dim)
            theta = la.random_vector(rng, dim)
            for lr in (1e-3, 0.05, 0.3, 1.0):  # exact at any step size, not only small ones
                diff = two_step_order_difference(theta, a, b, lr)
                pred = two_step_second_order_term(theta, a, b, lr)
                scale = max(1.0, la.max_abs(diff))
                self.assertLess(la.max_abs(la.sub(diff, pred)), 1e-12 * scale, (seed, lr))

    def test_same_curvature_different_targets_is_order_dependent(self):
        rng = random.Random(1)
        A = la.random_spd(rng, 3)
        ca, cb = la.random_vector(rng, 3), la.random_vector(rng, 3)
        theta, lr = la.random_vector(rng, 3), 0.1
        diff = two_step_order_difference(theta, Quadratic(A, ca), Quadratic(A, cb), lr)
        expected = la.scale(la.matvec(la.matmul(A, A), la.sub(cb, ca)), lr * lr)
        self.assertLess(la.max_abs(la.sub(diff, expected)), 1e-13)
        self.assertGreater(la.norm(diff), 1e-4)

    def test_identical_losses_commute(self):
        rng = random.Random(2)
        q = make_random_quadratic(rng, 4)
        diff = two_step_order_difference(la.random_vector(rng, 4), q, q, 0.2)
        self.assertEqual(la.max_abs(diff), 0.0)

    def test_commuting_hessians_shared_minimiser_commute(self):
        c = [0.3, -1.0, 2.0]
        qa = Quadratic([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 0.5]], c)
        qb = Quadratic([[0.2, 0, 0], [0, 1.5, 0], [0, 0, 3.0]], c)
        diff = two_step_order_difference([1.0, 1.0, 1.0], qa, qb, 0.1)
        self.assertLess(la.max_abs(diff), 1e-15)


class NonQuadraticResidual(unittest.TestCase):
    def test_two_step_residual_is_third_order(self):
        for seed in range(5):
            rng = random.Random(100 + seed)
            a, b = make_random_quartic(rng, 3, kappa=0.8), make_random_quartic(rng, 3, kappa=0.8)
            theta = la.random_vector(rng, 3)

            def resid(lr):
                return la.norm(la.sub(two_step_order_difference(theta, a, b, lr),
                                      two_step_second_order_term(theta, a, b, lr)))

            # step sizes are expressed in units of the local curvature scale L (largest |Hessian entry|):
            # the expansion is asymptotic in eta * L, not in eta alone (see test_second_order_term_is_only_local)
            L = max(abs(x) for q in (a, b) for row in q.hess(theta) for x in row)
            orders = residual_order_estimate(resid, [c / L for c in (0.08, 0.04, 0.02, 0.01, 0.005)])
            for p in orders[1:]:
                self.assertGreater(p, 2.8, (seed, orders))
                self.assertLess(p, 3.2, (seed, orders))
            # relative residual is O(eta * L): small once eta * L << 1
            lr = 1e-3 / L
            rel = resid(lr) / la.norm(two_step_second_order_term(theta, a, b, lr))
            self.assertLess(rel, 0.05, (seed, L, rel))

    def test_second_order_term_is_only_local(self):
        """Scope check: at eta * L ~ 0.3 the cubic remainder is comparable to the second-order term."""
        rng = random.Random(103)
        a, b = make_random_quartic(rng, 3, kappa=0.8), make_random_quartic(rng, 3, kappa=0.8)
        theta = la.random_vector(rng, 3)
        lr = 0.01
        pred = two_step_second_order_term(theta, a, b, lr)
        rel = la.norm(la.sub(two_step_order_difference(theta, a, b, lr), pred)) / la.norm(pred)
        self.assertGreater(rel, 0.1)

    def test_multi_step_prediction_residual_is_third_order(self):
        rng = random.Random(7)
        losses = [make_random_quadratic(rng, 4) for _ in range(4)]
        theta = la.random_vector(rng, 4)
        o1, o2 = [0, 1, 2, 3], [3, 1, 0, 2]

        def resid(lr):
            return la.norm(la.sub(multi_step_order_difference(theta, losses, o1, o2, lr),
                                  multi_step_second_order_prediction(theta, losses, o1, o2, lr)))

        orders = residual_order_estimate(resid, [0.02, 0.01, 0.005, 0.0025])
        for p in orders:
            self.assertGreater(p, 2.8, orders)
            self.assertLess(p, 3.2, orders)

    def test_permutations_of_different_multisets_rejected(self):
        rng = random.Random(8)
        losses = [make_random_quadratic(rng, 2) for _ in range(3)]
        with self.assertRaises(ValueError):
            multi_step_order_difference([0.0, 0.0], losses, [0, 1, 2], [0, 1, 1], 0.1)


class OutputLevel(unittest.TestCase):
    def test_linear_output_difference_is_exact_for_two_quadratic_steps(self):
        rng = random.Random(3)
        a, b = make_random_quadratic(rng, 5), make_random_quadratic(rng, 5)
        theta, u, lr = la.random_vector(rng, 5), la.random_vector(rng, 5), 0.2
        exact = la.dot(u, sgd_final(theta, [a, b], lr)) - la.dot(u, sgd_final(theta, [b, a], lr))
        pred = output_level_prediction(u, two_step_second_order_term(theta, a, b, lr))
        self.assertAlmostEqual(exact, pred, delta=1e-12)

    def test_nonlinear_output_residual_is_third_order(self):
        rng = random.Random(4)
        a, b = make_random_quadratic(rng, 3), make_random_quadratic(rng, 3)
        theta = la.random_vector(rng, 3)
        f = lambda t: 0.5 * la.dot(t, t) + t[0] ** 3  # noqa: E731
        grad_f = [theta[0] + 3 * theta[0] ** 2, theta[1], theta[2]]

        def resid(lr):
            exact = f(sgd_final(theta, [a, b], lr)) - f(sgd_final(theta, [b, a], lr))
            return abs(exact - output_level_prediction(grad_f, two_step_second_order_term(theta, a, b, lr)))

        orders = residual_order_estimate(resid, [0.02, 0.01, 0.005, 0.0025])
        for p in orders:
            self.assertGreater(p, 2.7, orders)

    def test_output_orthogonal_to_subspace_is_order_invariant(self):
        rng = random.Random(5)
        losses = [make_random_quadratic(rng, 4) for _ in range(3)]
        basis = la.orthonormalize([[1, 1, 0, 0], [0, 0, 1, 0]])
        anchor = la.random_vector(rng, 4)
        u = [1.0, -1.0, 0.0, 0.0]  # orthogonal to span(basis)
        proj = projected_losses(losses, basis, anchor)
        outs = []
        for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
            z = sgd_final([0.0, 0.0], [proj[k] for k in order], 0.3)
            outs.append(la.dot(u, proj[0].lift(z)))
        self.assertLess(max(outs) - min(outs), 1e-12)

    def test_projected_two_step_identity(self):
        rng = random.Random(6)
        a, b = make_random_quadratic(rng, 5), make_random_quadratic(rng, 5)
        basis = la.orthonormalize(la.random_matrix(rng, 2, 5))
        pa, pb = projected_losses([a, b], basis, la.random_vector(rng, 5))
        z0 = [0.1, -0.2]
        diff = two_step_order_difference(z0, pa, pb, 0.4)
        pred = two_step_second_order_term(z0, pa, pb, 0.4)
        self.assertLess(la.max_abs(la.sub(diff, pred)), 1e-12)


class Scope(unittest.TestCase):
    def test_only_plain_sgd_is_supported(self):
        rng = random.Random(9)
        q = make_random_quadratic(rng, 2)
        for opt in ("adam", "momentum", "adamw"):
            with self.assertRaises(NotImplementedError):
                sgd_path([0.0, 0.0], [q], 0.1, optimizer=opt)


if __name__ == "__main__":
    unittest.main()

"""Toy model gradients, terminal evaluator, and the no-adapt / small-lr / norm-matched controls."""

import copy
import math
import random
import unittest

from ordertta import linalg as la
from ordertta.evaluator import NonTerminalEvaluationError, TerminalHoldoutEvaluator, pairwise_output_difference
from ordertta.methods import (EntropySGD, NoAdapt, calibrate_lr_to_displacement, per_step_norm_schedule,
                              small_lr)
from ordertta.replay import IntegrityError, ReplayLoader, make_orders
from ordertta.subspace import random_subspace
from ordertta.toy import build_world, train_source_model

WORLD = {"seed": 3, "dim": 3, "n_classes": 3, "class_sep": 1.6, "sigma": 1.0, "n_test_domains": 2,
         "n_val_domains": 1, "n_meta_domains": 2, "scale_range": [0.5, 1.6], "shift_sd": 0.8, "domain_noise": 0.1,
         "n_per_class": {"source_train": 30, "meta_train": 6, "development": 4, "dev_holdout": 4,
                         "calibration": 4, "test_stream": 8, "test_holdout": 10}}


def _setup(norm="source"):
    world = build_world(WORLD)
    model, _ = train_source_model(world.splits["source_train"], 3, 80, 0.5, norm, seed=0)
    samples = {s.id: s for s in world.samples("test_stream")}
    others = {k: world.ids(k) for k in world.splits if k != "test_holdout"}
    ev = TerminalHoldoutEvaluator(world.samples("test_holdout"), world.labels("test_holdout"), others, 30)
    return world, model, samples, ev


def _replay(adapter, world, samples, order, ev, bs=8):
    loader = ReplayLoader(samples, world.ids("test_stream"), order, bs)
    for b in loader:
        adapter.step(b)
    loader.verify_complete()
    return ev.evaluate(adapter.predict_proba, loader)


class Gradients(unittest.TestCase):
    def _fd_check(self, fn, theta, tol=1e-6):
        _, g = fn(theta)
        for i in range(len(theta)):
            h = 1e-6
            tp, tm = list(theta), list(theta)
            tp[i] += h
            tm[i] -= h
            fd = (fn(tp)[0] - fn(tm)[0]) / (2 * h)
            self.assertAlmostEqual(g[i], fd, delta=tol * max(1.0, abs(fd)))

    def test_entropy_and_ce_gradients_match_finite_differences(self):
        for norm in ("source", "batch"):
            world, model, _, _ = _setup(norm)
            rows = world.splits["meta_train"][:12]
            xs, ys = [s.x for s, _ in rows], [y for _, y in rows]
            theta = la.add(model.initial_theta(), la.random_vector(random.Random(0), model.theta_dim, 0.2))
            self._fd_check(lambda t: model.entropy_and_grad(xs, t), theta)
            self._fd_check(lambda t: model.ce_and_grad(xs, ys, t), theta)

    def test_hvp_is_symmetric(self):
        world, model, _, _ = _setup()
        xs = [s.x for s, _ in world.splits["meta_train"][:12]]
        theta = model.initial_theta()
        rng = random.Random(1)
        v, w = la.random_vector(rng, model.theta_dim), la.random_vector(rng, model.theta_dim)
        self.assertAlmostEqual(la.dot(w, model.entropy_hvp(xs, theta, v)),
                               la.dot(v, model.entropy_hvp(xs, theta, w)), delta=1e-5)


class Evaluator(unittest.TestCase):
    def test_non_terminal_evaluation_refused(self):
        world, model, samples, ev = _setup()
        order = make_orders(world.ids("test_stream"), "uniform_permutation", [0])[0]
        loader = ReplayLoader(samples, world.ids("test_stream"), order, 8)
        a = NoAdapt(copy.deepcopy(model))
        for b in loader:
            a.step(b)
            break
        with self.assertRaises(NonTerminalEvaluationError):
            ev.evaluate(a.predict_proba, loader)

    def test_holdout_must_be_disjoint_from_stream(self):
        world, model, _, _ = _setup()
        with self.assertRaises(IntegrityError):
            TerminalHoldoutEvaluator(world.samples("test_holdout"), world.labels("test_holdout"),
                                     {"test_stream": world.ids("test_stream") + [world.ids("test_holdout")[0]]}, 30)

    def test_evaluation_does_not_change_parameters(self):
        world, model, samples, ev = _setup()
        order = make_orders(world.ids("test_stream"), "uniform_permutation", [4])[0]
        a = EntropySGD(copy.deepcopy(model), "tent", 0.5)
        loader = ReplayLoader(samples, world.ids("test_stream"), order, 8)
        for b in loader:
            a.step(b)
        loader.verify_complete()
        before = list(a.theta)
        r1 = ev.evaluate(a.predict_proba, loader)
        r2 = ev.evaluate(a.predict_proba, loader)
        self.assertEqual(before, a.theta)
        self.assertEqual(r1.predictions, r2.predictions)

    def test_pairwise_output_difference(self):
        world, model, samples, ev = _setup()
        orders = make_orders(world.ids("test_stream"), "uniform_permutation", [0, 1, 2])
        res = [_replay(NoAdapt(copy.deepcopy(model)), world, samples, o, ev) for o in orders]
        d = pairwise_output_difference(res)
        self.assertEqual((d["mean_pairwise_disagreement"], d["mean_pairwise_tv"], d["n_pairs"]), (0.0, 0.0, 3))


class Controls(unittest.TestCase):
    def test_no_adapt_is_exactly_order_invariant(self):
        for norm in ("source", "batch"):
            world, model, samples, ev = _setup(norm)
            orders = make_orders(world.ids("test_stream"), "uniform_permutation", [0, 1, 2, 3])
            errs = []
            for o in orders:
                a = NoAdapt(copy.deepcopy(model))
                errs.append(_replay(a, world, samples, o, ev).error)
                self.assertEqual(a.theta, a.theta0)
            self.assertEqual(len(set(errs)), 1)

    def test_adaptation_is_order_dependent_in_this_fixture(self):
        world, model, samples, ev = _setup()
        orders = make_orders(world.ids("test_stream"), "uniform_permutation", [0, 1])
        thetas = []
        for o in orders:
            a = EntropySGD(copy.deepcopy(model), "tent", 0.5)
            _replay(a, world, samples, o, ev)
            thetas.append(a.theta)
        self.assertGreater(la.norm(la.sub(*thetas)), 1e-6)

    def test_small_lr_validation_and_first_order_scaling(self):
        self.assertAlmostEqual(small_lr(0.5, 0.1), 0.05)
        for bad in (0.0, 1.0, 1.5, -0.1):
            with self.assertRaises(ValueError):
                small_lr(0.5, bad)
        world, model, samples, ev = _setup()
        o = make_orders(world.ids("test_stream"), "uniform_permutation", [0])[0]
        disp = {}
        for lr in (1e-3, 1e-4):
            a = EntropySGD(copy.deepcopy(model), "tent", lr)
            _replay(a, world, samples, o, ev)
            disp[lr] = a.displacement()
        self.assertAlmostEqual(disp[1e-3] / disp[1e-4], 10.0, delta=0.05)

    def test_per_step_norm_matching_copies_step_norms(self):
        world, model, samples, ev = _setup()
        o = make_orders(world.ids("test_stream"), "uniform_permutation", [2])[0]
        cand = EntropySGD(copy.deepcopy(model), "cand", 0.5, random_subspace(model.theta_dim, 2, 0))
        _replay(cand, world, samples, o, ev)
        norms = per_step_norm_schedule(cand.trace)
        ctrl = EntropySGD(copy.deepcopy(model), "ctrl", 0.5, None, norms)
        _replay(ctrl, world, samples, o, ev)
        for r, t in zip(ctrl.trace, norms):
            self.assertAlmostEqual(r.update_norm, t, delta=1e-12)
        short = EntropySGD(copy.deepcopy(model), "ctrl", 0.5, None, norms[:1])
        with self.assertRaises(IndexError):
            _replay(short, world, samples, o, ev)

    def test_global_calibration_hits_target_and_reports_mismatch(self):
        res = calibrate_lr_to_displacement(lambda lr: 3.0 * lr, 0.6, 1e-4, 10.0, iters=60)
        self.assertLess(res.relative_mismatch, 1e-3)
        self.assertAlmostEqual(res.lr, 0.2, delta=1e-3)
        unreachable = calibrate_lr_to_displacement(lambda lr: min(lr, 0.1), 5.0, 1e-4, 10.0, iters=40)
        self.assertGreater(unreachable.relative_mismatch, 0.5)
        self.assertTrue(math.isfinite(unreachable.lr))
        still = calibrate_lr_to_displacement(lambda lr: lr, 0.0, 1e-4, 10.0)
        self.assertEqual((still.lr, still.relative_mismatch), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()

"""Across-order statistics and the pre-registered decision rule on synthetic error tables."""

import unittest

from ordertta.analysis import (compare_order_families, decide, nearest_rank_quantile, order_stats,
                               paired_bootstrap_ci)

K = 12
WIGGLE = [((i * 7) % K - (K - 1) / 2) / ((K - 1) / 2) for i in range(K)]  # deterministic zero-mean pattern in [-1, 1]

RULE = {"candidate": "cand", "baseline": "base", "no_adapt": "none",
        "stability_controls": ["small"], "dimension_controls": [],
        "spread_metric": "std_terminal_error",
        "mie": {"spread_abs": 0.005, "gain_abs": 0.01}, "tolerance": {"spread_abs": 0.0025, "gain_abs": 0.005},
        "n_boot": 400, "boot_seed": 0}


def arm(mean, amp):
    return [mean + amp * w for w in WIGGLE]


def table(**arms):
    return {"r0": dict(arms)}


def ok(*names):
    return {n: "OK" for n in names}


class Stats(unittest.TestCase):
    def test_nearest_rank_quantile(self):
        xs = [0.1, 0.4, 0.2, 0.3]
        self.assertEqual(nearest_rank_quantile(xs, 0.9), 0.4)
        self.assertEqual(nearest_rank_quantile(xs, 0.5), 0.2)

    def test_order_stats(self):
        s = order_stats([0.2, 0.3, 0.25], [0.4, 0.4, 0.4])
        self.assertAlmostEqual(s["mean_error"], 0.25)
        self.assertAlmostEqual(s["observed_worst_error"], 0.3)
        self.assertAlmostEqual(s["mean_gain"], 0.15)
        self.assertAlmostEqual(s["observed_worst_gain"], 0.1)

    def test_bootstrap_ci_is_paired(self):
        t = table(a=arm(0.3, 0.02), b=[x + 0.01 for x in arm(0.3, 0.02)], none=[0.4] * K)
        ci = paired_bootstrap_ci(t, "a", "b", "none", "gain", 200, 0)
        self.assertAlmostEqual(ci["point"], 0.01)
        self.assertAlmostEqual(ci["ci_lo"], 0.01)  # the shift is constant across orders
        self.assertAlmostEqual(ci["ci_hi"], 0.01)


class Families(unittest.TestCase):
    def test_compare_order_families(self):
        a = table(base=arm(0.3, 0.002))
        b = table(base=arm(0.3, 0.05))
        c = compare_order_families(a, b, "base", "std_terminal_error", 300, 0)
        self.assertGreater(c["point"], 0.03)
        self.assertGreater(c["ci_lo"], 0.0)
        with self.assertRaises(ValueError):
            compare_order_families(a, {"r1": b["r0"]}, "base", "std_terminal_error", 10, 0)


class Decision(unittest.TestCase):
    def test_incomplete_when_any_required_arm_not_ok(self):
        t = table(cand=arm(0.3, 0.0), base=arm(0.3, 0.05), none=[0.4] * K, small=arm(0.4, 0.0))
        d = decide(t, {"cand": "OK", "base": "FAILED", "none": "OK", "small": "OK"}, RULE)
        self.assertEqual(d["label"], "INCOMPLETE")
        d = decide(t, {"cand": "OK", "base": "OK", "none": "OK"}, RULE)  # control never ran
        self.assertEqual(d["label"], "INCOMPLETE")

    def test_stable_only_because_not_adapting(self):
        t = table(cand=[0.4] * K, base=arm(0.3, 0.05), none=[0.4] * K, small=arm(0.39, 0.001))
        d = decide(t, ok("cand", "base", "none", "small"), RULE)
        self.assertEqual(d["label"], "NO_ADDED_UTILITY_GAIN_BELOW_MIE")

    def test_no_stabilisation(self):
        t = table(cand=arm(0.3, 0.05), base=arm(0.3, 0.05), none=[0.4] * K, small=arm(0.39, 0.001))
        d = decide(t, ok("cand", "base", "none", "small"), RULE)
        self.assertEqual(d["label"], "NO_STABILIZATION_OBSERVED")

    def test_matched_by_small_lr_control(self):
        t = table(cand=arm(0.32, 0.005), base=arm(0.3, 0.05), none=[0.4] * K, small=arm(0.321, 0.005))
        d = decide(t, ok("cand", "base", "none", "small"), RULE)
        self.assertEqual(d["label"], "NO_ADDED_UTILITY_VS_CONTROLS")
        self.assertEqual(d["undominated_controls"], ["small"])

    def test_dominates_controls_on_sampled_orders(self):
        t = table(cand=arm(0.30, 0.004), base=arm(0.3, 0.05), none=[0.4] * K, small=arm(0.36, 0.004))
        d = decide(t, ok("cand", "base", "none", "small"), RULE)
        self.assertEqual(d["label"], "ADDED_UTILITY_OBSERVED_ON_SAMPLED_ORDERS")
        self.assertIn("not a guarantee", d["caveat"])


if __name__ == "__main__":
    unittest.main()

"""Across-order statistics and the pre-registered decision rule on synthetic error tables."""

import unittest

from ordertta.analysis import (compare_order_families, decide, decide_stage, nearest_rank_quantile, order_stats,
                               paired_bootstrap_ci, seedwise_condition_summary)

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


ARMS = ["none", "base", "cand", "util", "small"]
STAGE_RULE = {"candidate": "cand", "no_adapt": "none", "baseline": "base", "required_controls": ["util", "small"],
              "drift_reference": "util", "mie": {"spread_abs": 0.005, "gain_abs": 0.01},
              "tolerance": {"spread_abs": 0.0025, "gain_abs": 0.005}, "n_seeds": 3}
W8 = WIGGLE[:8]


def cond(**arms):
    """Three seeds with the same per-order pattern (plus a per-seed offset) for every arm."""
    return {f"r{k}": {a: [v[0] + 0.001 * k + v[1] * w for w in W8] for a, v in arms.items()} for k in range(3)}


def summ(t):
    return seedwise_condition_summary(t, ARMS, "cand", "none", [a for a in ARMS if a != "cand"])


def stage(stat, drift, drift_online=None, statuses=None):
    st = statuses or {"stat": {a: "OK" for a in ARMS}, "drift": {a: "OK" for a in ARMS}}
    return decide_stage(summ(stat), summ(drift), st, dict(STAGE_RULE, drift_online=summ(drift_online or drift)))


GOOD_STAT = dict(none=(0.40, 0.0), base=(0.30, 0.05), cand=(0.30, 0.004), util=(0.30, 0.03), small=(0.36, 0.004))
GOOD_DRIFT = dict(none=(0.40, 0.0), base=(0.30, 0.02), cand=(0.31, 0.01), util=(0.31, 0.01), small=(0.35, 0.01))


class StageDecision(unittest.TestCase):
    def test_seedwise_summary_has_no_pooled_ci(self):
        s = summ(cond(**GOOD_STAT))
        self.assertEqual(s["seeds"], ["r0", "r1", "r2"])
        self.assertEqual(len(s["paired_vs_candidate"]["util"]["per_seed"]), 3)
        self.assertNotIn("ci_lo", str(s))

    def test_candidate_flag(self):
        d = stage(cond(**GOOD_STAT), cond(**GOOD_DRIFT))
        self.assertEqual(d["label"], "REALDATA_PILOT_CANDIDATE", d.get("reasons"))

    def test_incomplete_when_required_arm_missing(self):
        st = {"stat": {a: "OK" for a in ARMS}, "drift": dict({a: "OK" for a in ARMS}, util="CAP_EXCEEDED")}
        self.assertEqual(stage(cond(**GOOD_STAT), cond(**GOOD_DRIFT), statuses=st)["label"], "INCOMPLETE")

    def test_same_as_utility_only_is_not_supported(self):
        d = stage(cond(**dict(GOOD_STAT, util=(0.30, 0.004))), cond(**GOOD_DRIFT))
        self.assertEqual(d["label"], "ORDER_PENALTY_ADDED_UTILITY_NOT_SUPPORTED")
        self.assertIn("NOT_BETTER_THAN_util", d["reasons"])

    def test_explained_by_small_lr(self):
        d = stage(cond(**dict(GOOD_STAT, small=(0.301, 0.004))), cond(**GOOD_DRIFT))
        self.assertIn("NOT_BETTER_THAN_small", d["reasons"])

    def test_stable_but_no_gain_is_not_success(self):
        d = stage(cond(**dict(GOOD_STAT, cand=(0.40, 0.0))), cond(**GOOD_DRIFT))
        self.assertIn("ADAPTATION_GAIN_LOST_OR_BELOW_MIE", d["reasons"])

    def test_drift_damage_blocks_flag(self):
        d = stage(cond(**GOOD_STAT), cond(**dict(GOOD_DRIFT, cand=(0.33, 0.01))))
        self.assertIn("DRIFT_DAMAGE_REGIME_END_ERROR", d["reasons"])

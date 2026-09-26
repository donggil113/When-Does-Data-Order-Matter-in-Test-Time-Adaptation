"""End-to-end runner checks on a tiny config: outputs, audits, and preservation of failures / caps."""

import copy
import glob
import json
import os
import shutil
import tempfile
import unittest

from ordertta.runner import ConfigError, dry_run, run, validate_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = os.path.join(ROOT, "configs", "cpu_smoke.json")


def tiny_config():
    with open(SMOKE) as f:
        cfg = json.load(f)
    cfg["world"]["n_per_class"] = {"source_train": 30, "meta_train": 8, "development": 6, "dev_holdout": 6,
                                   "calibration": 6, "test_stream": 8, "test_holdout": 10}
    cfg["model"]["source_epochs"] = 60
    cfg["stream"]["order_seeds"] = [1, 2, 3]
    cfg["decision"]["n_boot"] = 100
    for m in cfg["methods"]:
        if m.get("norm_match", {}).get("mode") == "calibrated_global":
            m["norm_match"]["iters"] = 20
            m["norm_match"]["calibration_order_seeds"] = [301]
    return cfg


class RunnerTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ordertta_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _run(self, cfg, run_id):
        path = os.path.join(self.tmp, f"{run_id}.json")
        with open(path, "w") as f:
            json.dump(cfg, f)
        out = run(path, os.path.join(self.tmp, "runs"), run_id)
        with open(os.path.join(out, "summary.json")) as f:
            summary = json.load(f)
        with open(os.path.join(out, "manifest.json")) as f:
            manifest = json.load(f)
        with open(os.path.join(out, "raw_log.jsonl")) as f:
            events = [json.loads(line) for line in f]
        return summary, manifest, events


class EndToEnd(RunnerTestCase):
    def test_outputs_audits_and_invariants(self):
        s, m, events = self._run(tiny_config(), "e2e")
        self.assertEqual(s["software_status"], "RUN_COMPLETED")
        self.assertEqual(s["science_status"], "SCIENCE_NOT_EVALUATED")
        self.assertTrue(all(v == "OK" for v in s["method_status"].values()), s["method_status"])
        rep = s["replicates"][0]
        # adapters never read stream labels; only the prequential scorer does, after committing predictions
        self.assertTrue(all(v == 0 for v in rep["non_scorer_stream_label_reads"].values()))
        # no-adaptation is order-invariant by construction
        na = s["metrics"]["no_adapt"]["pooled_mean_over_replicates"]
        self.assertEqual(na["std_error"], 0.0)
        self.assertEqual(rep["output_level_order_difference"]["no_adapt"]["mean_pairwise_disagreement"], 0.0)
        # every terminal evaluation used the same holdout
        fps = {e["holdout_fingerprint"] for e in events if e["event"] == "terminal"}
        self.assertEqual(len(fps), 1)
        # every step logged belongs to a complete replay: steps per (method, order) equal ceil(N / B)
        n_steps = {}
        for e in events:
            if e["event"] == "step" and e["phase"] == "test_time":
                n_steps[(e["method"], e["order"])] = n_steps.get((e["method"], e["order"]), 0) + 1
        self.assertEqual(len(set(n_steps.values())), 1)
        # manifest completeness
        for key in ("git", "config_sha256", "data", "source_models", "subspaces", "holdout", "orders",
                    "environment", "wall_seconds", "peak_rss_mib", "peak_python_heap_mib", "raw_log_sha256"):
            self.assertIn(key, m)
        # meta-training cost is recorded separately from test-time cost
        totals = s["cost_ledger"]["totals"]
        self.assertGreater(totals["meta_training"]["hvp_evals"], 0)
        self.assertTrue(totals["meta_training"]["any_labels"])
        self.assertFalse(totals["test_time"]["any_labels"])
        self.assertIn(s["decision"]["label"], {
            "NO_ADDED_UTILITY_GAIN_BELOW_MIE", "NO_STABILIZATION_OBSERVED", "NO_ADDED_UTILITY_VS_CONTROLS",
            "ADDED_UTILITY_OBSERVED_ON_SAMPLED_ORDERS"})

    def test_same_config_is_deterministic(self):
        cfg = tiny_config()
        s1, _, _ = self._run(cfg, "d1")
        s2, _, _ = self._run(cfg, "d2")
        self.assertEqual(s1["replicates"][0]["errors"], s2["replicates"][0]["errors"])
        self.assertEqual(s1["decision"]["label"], s2["decision"]["label"])

    def test_tuning_and_drift_family_path(self):
        cfg = tiny_config()
        cfg["stream"]["order_family"] = "domain_blocked"
        cfg["tuning"] = {"enabled": True, "lr_grid": [0.1, 0.5], "order_seeds": [900]}
        cfg["subspaces"]["order_aware"].update({"normalized": True, "lam": 1.0})
        for m in cfg["methods"]:
            if m["name"] in ("tent_full", "tent_orderaware"):
                m["lr"] = "tune"
        s, _, events = self._run(cfg, "tuned")
        rep = s["replicates"][0]
        self.assertEqual(set(rep["tuned_lrs"]), {"tent_full", "tent_orderaware"})
        self.assertTrue(all(v in (0.1, 0.5) for v in rep["tuned_lrs"].values()))
        self.assertAlmostEqual(rep["lrs"]["tent_full_lr_x0.1"], 0.1 * rep["tuned_lrs"]["tent_full"])
        self.assertTrue(all(o["name"].startswith("blocked_") for o in rep["orders"]))
        self.assertGreater(s["cost_ledger"]["totals"]["tuning"]["gradient_evals"], 0)
        self.assertTrue(s["cost_ledger"]["totals"]["tuning"]["any_labels"])  # selection reads dev_holdout labels
        self.assertFalse(s["cost_ledger"]["totals"]["test_time"]["any_labels"])
        self.assertEqual(sum(1 for e in events if e["event"] == "tuning"), 4)

    def test_empty_candidate_subspace_is_reported_not_padded(self):
        cfg = tiny_config()
        cfg["subspaces"]["order_aware"]["lam"] = 1e12
        s, _, _ = self._run(cfg, "empty")
        rep = s["replicates"][0]
        self.assertEqual(rep["subspaces"]["order_aware"]["dim"], 0)
        self.assertEqual(rep["subspaces"]["random"]["dim"], 0)  # dimension-matched control follows
        self.assertEqual(s["decision"]["label"], "NO_ADDED_UTILITY_GAIN_BELOW_MIE")


class FailurePreservation(RunnerTestCase):
    def test_failed_calibration_marks_cells_not_run_and_decision_incomplete(self):
        cfg = tiny_config()
        for m in cfg["methods"]:
            if m.get("norm_match", {}).get("mode") == "calibrated_global":
                m["norm_match"]["lr_hi"] = 2e-4  # target displacement unreachable -> mismatch > tolerance
        s, _, events = self._run(cfg, "fail")
        self.assertEqual(s["method_status"]["norm_matched_global"], "NOT_RUN")
        cells = s["replicates"][0]["cells"]["norm_matched_global"]
        self.assertTrue(all(c["status"] == "NOT_RUN" and "mismatch" in c["reason"] for c in cells.values()))
        self.assertEqual(s["decision"]["label"], "INCOMPLETE")
        self.assertTrue(any(e["event"] == "cell_status" for e in events))
        self.assertNotIn("norm_matched_global", s["metrics"])  # non-OK arms are not summarised as if OK

    def test_failed_subspace_fit_only_blocks_dependent_arms(self):
        cfg = tiny_config()
        cfg["subspaces"]["order_aware"]["n_meta_batches"] = 1  # pairwise order terms need >= 2 batches
        s, _, events = self._run(cfg, "fitfail")
        self.assertEqual(s["software_status"], "RUN_COMPLETED")
        st = s["method_status"]
        for arm in ("tent_orderaware", "tent_gradpca", "tent_random", "norm_matched_per_step",
                    "norm_matched_global"):
            self.assertEqual(st[arm], "NOT_RUN", arm)
        for arm in ("no_adapt", "tent_full", "tent_full_lr_x0.1"):
            self.assertEqual(st[arm], "OK", arm)
        self.assertEqual(set(s["replicates"][0]["failed_subspaces"]), {"order_aware", "gradpca", "random"})
        self.assertTrue(any(e["event"] == "subspace_fit_failed" for e in events))
        self.assertEqual(s["decision"]["label"], "INCOMPLETE")

    def test_cap_exceeded_is_preserved(self):
        cfg = tiny_config()
        cfg["resource_cap"]["max_test_time_gradient_evals"] = 5
        s, _, _ = self._run(cfg, "cap")
        self.assertIn("CAP_EXCEEDED", s["method_status"].values())
        self.assertEqual(s["decision"]["label"], "INCOMPLETE")


class Validation(unittest.TestCase):
    def test_rejects_bad_configs(self):
        base = tiny_config()
        cases = []
        c = copy.deepcopy(base); c["run_type"] = "toy_pilot"; cases.append(c)  # noqa: E702 - no preregistration
        c = copy.deepcopy(base); c["methods"][1], c["methods"][2] = c["methods"][2], c["methods"][1]  # noqa: E702
        cases.append(c)  # forward reference
        c = copy.deepcopy(base); c["methods"].append(dict(c["methods"][1])); cases.append(c)  # noqa: E702
        c = copy.deepcopy(base); c["methods"][2]["lr_factor"] = 1.5; cases.append(c)  # noqa: E702
        c = copy.deepcopy(base); c["methods"][6]["subspace"] = "random"; cases.append(c)  # noqa: E702
        c = copy.deepcopy(base); c["stream"]["order_seeds"] = [1]; cases.append(c)  # noqa: E702
        c = copy.deepcopy(base); c["decision"]["candidate"] = "nope"; cases.append(c)  # noqa: E702
        for i, c in enumerate(cases):
            with self.assertRaises(ConfigError, msg=f"case {i}"):
                validate_config(c)

    def test_all_shipped_configs_validate_and_dry_run(self):
        paths = sorted(glob.glob(os.path.join(ROOT, "configs", "*.json")))
        self.assertGreaterEqual(len(paths), 1)
        for p in paths:
            plan = dry_run(p)
            self.assertTrue(plan["replicates"], p)
            for rep in plan["replicates"]:
                self.assertGreaterEqual(len(rep["orders"]), 2)


if __name__ == "__main__":
    unittest.main()

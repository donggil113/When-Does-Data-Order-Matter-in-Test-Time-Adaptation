"""Stage driver: run several condition configs under one shared CPU budget and apply the v2 stage rule.

``PYTHONPATH=src python3 -m ordertta.stage --stage configs/stage2_order_penalty_diagnostic.json``

Limits (set before any work, recorded in the stage manifest):
  * one process, one thread (pure-Python code; thread count is checked at start and end);
  * RLIMIT_AS = ``budget.max_address_space_bytes`` (address space bounds resident memory);
  * RLIMIT_CPU = total CPU budget + small grace as a hard backstop, plus an in-process CPU deadline
    (budget minus a reserve for writing outputs) checked before every batch, including tuning and
    calibration. Cells that hit the deadline are recorded as CAP_EXCEEDED, never dropped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import threading
import time
from typing import Dict, Optional

from .analysis import decide_stage
from .manifest import canonical_sha256, environment, file_sha256, git_info, utc_now
from .runner import REPO_ROOT, ConfigError, cpu_seconds_used, run, validate_config

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
SHARED_KEYS = ("world", "replicate_seeds", "model", "subspaces", "prequential")


def load_stage(path: str) -> dict:
    with open(path) as f:
        stage = json.load(f)
    roles = [c["role"] for c in stage["conditions"]]
    if roles.count("primary_stationary") != 1 or roles.count("primary_drift") != 1:
        raise ConfigError("a stage needs exactly one primary_stationary and one primary_drift condition")
    cfgs = {}
    for c in stage["conditions"]:
        with open(os.path.join(REPO_ROOT, c["config"])) as f:
            cfg = json.load(f)
        validate_config(cfg)
        if cfg["decision"].get("mode") != "seedwise_v2":
            raise ConfigError(f"{c['config']}: stage conditions must use decision.mode = seedwise_v2")
        cfgs[c["name"]] = cfg
    first = next(iter(cfgs.values()))
    for name, cfg in cfgs.items():
        for k in SHARED_KEYS:
            if cfg.get(k) != first.get(k):
                raise ConfigError(f"condition {name} differs from the others in {k!r}")
        if [m["name"] for m in cfg["methods"]] != [m["name"] for m in first["methods"]]:
            raise ConfigError(f"condition {name} has a different arm list")
        if cfg.get("tuning", {}).get("lr_grid") != first.get("tuning", {}).get("lr_grid"):
            raise ConfigError(f"condition {name} has a different lr grid")
    rule = stage["decision"]
    arms = {m["name"] for m in first["methods"]}
    for a in [rule["candidate"], rule["no_adapt"], rule["baseline"], rule["drift_reference"],
              *rule["required_controls"]]:
        if a not in arms:
            raise ConfigError(f"decision arm {a!r} is not in the condition configs")
    if len(first["replicate_seeds"]) != rule["n_seeds"]:
        raise ConfigError("decision.n_seeds must equal the number of replicate seeds")
    return {"stage": stage, "configs": cfgs}


def apply_limits(budget: dict) -> dict:
    for v in THREAD_ENV:
        os.environ[v] = str(budget.get("threads", 1))
    as_bytes = int(budget["max_address_space_bytes"])
    resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))
    used = cpu_seconds_used()
    soft = int(math.ceil(budget["max_cpu_seconds_total"] + budget.get("rlimit_grace_seconds", 5)))
    resource.setrlimit(resource.RLIMIT_CPU, (soft, soft + 5))
    return {"RLIMIT_AS": resource.getrlimit(resource.RLIMIT_AS), "RLIMIT_CPU": resource.getrlimit(resource.RLIMIT_CPU),
            "cpu_seconds_before_limits": used, "thread_env": {v: os.environ[v] for v in THREAD_ENV},
            "threads_at_start": threading.active_count()}


def run_stage(stage_path: str, out_root: str, run_id: Optional[str] = None, set_limits: bool = True) -> str:
    loaded = load_stage(stage_path)
    stage, cfgs = loaded["stage"], loaded["configs"]
    budget = stage["budget"]
    if budget.get("workers", 1) != 1 or budget.get("threads", 1) != 1:
        raise ConfigError("this stage runs with exactly one worker and one thread")
    git_before = git_info(REPO_ROOT)
    run_id = run_id or f"{stage['stage']}_{time.strftime('%Y%m%dT%H%M%S')}"
    out_dir = os.path.join(out_root, run_id)
    os.makedirs(out_dir, exist_ok=False)
    limits = apply_limits(budget) if set_limits else {"note": "limits not applied (test mode)"}
    cpu0, wall0, utc0 = cpu_seconds_used(), time.perf_counter(), utc_now()
    deadline = cpu0 + budget["max_cpu_seconds_total"] - budget.get("reserve_cpu_seconds", 0)
    cond_out: Dict[str, dict] = {}
    for c in stage["conditions"]:
        path = os.path.join(REPO_ROOT, c["config"])
        rec = {"role": c["role"], "config": c["config"], "config_sha256": canonical_sha256(cfgs[c["name"]]),
               "config_file_sha256": file_sha256(path)}
        if cpu_seconds_used() > deadline:
            rec.update({"status": "CAP_EXCEEDED", "reason": "CPU budget exhausted before this condition started"})
        else:
            t = cpu_seconds_used()
            d = run(path, out_dir, c["name"], cpu_deadline=deadline)
            with open(os.path.join(d, "summary.json")) as f:
                summ = json.load(f)
            rec.update({"status": summ["software_status"], "out_dir": os.path.relpath(d, REPO_ROOT),
                        "cpu_seconds": cpu_seconds_used() - t, "method_status": summ["method_status"],
                        "summary": summ})
        cond_out[c["name"]] = rec

    rule = stage["decision"]
    by_role = {c["role"]: c["name"] for c in stage["conditions"]}

    def seedwise(role: str, metric: str) -> Optional[dict]:
        rec = cond_out[by_role[role]]
        if "summary" not in rec:
            return None
        return rec["summary"]["decision"].get("seedwise", {}).get(metric)

    statuses = {by_role[r]: cond_out[by_role[r]].get("method_status", {}) for r in ("primary_stationary",
                                                                                     "primary_drift")}
    primary = {c["role"]: c for c in stage["conditions"]}
    decision = decide_stage(seedwise("primary_stationary", primary["primary_stationary"]["primary_metric"]),
                            seedwise("primary_drift", primary["primary_drift"]["primary_metric"]),
                            statuses, dict(rule, drift_online=seedwise("primary_drift", "online_error")))
    cpu_total = cpu_seconds_used() - cpu0
    wall = time.perf_counter() - wall0
    self_ru, child_ru = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    report = {
        "stage": stage["stage"], "run_id": run_id, "software_status": (
            "STAGE_COMPLETED" if all(r["status"] == "RUN_COMPLETED" for r in cond_out.values()) else "STAGE_PARTIAL"),
        "science_status": "TOY_DIAGNOSTIC_ONLY_NOT_EVIDENCE_ABOUT_REAL_TTA",
        "decision": decision,
        "conditions": {k: {kk: vv for kk, vv in v.items() if kk != "summary"} for k, v in cond_out.items()},
        "secondary_seedwise": {c["name"]: (cond_out[c["name"]].get("summary") or {}).get("decision", {})
                               .get("seedwise") for c in stage["conditions"] if c["role"] == "secondary"},
    }
    with open(os.path.join(out_dir, "stage_summary.json"), "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    manifest = {
        "run_id": run_id, "stage_config": stage_path, "stage_config_sha256": canonical_sha256(stage),
        "stage_config_file_sha256": file_sha256(stage_path), "git": git_before, "utc_start": utc0,
        "utc_end": utc_now(), "wall_seconds": wall, "cpu_seconds_total_incl_children": cpu_total,
        "cpu_budget_seconds": budget["max_cpu_seconds_total"],
        "cpu_deadline_reserve": budget.get("reserve_cpu_seconds"),
        "peak_rss_mib_self": self_ru.ru_maxrss / 1024.0, "peak_rss_mib_children": child_ru.ru_maxrss / 1024.0,
        "limits": limits, "threads_at_end": threading.active_count(), "workers": 1,
        "install_and_download_cost": "none: no packages installed, no models or data downloaded in this stage",
        "conditions": {k: {kk: v.get(kk) for kk in ("config", "config_sha256", "config_file_sha256", "status",
                                                    "out_dir", "cpu_seconds")} for k, v in cond_out.items()},
        "environment": environment(), "argv": sys.argv,
        "stage_summary_sha256": file_sha256(os.path.join(out_dir, "stage_summary.json")),
    }
    with open(os.path.join(out_dir, "stage_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    return out_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "runs"))
    ap.add_argument("--run-id")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args(argv)
    if args.validate_only:
        loaded = load_stage(args.stage)
        print(json.dumps({"stage": loaded["stage"]["stage"], "conditions": list(loaded["configs"])}, indent=1))
        return 0
    out = run_stage(args.stage, args.out, args.run_id)
    with open(os.path.join(out, "stage_summary.json")) as f:
        s = json.load(f)
    print(json.dumps({"out_dir": out, "software_status": s["software_status"], "science_status": s["science_status"],
                      "decision": s["decision"]["label"], "reasons": s["decision"].get("reasons"),
                      "conditions": {k: v["status"] for k, v in s["conditions"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Compare two runs that replay the same multisets under different order families.

``PYTHONPATH=src python -m ordertta.compare --a runs/<stationary>/summary.json --b runs/<drift>/summary.json``
prints, per method, the pooled across-order spread and mean error of b minus a with bootstrap CIs.
"""

import argparse
import json
import sys

from .analysis import compare_order_families


def compare(path_a: str, path_b: str, metric: str, n_boot: int, seed: int) -> dict:
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)
    fa = {r["replicate"]: r["world_fingerprint"] for r in a["replicates"]}
    fb = {r["replicate"]: r["world_fingerprint"] for r in b["replicates"]}
    if fa != fb:
        raise SystemExit("runs do not share the same replicates/worlds; family comparison is not valid")
    ea = {r["replicate"]: r["errors"] for r in a["replicates"]}
    eb = {r["replicate"]: r["errors"] for r in b["replicates"]}
    ok = sorted(m for m, s in a["method_status"].items() if s == "OK" and b["method_status"].get(m) == "OK")
    out = {"a": a["run_id"], "b": b["run_id"], "metric": metric, "science_status": b["science_status"],
           "methods": {}}
    for m in ok:
        out["methods"][m] = {"spread_b_minus_a": compare_order_families(ea, eb, m, metric, n_boot, seed),
                             "mean_error_b_minus_a": compare_order_families(ea, eb, m, "mean_error", n_boot, seed)}
    out["not_compared"] = sorted(set(a["method_status"]) - set(ok))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--metric", default="std_terminal_error")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    print(json.dumps(compare(args.a, args.b, args.metric, args.n_boot, args.seed), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

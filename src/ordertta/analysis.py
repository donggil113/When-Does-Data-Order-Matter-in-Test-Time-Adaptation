"""Across-order summaries, paired stratified bootstrap, and the pre-registered decision rule.

Vocabulary: statistics over the *observed* orders only. ``observed_worst_error`` is the worst of the
K orders that were actually run; it is not a bound on the risk over all orders.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Dict, List, Mapping, Optional, Sequence

# errors[replicate][method][order_index] -> terminal holdout error
Errors = Mapping[str, Mapping[str, Sequence[float]]]


def nearest_rank_quantile(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    k = max(1, math.ceil(q * len(s)))
    return s[k - 1]


def order_stats(errors: Sequence[float], no_adapt: Sequence[float]) -> Dict[str, float]:
    k = len(errors)
    gains = [b - e for b, e in zip(no_adapt, errors)]
    return {
        "n_orders": k,
        "mean_error": statistics.fmean(errors),
        "std_error": statistics.stdev(errors) if k > 1 else 0.0,
        "var_error": statistics.variance(errors) if k > 1 else 0.0,
        "min_error": min(errors),
        "observed_worst_error": max(errors),
        "range_error": max(errors) - min(errors),
        "q90_error_nearest_rank": nearest_rank_quantile(errors, 0.9),
        "mean_gain": statistics.fmean(gains),
        "observed_worst_gain": min(gains),
    }


def spread(xs: Sequence[float], metric: str) -> float:
    if metric == "std_terminal_error":
        return statistics.stdev(xs) if len(xs) > 1 else 0.0
    if metric == "range_terminal_error":
        return max(xs) - min(xs)
    if metric == "observed_worst_minus_mean":
        return max(xs) - statistics.fmean(xs)
    raise ValueError(f"unknown spread metric {metric!r}")


def pooled_metric(errors: Errors, method: str, no_adapt: str, metric: str,
                  idx: Optional[Mapping[str, Sequence[int]]] = None) -> float:
    """Mean over replicates of a per-replicate statistic ('gain' or a spread metric)."""
    vals = []
    for rep, per_m in errors.items():
        ii = idx[rep] if idx is not None else range(len(per_m[method]))
        e = [per_m[method][i] for i in ii]
        if metric == "gain":
            b = [per_m[no_adapt][i] for i in ii]
            vals.append(statistics.fmean(bi - ei for bi, ei in zip(b, e)))
        elif metric == "mean_error":
            vals.append(statistics.fmean(e))
        else:
            vals.append(spread(e, metric))
    return statistics.fmean(vals)


def paired_bootstrap_ci(errors: Errors, method_a: str, method_b: str, no_adapt: str, metric: str,
                        n_boot: int, seed: int, alpha: float = 0.05) -> Dict[str, float]:
    """CI for pooled_metric(a) - pooled_metric(b); orders resampled with replacement within each
    replicate, with the same indices for both methods (paired)."""
    rng = random.Random(seed)
    point = (pooled_metric(errors, method_a, no_adapt, metric)
             - pooled_metric(errors, method_b, no_adapt, metric))
    draws = []
    for _ in range(n_boot):
        idx = {rep: [rng.randrange(len(per_m[method_a])) for _ in per_m[method_a]]
               for rep, per_m in errors.items()}
        draws.append(pooled_metric(errors, method_a, no_adapt, metric, idx)
                     - pooled_metric(errors, method_b, no_adapt, metric, idx))
    draws.sort()
    lo = draws[int(math.floor(alpha / 2 * n_boot))]
    hi = draws[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "n_boot": n_boot, "alpha": alpha}


def decide(errors: Errors, statuses: Mapping[str, str], rule: dict) -> Dict[str, object]:
    """Pre-registered rule; returns a label plus the evidence used.

    Labels:
      INCOMPLETE                            some required (method, order) run is not OK;
      NO_ADDED_UTILITY_GAIN_BELOW_MIE        candidate's gain over no-adaptation < MIE (stable only by not adapting);
      NO_STABILIZATION_OBSERVED              candidate's spread is not below the full-lr baseline by MIE;
      NO_ADDED_UTILITY_VS_CONTROLS           some small-step / norm-matched / dimension control is not
                                            dominated (stabilisation explained by smaller or fewer moves);
      ADDED_UTILITY_OBSERVED_ON_SAMPLED_ORDERS  dominated every control on the sampled orders only.
    """
    c, base, na = rule["candidate"], rule["baseline"], rule["no_adapt"]
    controls = list(rule.get("stability_controls", [])) + list(rule.get("dimension_controls", []))
    sm = rule["spread_metric"]
    mie_s, mie_g = rule["mie"]["spread_abs"], rule["mie"]["gain_abs"]
    tol_s, tol_g = rule["tolerance"]["spread_abs"], rule["tolerance"]["gain_abs"]
    nb, seed = rule.get("n_boot", 2000), rule.get("boot_seed", 0)
    required = [c, base, na] + controls
    bad = {m: s for m, s in statuses.items() if m in required and s != "OK"}
    missing = [m for m in required if m not in statuses]
    if bad or missing:
        return {"label": "INCOMPLETE", "non_ok": bad, "missing": missing}
    ev: Dict[str, object] = {}
    g_c = paired_bootstrap_ci(errors, c, na, na, "gain", nb, seed)  # gain(c) - gain(no_adapt)=gain(c)
    ev["candidate_gain"] = g_c
    if g_c["point"] < mie_g or g_c["ci_lo"] <= 0:
        return {"label": "NO_ADDED_UTILITY_GAIN_BELOW_MIE", "evidence": ev}
    s_cb = paired_bootstrap_ci(errors, base, c, na, sm, nb, seed + 1)  # spread(base) - spread(c)
    ev["spread_reduction_vs_baseline"] = s_cb
    if s_cb["point"] < mie_s or s_cb["ci_lo"] <= 0:
        return {"label": "NO_STABILIZATION_OBSERVED", "evidence": ev}
    undominated = []
    per_control = {}
    for i, k in enumerate(controls):
        ds = paired_bootstrap_ci(errors, k, c, na, sm, nb, seed + 10 + 2 * i)  # spread(k) - spread(c)
        dg = paired_bootstrap_ci(errors, c, k, na, "gain", nb, seed + 11 + 2 * i)  # gain(c) - gain(k)
        better_spread = ds["point"] >= mie_s and ds["ci_lo"] > 0 and dg["point"] >= -tol_g
        better_gain = dg["point"] >= mie_g and dg["ci_lo"] > 0 and ds["point"] >= -tol_s
        per_control[k] = {"spread_k_minus_c": ds, "gain_c_minus_k": dg,
                          "candidate_more_stable_at_no_gain_loss": better_spread,
                          "candidate_more_gain_at_no_stability_loss": better_gain}
        if not (better_spread or better_gain):
            undominated.append(k)
    ev["controls"] = per_control
    if undominated:
        return {"label": "NO_ADDED_UTILITY_VS_CONTROLS", "undominated_controls": undominated, "evidence": ev}
    return {"label": "ADDED_UTILITY_OBSERVED_ON_SAMPLED_ORDERS", "evidence": ev,
            "caveat": "holds for the sampled orders/replicates only; not a guarantee over all orders"}


def compare_order_families(errors_a: Errors, errors_b: Errors, method: str, metric: str, n_boot: int,
                           seed: int, alpha: float = 0.05) -> Dict[str, float]:
    """pooled spread/mean of ``method`` under family b minus family a (e.g. drift minus stationary).

    The two families replay the same multiset per replicate but with different orders, so orders are
    resampled independently for a and b within each replicate (stratified, unpaired bootstrap).
    """
    if set(errors_a) != set(errors_b):
        raise ValueError("families must cover the same replicates")

    def stat(err: Errors, idx) -> float:
        vals = []
        for rep in sorted(err):
            e = [err[rep][method][i] for i in idx[rep]] if idx else list(err[rep][method])
            vals.append(statistics.fmean(e) if metric == "mean_error" else spread(e, metric))
        return statistics.fmean(vals)

    point = stat(errors_b, None) - stat(errors_a, None)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        ia = {r: [rng.randrange(len(errors_a[r][method])) for _ in errors_a[r][method]] for r in errors_a}
        ib = {r: [rng.randrange(len(errors_b[r][method])) for _ in errors_b[r][method]] for r in errors_b}
        draws.append(stat(errors_b, ib) - stat(errors_a, ia))
    draws.sort()
    lo = draws[int(math.floor(alpha / 2 * n_boot))]
    hi = draws[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "n_boot": n_boot, "alpha": alpha}


# ------------------------------------------------------------------ seedwise (v2) summaries and stage rule
#
# Orders are nested in seeds (worlds); they are not independent replicates. v2 therefore reports, per seed,
# statistics over that seed's observed orders and paired (same seed, same order) differences, and then only
# the list of per-seed values, their mean and their sign count across seeds. No confidence interval is formed
# by pooling seeds and orders.

def _std(xs: Sequence[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def seedwise_condition_summary(table: Errors, arms: Sequence[str], candidate: str, no_adapt: str,
                               controls: Sequence[str]) -> Dict[str, object]:
    """``table[seed][arm]`` = metric per order (same order list for every arm within a seed; lower is better)."""
    seeds = sorted(table)
    per_arm: Dict[str, object] = {}
    for a in arms:
        rows = {}
        for sd in seeds:
            v = table[sd].get(a)
            if not v:
                continue
            na = table[sd].get(no_adapt)
            rows[sd] = {"mean": statistics.fmean(v), "std_over_orders": _std(v), "observed_max": max(v),
                        "observed_min": min(v), "n_orders": len(v),
                        "mean_gain_vs_no_adapt": (statistics.fmean(n - x for n, x in zip(na, v))
                                                  if na and len(na) == len(v) else None)}
        per_arm[a] = {"per_seed": rows, "seed_mean": {k: statistics.fmean(r[k] for r in rows.values())
                                                      for k in ("mean", "std_over_orders", "observed_max")}
                      if rows else None}
    paired: Dict[str, object] = {}
    for k in controls:
        rows = {}
        for sd in seeds:
            c, o = table[sd].get(candidate), table[sd].get(k)
            if not c or not o or len(c) != len(o):
                continue
            rows[sd] = {"mean_diff_c_minus_k": statistics.fmean(x - y for x, y in zip(c, o)),
                        "std_diff_c_minus_k": _std(c) - _std(o),
                        "max_diff_c_minus_k": max(c) - max(o)}
        paired[k] = {"per_seed": rows, "n_seeds": len(rows),
                     "seed_mean": ({m: statistics.fmean(r[m] for r in rows.values())
                                    for m in ("mean_diff_c_minus_k", "std_diff_c_minus_k", "max_diff_c_minus_k")}
                                   if rows else None)}
    return {"seeds": seeds, "per_arm": per_arm, "paired_vs_candidate": paired,
            "note": "no pooled CI: orders are nested in seeds; see per-seed values"}


def _dominates(pair: dict, mie: dict, tol: dict, n_seeds: int) -> Dict[str, object]:
    """Candidate better than control on one axis by MIE, consistently in every seed, without losing more than
    the tolerance on the other axis (seed means; lower error / lower std is better)."""
    rows = pair["per_seed"]
    if pair["n_seeds"] != n_seeds or not rows:
        return {"dominates": False, "complete": False}
    m = pair["seed_mean"]
    all_mean_better = all(r["mean_diff_c_minus_k"] < 0 for r in rows.values())
    all_std_better = all(r["std_diff_c_minus_k"] < 0 for r in rows.values())
    by_mean = -m["mean_diff_c_minus_k"] >= mie["gain_abs"] and all_mean_better and \
        m["std_diff_c_minus_k"] <= tol["spread_abs"]
    by_spread = -m["std_diff_c_minus_k"] >= mie["spread_abs"] and all_std_better and \
        m["mean_diff_c_minus_k"] <= tol["gain_abs"]
    return {"dominates": bool(by_mean or by_spread), "complete": True, "by_mean_error": by_mean,
            "by_order_spread": by_spread, "seed_mean": m}


def decide_stage(stationary: Optional[dict], drift: Optional[dict], statuses: Mapping[str, Mapping[str, str]],
                 rule: dict) -> Dict[str, object]:
    """Pre-registered v2 rule for the order-penalty question.

    ``stationary`` / ``drift``: outputs of ``seedwise_condition_summary`` on the primary metric of each
    primary condition (stationary: terminal common-holdout error; drift: regime-end current-regime holdout
    error, with online error checked separately via ``rule['drift_online']``). ``statuses[condition][arm]``.
    """
    c, na, base = rule["candidate"], rule["no_adapt"], rule["baseline"]
    controls = list(rule["required_controls"])
    mie, tol, n_seeds = rule["mie"], rule["tolerance"], rule["n_seeds"]
    required = [c, na, base] + controls
    missing = {cond: sorted(a for a in required if st.get(a) != "OK") for cond, st in statuses.items()}
    missing = {k: v for k, v in missing.items() if v}
    if missing or stationary is None or drift is None:
        return {"label": "INCOMPLETE", "non_ok_required_arms": missing}
    reasons: List[str] = []
    ev: Dict[str, object] = {}
    cand = stationary["per_arm"][c]["per_seed"]
    gains = [r["mean_gain_vs_no_adapt"] for r in cand.values()]
    ev["stationary_gain_vs_no_adapt_per_seed"] = gains
    if len(gains) != n_seeds or statistics.fmean(gains) < mie["gain_abs"] or min(gains) <= 0:
        reasons.append("ADAPTATION_GAIN_LOST_OR_BELOW_MIE")
    base_pair = stationary["paired_vs_candidate"].get(base)
    if base_pair is None or base_pair["n_seeds"] != n_seeds:
        return {"label": "INCOMPLETE", "reason": f"paired rows vs {base} missing"}
    std_red = [-r["std_diff_c_minus_k"] for r in base_pair["per_seed"].values()]
    ev["stationary_std_reduction_vs_baseline_per_seed"] = std_red
    if statistics.fmean(std_red) < mie["spread_abs"] or min(std_red) <= 0:
        reasons.append("NO_ORDER_ROBUSTNESS_GAIN_VS_BASELINE")
    dom = {}
    for k in controls:
        d = _dominates(stationary["paired_vs_candidate"][k], mie, tol, n_seeds)
        dom[k] = d
        if not d["complete"]:
            return {"label": "INCOMPLETE", "reason": f"paired rows vs {k} missing"}
        if not d["dominates"]:
            reasons.append(f"NOT_BETTER_THAN_{k}")
    ev["stationary_domination"] = dom
    guard = {}
    for metric_name, summ in (("regime_end_error", drift), ("online_error", rule.get("drift_online"))):
        if summ is None:
            return {"label": "INCOMPLETE", "reason": f"drift {metric_name} summary missing"}
        ref = summ["paired_vs_candidate"].get(rule["drift_reference"])
        if ref is None or ref["n_seeds"] != n_seeds:
            return {"label": "INCOMPLETE", "reason": f"drift {metric_name} rows vs reference missing"}
        dmg = ref["seed_mean"]["mean_diff_c_minus_k"]
        gain = statistics.fmean(r["mean_gain_vs_no_adapt"] for r in summ["per_arm"][c]["per_seed"].values())
        guard[metric_name] = {"candidate_minus_reference_mean": dmg, "candidate_gain_vs_no_adapt": gain,
                              "within_tolerance": dmg <= tol["gain_abs"] and gain >= 0}
        if not guard[metric_name]["within_tolerance"]:
            reasons.append(f"DRIFT_DAMAGE_{metric_name.upper()}")
    ev["drift_guard"] = guard
    if reasons:
        return {"label": "ORDER_PENALTY_ADDED_UTILITY_NOT_SUPPORTED", "reasons": reasons, "evidence": ev,
                "scope": "this toy diagnostic only; non-support is not evidence of equivalence"}
    return {"label": "REALDATA_PILOT_CANDIDATE", "evidence": ev,
            "scope": "toy diagnostic on 3 seeds x sampled orders; not a performance, robustness or novelty claim"}


def summarize(errors: Errors, no_adapt: str) -> Dict[str, Dict[str, object]]:
    methods = sorted({m for per_m in errors.values() for m in per_m})
    out: Dict[str, Dict[str, object]] = {}
    for m in methods:
        per_rep = {rep: order_stats(per_m[m], per_m[no_adapt]) for rep, per_m in errors.items() if m in per_m}
        keys = next(iter(per_rep.values())).keys()
        pooled = {k: statistics.fmean(v[k] for v in per_rep.values()) for k in keys}
        out[m] = {"pooled_mean_over_replicates": pooled, "per_replicate": per_rep}
    return out


__all__ = ["order_stats", "spread", "pooled_metric", "paired_bootstrap_ci", "decide", "summarize",
           "nearest_rank_quantile", "compare_order_families", "seedwise_condition_summary", "decide_stage"]

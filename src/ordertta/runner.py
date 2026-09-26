"""CPU runner: replay one multiset in K fixed orders for every arm, score on a common terminal holdout,
and write raw_log.jsonl, summary.json and manifest.json.

Every (replicate, method, order) cell ends in exactly one status: OK, FAILED, CAP_EXCEEDED or NOT_RUN.
Non-OK cells are preserved in the outputs and make the decision INCOMPLETE; nothing is silently skipped.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
import tracemalloc
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from . import linalg as la
from .analysis import decide, seedwise_condition_summary, summarize
from .evaluator import (RegimeHoldoutEvaluator, TerminalHoldoutEvaluator, TerminalResult, class_distribution,
                        pairwise_output_difference)
from .manifest import canonical_sha256, environment, file_sha256, git_info, peak_rss_mib, utc_now
from .methods import (Adapter, CausalNormReference, EntropySGD, NoAdapt, calibrate_lr_to_displacement,
                      per_step_norm_schedule, small_lr)
from .replay import (IntegrityError, LabelVault, OrderSpec, PrequentialLabelOracle, ReplayLoader, Sample,
                     assert_disjoint, make_orders, sha256_json)
from .subspace import (AdaptationSubspace, CostLedger, CostRecord, MetaBatchStats, complete_basis,
                       full_space, gradient_pca_subspace, order_aware_subspace, random_subspace,
                       top_eigenvectors, utility_only_subspace)
from .toy import LinearTTAModel, SyntheticWorld, build_world, train_source_model

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_SPLITS = ("test_stream", "test_holdout")
PREREG_KEYS = ("primary_metric", "spread_metric", "mie", "seeds", "tuning_range", "resource_cap", "splits")
ORDER_FAMILIES = ("uniform_permutation", "domain_blocked", "batch_permutation", "domain_blocked_batches")
FIXED_BATCH_FAMILIES = ("batch_permutation", "domain_blocked_batches")
DRIFT_FAMILIES = ("domain_blocked", "domain_blocked_batches")
TUNING_METRICS = ("mean_terminal_error", "mean_regime_end_error")


class ConfigError(ValueError):
    pass


class CapExceeded(RuntimeError):
    pass


# ------------------------------------------------------------------------------------ validation

def validate_config(cfg: dict) -> List[str]:
    """Raise ConfigError on structural problems; return a list of non-fatal warnings."""
    warnings: List[str] = []
    for key in ("run_type", "world", "model", "stream", "methods", "decision", "resource_cap"):
        if key not in cfg:
            raise ConfigError(f"missing top-level key {key!r}")
    if cfg["run_type"] not in ("smoke", "toy_pilot", "toy_diagnostic"):
        raise ConfigError("run_type must be 'smoke', 'toy_pilot' or 'toy_diagnostic'")
    if cfg["run_type"] != "smoke":
        pre = cfg.get("preregistration")
        if not pre:
            raise ConfigError("pilot configs must carry a 'preregistration' block fixed before running")
        for k in PREREG_KEYS:
            if k not in pre:
                raise ConfigError(f"preregistration missing {k!r}")
        if pre["mie"] != cfg["decision"]["mie"]:
            raise ConfigError("preregistration.mie and decision.mie differ")
    cap = cfg["resource_cap"]
    if "max_wall_seconds" not in cap and "max_cpu_seconds" not in cap:
        raise ConfigError("resource_cap needs max_cpu_seconds and/or max_wall_seconds")
        if pre["spread_metric"] != cfg["decision"]["spread_metric"]:
            raise ConfigError("preregistration.spread_metric and decision.spread_metric differ")
    names: List[str] = []
    subspaces = cfg.get("subspaces", {})
    for m in cfg["methods"]:
        n = m["name"]
        if n in names:
            raise ConfigError(f"duplicate method name {n!r}")
        kind = m["kind"]
        if kind not in ("no_adapt", "entropy_sgd"):
            raise ConfigError(f"{n}: unknown kind {kind!r}")
        if kind == "entropy_sgd":
            sources = [k for k in ("lr", "lr_from", "norm_match") if k in m]
            if len(sources) != 1:
                raise ConfigError(f"{n}: exactly one of lr / lr_from / norm_match is required")
            if m.get("lr") == "tune" and not cfg.get("tuning", {}).get("enabled"):
                raise ConfigError(f"{n}: lr='tune' but tuning is not enabled")
            for ref in (m.get("lr_from"), (m.get("norm_match") or {}).get("target")):
                if ref is not None and ref not in names:
                    raise ConfigError(f"{n}: reference {ref!r} must be defined earlier in 'methods'")
            if m.get("subspace") is not None and m["subspace"] not in subspaces:
                raise ConfigError(f"{n}: unknown subspace {m['subspace']!r}")
            if "lr_factor" in m and not 0 < m["lr_factor"] < 1:
                raise ConfigError(f"{n}: lr_factor must be in (0, 1)")
            if m.get("norm_match") and m.get("subspace") is not None:
                raise ConfigError(f"{n}: norm-matched controls adapt in the full space")
            mode = (m.get("norm_match") or {}).get("mode")
            if mode is not None and mode not in ("per_step", "per_step_causal", "calibrated_global"):
                raise ConfigError(f"{n}: unknown norm_match mode {mode!r}")
            if mode == "per_step" and cfg["run_type"] == "toy_diagnostic":
                raise ConfigError(f"{n}: diagnostic configs must use per_step_causal (no precomputed trace)")
        names.append(n)
    dec = cfg["decision"]
    if dec.get("mode", "pooled_bootstrap_v1") not in ("pooled_bootstrap_v1", "seedwise_v2"):
        raise ConfigError("decision.mode must be pooled_bootstrap_v1 or seedwise_v2")
    for key in ("candidate", "baseline", "no_adapt"):
        if dec[key] not in names:
            raise ConfigError(f"decision.{key}={dec[key]!r} is not a method")
    for k in dec.get("stability_controls", []) + dec.get("dimension_controls", []):
        if k not in names:
            raise ConfigError(f"decision control {k!r} is not a method")
    tcfg = cfg.get("tuning", {})
    st = cfg["stream"]
    if tcfg.get("enabled"):
        for key, allowed in (("split", "development"), ("holdout", "dev_holdout")):
            if tcfg.get(key, allowed) != allowed:
                raise ConfigError(f"tuning.{key} must be {allowed!r} (the only supported value)")
        if tcfg.get("metric", "mean_terminal_error") not in TUNING_METRICS:
            raise ConfigError(f"tuning.metric must be one of {TUNING_METRICS}")
        if tcfg.get("metric") == "mean_regime_end_error" and st["order_family"] not in DRIFT_FAMILIES:
            raise ConfigError("mean_regime_end_error tuning needs a drift (domain-blocked) order family")
        if not tcfg.get("lr_grid") or not tcfg.get("order_seeds"):
            raise ConfigError("tuning needs lr_grid and order_seeds")
    if st["order_family"] not in ORDER_FAMILIES:
        raise ConfigError(f"stream.order_family must be one of {ORDER_FAMILIES}")
    if st["order_family"] in FIXED_BATCH_FAMILIES and "partition_seed" not in st:
        raise ConfigError(f"{st['order_family']} needs stream.partition_seed")
    if st.get("regime_eval") and st["order_family"] != "domain_blocked_batches":
        raise ConfigError("regime_eval needs single-domain fixed batches (domain_blocked_batches)")
    for name, sc in cfg.get("subspaces", {}).items():
        if sc.get("fitter") == "utility_only":
            ref = cfg["subspaces"].get(sc.get("of"), {})
            if ref.get("fitter") != "order_aware" or list(cfg["subspaces"]).index(sc["of"]) > \
                    list(cfg["subspaces"]).index(name):
                raise ConfigError(f"subspace {name}: 'of' must name an order_aware subspace defined earlier")
    if len(st["order_seeds"]) < 2:
        raise ConfigError("need at least two orders")
    other = set(cfg.get("tuning", {}).get("order_seeds", []))
    for m in cfg["methods"]:
        other |= set((m.get("norm_match") or {}).get("calibration_order_seeds", []))
    if other & set(st["order_seeds"]):
        warnings.append("test order seeds reused on development/calibration splits (different multisets)")
    return warnings


# --------------------------------------------------------------------------------------- logging

class RawLog:
    def __init__(self, path: Optional[str]):
        self.path = path
        self._fh = open(path, "w") if path else None
        self.n_events = 0

    def write(self, **event) -> None:
        self.n_events += 1
        if self._fh:
            self._fh.write(json.dumps(event, sort_keys=True, default=_json_default) + "\n")

    def close(self) -> None:
        if self._fh:
            self._fh.close()


def _json_default(o):
    if isinstance(o, float):
        return repr(o)
    raise TypeError(type(o))


# ------------------------------------------------------------------------------------- replicate

@dataclass
class Context:
    cfg: dict
    world: SyntheticWorld
    model: LinearTTAModel
    ledger: CostLedger
    log: RawLog
    replicate: str
    started: float
    test_time_grad_evals: int = 0
    cpu_deadline: Optional[float] = None  # absolute value of cpu_seconds_used() at which to stop


def cpu_seconds_used() -> float:
    """CPU time (user + system) of this process and its reaped children."""
    import resource
    ch = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + ch.ru_utime + ch.ru_stime


def _samples(world: SyntheticWorld, split: str) -> Dict[str, Sample]:
    return {s.id: s for s in world.samples(split)}


def _orders_for(cfg: dict, world: SyntheticWorld, split: str, seeds: Sequence[int]) -> List[OrderSpec]:
    st = cfg["stream"]
    samples = world.samples(split)
    return make_orders([s.id for s in samples], st["order_family"], seeds, {s.id: s.domain for s in samples},
                       st.get("domain_sequences"), st["batch_size"], st.get("partition_seed"))


def _check_cap(ctx: Context, phase: str) -> None:
    """CPU cap (run or stage budget) and wall cap: every phase. Gradient-eval cap: test time, per replicate."""
    cap = ctx.cfg["resource_cap"]
    if ctx.cpu_deadline is not None and cpu_seconds_used() > ctx.cpu_deadline:
        raise CapExceeded("CPU-time budget exhausted")
    if "max_wall_seconds" in cap and time.perf_counter() - ctx.started > cap["max_wall_seconds"]:
        raise CapExceeded(f"wall-clock cap {cap['max_wall_seconds']}s exceeded")
    if phase == "test_time" and ctx.test_time_grad_evals > cap["max_test_time_gradient_evals"]:
        raise CapExceeded(f"test-time gradient-eval cap {cap['max_test_time_gradient_evals']} exceeded")


def _block_ends(loader: ReplayLoader, samples: Dict[str, Sample]) -> Dict[int, str]:
    """step -> domain for the last batch of every contiguous single-domain block."""
    doms = []
    for b in loader.batch_ids():
        ds = {samples[i].domain for i in b}
        if len(ds) != 1:
            raise IntegrityError("regime evaluation needs single-domain batches")
        doms.append(ds.pop())
    return {t: d for t, d in enumerate(doms) if t == len(doms) - 1 or doms[t + 1] != d}


def replay_one(ctx: Context, adapter: Adapter, split: str, holdout_split: str, order: OrderSpec,
               evaluator: Optional[TerminalHoldoutEvaluator], method: str, phase: str,
               log_steps: bool = True, regime_eval: Optional[RegimeHoldoutEvaluator] = None
               ) -> Tuple[Optional[TerminalResult], dict]:
    """Adapt on one full order of ``split``; optionally score regime ends and the terminal holdout.

    ``evaluator=None`` skips terminal scoring (used by norm calibration, which needs no labels).
    State restoration is checked, not assumed: the adapter must start at theta0 with the source model's
    frozen parameters, leave those unchanged, and must not consume the global RNG.
    """
    world = ctx.world
    samples = _samples(world, split)
    loader = ReplayLoader(samples, world.ids(split), order, ctx.cfg["stream"]["batch_size"])
    stream_vault = LabelVault(world.labels(split), allowed_readers=[PrequentialLabelOracle.READER])
    oracle = PrequentialLabelOracle(stream_vault, order) if ctx.cfg.get("prequential", True) else None
    ends = _block_ends(loader, samples) if regime_eval is not None else {}
    source_fp = ctx.model.fingerprint()
    if adapter.model.fingerprint() != source_fp or adapter.theta != adapter.model.initial_theta():
        raise IntegrityError(f"{method}/{order.name}: adapter does not start from the restored source state")
    rng_state = random.getstate()
    g0 = adapter.total_grad_evals()
    t0 = time.perf_counter()
    online_wrong = 0
    online_preds: List[int] = []
    block_wrong, block_n = 0, 0
    regime: List[dict] = []
    for batch in loader:
        _check_cap(ctx, phase)
        if oracle is not None:  # predict-then-score; the adapter never sees the oracle
            preds = [max(range(len(p)), key=p.__getitem__) for p in adapter.predict_proba(batch.xs)]
            oracle.commit(batch.ids)
            wrong = sum(int(p != oracle.label(i)) for p, i in zip(preds, batch.ids))
            online_wrong += wrong
            block_wrong += wrong
            block_n += len(batch.ids)
            online_preds.extend(preds)
        rec = adapter.step(batch)
        if log_steps:
            ctx.log.write(event="step", phase=phase, replicate=ctx.replicate, method=method, order=order.name,
                          ids_sha256=sha256_json(list(batch.ids)), **rec.as_dict())
        if batch.step in ends:
            dom = ends[batch.step]
            regime.append({"domain": dom, "step": batch.step,
                           "holdout_error": regime_eval.evaluate(adapter.predict_proba, dom),
                           "block_online_error": block_wrong / block_n if block_n else None})
            block_wrong, block_n = 0, 0
    loader.verify_complete()
    result = evaluator.evaluate(adapter.predict_proba, loader) if evaluator is not None else None
    if adapter.model.fingerprint() != source_fp:
        raise IntegrityError(f"{method}/{order.name}: frozen source parameters changed during adaptation")
    if random.getstate() != rng_state:
        raise IntegrityError(f"{method}/{order.name}: adapter consumed the global RNG")
    wall = time.perf_counter() - t0
    grads = adapter.total_grad_evals() - g0
    if phase == "test_time":
        ctx.test_time_grad_evals += grads
    n = len(world.ids(split))
    n_classes = len(ctx.model.b)
    steps = [r.update_norm for r in adapter.trace]
    info = {"online_error": online_wrong / n if oracle is not None else None,
            "terminal_displacement": adapter.displacement(),
            "path_length": adapter.path_length(),
            "mean_step_norm": sum(steps) / len(steps) if steps else 0.0,
            "max_step_norm": max(steps) if steps else 0.0,
            "theta_sha256": sha256_json([round(x, 12) for x in adapter.theta]),
            "prequential_label_reads": stream_vault.count(),
            "non_scorer_stream_label_reads": sum(1 for r, _ in stream_vault.access_log
                                                 if r != PrequentialLabelOracle.READER),
            "prequential_violations": len(oracle.violations) if oracle else 0,
            "online_class_distribution": class_distribution(online_preds, n_classes) if online_preds else None,
            "terminal_class_distribution": (class_distribution(result.predictions, n_classes)
                                            if result is not None else None),
            "regime": regime,
            "regime_end_error_mean": (sum(r["holdout_error"] for r in regime) / len(regime)) if regime else None,
            "terminal_regime_error": regime[-1]["holdout_error"] if regime else None,
            "state_checks": {"started_from_source_state": True, "frozen_params_unchanged": True,
                             "global_rng_untouched": True, "optimizer_state": "none (plain SGD)",
                             "normalization_buffers": ("none (fixed source statistics)"
                                                       if ctx.model.norm == "source"
                                                       else "none (per-batch statistics, no running buffer)")},
            "gradient_evals": grads, "wall_seconds": wall, "n_steps": loader.num_batches}
    # Adapters never read labels. Tuning *selects* an lr by holdout error, so that phase uses holdout labels.
    uses_labels = phase == "tuning"
    note = (f"terminal holdout={holdout_split if evaluator is not None else 'none'}; "
            f"regime holdout={'yes' if regime_eval is not None else 'no'}; "
            "stream labels read only by the prequential scorer"
            + ("; lr selection reads holdout labels" if uses_labels else ""))
    ctx.ledger.add(CostRecord(f"{phase}:{method}/{order.name}", split, n, sha256_json(world.ids(split)),
                              uses_labels=uses_labels, gradient_evals=grads, wall_seconds=wall, notes=note))
    return result, info


def _meta_batches(world: SyntheticWorld, n_batches: int, batch_size: int, seed: int
                  ) -> List[List[Tuple[Sample, int]]]:
    """Single-domain meta batches, round-robin over meta domains (so pairs span different shifts)."""
    rng = random.Random(seed)
    by_dom: Dict[str, List[Tuple[Sample, int]]] = {}
    for s, y in world.splits["meta_train"]:
        by_dom.setdefault(s.domain, []).append((s, y))
    doms = sorted(by_dom)
    out = []
    for i in range(n_batches):
        rows = by_dom[doms[i % len(doms)]]
        out.append(rng.sample(rows, min(batch_size, len(rows))))
    return out


def fit_subspaces(ctx: Context) -> Tuple[Dict[str, AdaptationSubspace], Dict[str, str]]:
    """Fit every configured subspace on meta_train. A failed fit is recorded (not raised) so that only
    the arms using that subspace become NOT_RUN."""
    world = ctx.world
    assert_disjoint({"meta_train": world.ids("meta_train"), **{s: world.ids(s) for s in TEST_SPLITS}})
    out: Dict[str, AdaptationSubspace] = {}
    failed: Dict[str, str] = {}
    caches: Dict[str, dict] = {}
    for name, sc in ctx.cfg.get("subspaces", {}).items():
        try:
            if sc.get("fitter") == "utility_only":
                ref = sc["of"]
                if ref in failed or ref not in caches:
                    raise RuntimeError(f"order-aware reference {ref!r} is unavailable")
                out[name] = _fit_utility_only(ctx, name, ref, out[ref], caches[ref])
                continue
            if isinstance(sc.get("k"), str):  # "match:<fitted subspace>" -> dimension-matched control
                ref = sc["k"].split(":", 1)[1]
                if ref in failed:
                    raise RuntimeError(f"dimension reference {ref!r} failed to fit")
                if ref not in out:
                    raise ConfigError(f"subspace {name}: k refers to {ref!r}, which must be fitted earlier")
                sc = dict(sc, k=out[ref].dim)
            out[name], cache = _fit_one(ctx, name, sc)
            if cache is not None:
                caches[name] = cache
        except Exception as e:  # noqa: BLE001 - preserved, dependent arms become NOT_RUN
            failed[name] = f"{type(e).__name__}: {e}"
            ctx.log.write(event="subspace_fit_failed", replicate=ctx.replicate, name=name, reason=failed[name],
                          traceback=traceback.format_exc())
    return out, failed


def _fit_utility_only(ctx: Context, name: str, ref: str, ref_sub: AdaptationSubspace,
                      cache: dict) -> AdaptationSubspace:
    """Rank-matched, penalty-free selection from the *same* meta statistics as ``ref`` (no new data)."""
    t0 = time.perf_counter()
    sub = utility_only_subspace(cache["pool"], cache["stats"], cache["jvps"], ref_sub.dim)
    sub.provenance.update({"shares_meta_statistics_with": ref, "rank_matched_to": ref,
                           "meta_ids_sha256": cache["ids_sha256"]})
    rec = CostRecord(f"meta_training:{name}", "meta_train", cache["n_ids"], cache["ids_sha256"], True, 0, 0, 0,
                     time.perf_counter() - t0,
                     notes=f"selection only; reuses {ref}'s meta batches, meta labels, pool, HVPs and JVPs "
                           f"(their cost is recorded once under meta_training:{ref})")
    ctx.ledger.add(rec)
    ctx.log.write(event="subspace_fit", replicate=ctx.replicate, name=name, fitter="utility_only", dim=sub.dim,
                  fingerprint=sub.fingerprint(), provenance=sub.provenance, cost=rec.as_dict())
    return sub


def _fit_one(ctx: Context, name: str, sc: dict) -> Tuple[AdaptationSubspace, Optional[dict]]:
    world, model = ctx.world, ctx.model
    cache: Optional[dict] = None
    theta0 = model.initial_theta()
    D = model.theta_dim
    t0 = time.perf_counter()
    g0 = model.grad_evals
    fitter = sc["fitter"]
    uses_labels, n_hvp, n_jvp, used_ids = False, 0, 0, []
    if fitter == "full":
        sub = full_space(D)
    elif fitter == "random":
        sub = (random_subspace(D, sc["k"], sc["seed"]) if sc["k"] > 0 else
               AdaptationSubspace("random_k0", [], D, {"fitter": "random_subspace", "k": 0}))
    elif fitter in ("gradient_pca", "order_aware"):
        batches = _meta_batches(world, sc["n_meta_batches"], sc["meta_batch_size"], sc.get("seed", 0))
        used_ids = [s.id for b in batches for s, _ in b]
        grads = [model.entropy_and_grad([s.x for s, _ in b], theta0)[1] for b in batches]
        if fitter == "gradient_pca":
            sub = (gradient_pca_subspace(grads, sc["k"], seed=sc.get("seed", 0)) if sc["k"] > 0 else
                   AdaptationSubspace("gradpca_k0", [], D, {"fitter": "gradient_pca_subspace", "k": 0}))
        else:
            uses_labels = True
            second = [[0.0] * D for _ in range(D)]
            for g in grads:
                second = la.mat_add(second, la.outer(g, g))
            pool = complete_basis(top_eigenvectors(second, D, seed=sc.get("seed", 0)), D)
            stats = []
            for b, g in zip(batches, grads):
                xs, ys = [s.x for s, _ in b], [y for _, y in b]
                g_sup = model.ce_and_grad(xs, ys, theta0)[1]
                h_pool = [model.entropy_hvp(xs, theta0, p) for p in pool]
                n_hvp += len(pool)
                stats.append(MetaBatchStats(g, g_sup, h_pool))
            probe_rows = random.Random(sc.get("seed", 0) + 1).sample(
                world.splits["meta_train"], min(sc["n_probe"], len(world.splits["meta_train"])))
            used_ids += [s.id for s, _ in probe_rows]
            probe = [s.x for s, _ in probe_rows]
            jvps = []
            for p in pool:
                h = 1e-4
                fp = [v for row in model.probs(probe, la.axpy(h, p, theta0)) for v in row]
                fm = [v for row in model.probs(probe, la.axpy(-h, p, theta0)) for v in row]
                jvps.append([(a - b_) / (2 * h) for a, b_ in zip(fp, fm)])
                n_jvp += 1
            sub = order_aware_subspace(pool, stats, jvps, sc["k"], sc.get("lr_for_scoring", 1.0), sc["lam"],
                                       normalized=sc.get("normalized", False))
            cache = {"pool": pool, "stats": stats, "jvps": jvps, "n_ids": len(used_ids),
                     "ids_sha256": sha256_json(sorted(used_ids))}
    else:
        raise ConfigError(f"unknown fitter {fitter!r}")
    wall = time.perf_counter() - t0
    rec = CostRecord(f"meta_training:{name}", "meta_train" if used_ids else "none", len(used_ids),
                     sha256_json(sorted(used_ids)), uses_labels, model.grad_evals - g0, n_hvp, n_jvp, wall,
                     notes=f"fitter={fitter}; HVP/JVP by central finite differences")
    ctx.ledger.add(rec)
    ctx.log.write(event="subspace_fit", replicate=ctx.replicate, name=name, fitter=fitter, dim=sub.dim,
                  fingerprint=sub.fingerprint(), provenance=sub.provenance, cost=rec.as_dict())
    return sub, cache


def _make_adapter(ctx: Context, m: dict, lr: float, subspaces: Dict[str, AdaptationSubspace],
                  step_norms=None, norm_reference: Optional[CausalNormReference] = None) -> Adapter:
    model = copy.deepcopy(ctx.model)
    model.grad_evals = 0
    if m["kind"] == "no_adapt":
        a = NoAdapt(model, m["name"])
    else:
        sub = subspaces[m["subspace"]] if m.get("subspace") else None
        a = EntropySGD(model, m["name"], lr, sub, step_norms, control_of=m.get("control_of"),
                       subspace_fit_uses_source_labels=bool(
                           sub and sub.provenance.get("fitter") in ("order_aware_subspace",
                                                                    "utility_only_subspace")),
                       norm_reference=norm_reference)
    return a


def run_replicate(cfg: dict, replicate_seed: int, log: RawLog, ledger: CostLedger, started: float,
                  cpu_deadline: Optional[float] = None) -> dict:
    wcfg = dict(cfg["world"], seed=replicate_seed)
    world = build_world(wcfg)
    all_splits = {k: world.ids(k) for k in world.splits}
    assert_disjoint(all_splits)
    mcfg = cfg["model"]
    t0 = time.perf_counter()
    model, src_cost = train_source_model(world.splits["source_train"], wcfg["n_classes"], mcfg["source_epochs"],
                                         mcfg["source_lr"], mcfg["norm"], seed=replicate_seed)
    ledger.add(CostRecord("source_training", "source_train", src_cost["n_samples"],
                          sha256_json(world.ids("source_train")), True,
                          gradient_evals=src_cost["full_batch_gradient_evals"],
                          wall_seconds=time.perf_counter() - t0, notes="full-batch GD on (W, b)"))
    rep = f"r{replicate_seed}"
    ctx = Context(cfg, world, model, ledger, log, rep, started, cpu_deadline=cpu_deadline)
    log.write(event="replicate_start", replicate=rep, world_fingerprint=world.fingerprint(),
              source_model_fingerprint=model.fingerprint(), source_cost=src_cost)

    subspaces, failed_subspaces = fit_subspaces(ctx)
    st = cfg["stream"]
    test_orders = _orders_for(cfg, world, "test_stream", st["order_seeds"])
    for o in test_orders:
        log.write(event="order", replicate=rep, split="test_stream", name=o.name, fingerprint=o.fingerprint())
    others = {k: v for k, v in all_splits.items() if k != "test_holdout"}
    test_eval = TerminalHoldoutEvaluator(world.samples("test_holdout"), world.labels("test_holdout"),
                                         others, st["eval_batch_size"])
    test_regime = (RegimeHoldoutEvaluator(world.samples("test_holdout"), world.labels("test_holdout"), others,
                                          st["eval_batch_size"]) if st.get("regime_eval") else None)

    # ---- tuning on the development split (never on test) -------------------------------------
    tuned: Dict[str, float] = {}
    tuning_failed: Dict[str, str] = {}
    tcfg = cfg.get("tuning", {})
    if tcfg.get("enabled"):
        dev_orders = _orders_for(cfg, world, "development", tcfg["order_seeds"])
        dev_others = {k: v for k, v in all_splits.items() if k != "dev_holdout"}
        dev_eval = TerminalHoldoutEvaluator(world.samples("dev_holdout"), world.labels("dev_holdout"),
                                            dev_others, st["eval_batch_size"])
        regime_metric = tcfg.get("metric") == "mean_regime_end_error"
        dev_regime = (RegimeHoldoutEvaluator(world.samples("dev_holdout"), world.labels("dev_holdout"), dev_others,
                                             st["eval_batch_size"]) if regime_metric else None)
        for m in cfg["methods"]:
            if m.get("lr") != "tune" or m.get("subspace") in failed_subspaces:
                continue
            try:
                scores = []
                for lr in tcfg["lr_grid"]:
                    errs = []
                    for o in dev_orders:
                        a = _make_adapter(ctx, m, lr, subspaces)
                        res, info = replay_one(ctx, a, "development", "dev_holdout", o,
                                               None if regime_metric else dev_eval, m["name"], "tuning",
                                               log_steps=False, regime_eval=dev_regime)
                        errs.append(info["regime_end_error_mean"] if regime_metric else res.error)
                    scores.append((sum(errs) / len(errs), lr))
                    log.write(event="tuning", replicate=rep, method=m["name"], lr=lr, dev_errors=errs)
                best = min(scores)  # ties -> smaller lr (tuple ordering)
                tuned[m["name"]] = best[1]
                log.write(event="tuning_selected", replicate=rep, method=m["name"], lr=best[1],
                          dev_mean_error=best[0])
            except Exception as e:  # noqa: BLE001 - preserved; the arm becomes NOT_RUN
                tuning_failed[m["name"]] = f"{type(e).__name__}: {e}"
                log.write(event="tuning_failed", replicate=rep, method=m["name"], reason=tuning_failed[m["name"]])

    # ---- test-time arms --------------------------------------------------------------------------
    lrs: Dict[str, float] = {}
    results: Dict[str, Dict[str, TerminalResult]] = {}
    infos: Dict[str, Dict[str, dict]] = {}
    status: Dict[str, Dict[str, dict]] = {}
    traces: Dict[str, Dict[str, list]] = {}
    cards: Dict[str, dict] = {}
    for m in cfg["methods"]:
        name = m["name"]
        results[name], infos[name], status[name], traces[name] = {}, {}, {}, {}
        dep_problem, dep_status = None, "NOT_RUN"
        step_norm_target = None
        causal_target: Optional[dict] = None
        try:
            if m.get("subspace") in failed_subspaces:
                raise RuntimeError(f"subspace {m['subspace']!r} failed to fit: {failed_subspaces[m['subspace']]}")
            if m["kind"] == "no_adapt":
                lr = 0.0
            elif "lr" in m:
                if m["lr"] == "tune" and name not in tuned:
                    why = tuning_failed.get(name, "not tuned")
                    if why.startswith("CapExceeded"):
                        raise CapExceeded(f"tuning stopped by cap: {why}")
                    raise RuntimeError(f"tuning did not produce an lr: {why}")
                lr = tuned[name] if m["lr"] == "tune" else float(m["lr"])
            elif "lr_from" in m:
                if m["lr_from"] not in lrs:
                    raise KeyError(f"lr of {m['lr_from']} unavailable")
                lr = lrs[m["lr_from"]]
                if "lr_factor" in m:
                    lr = small_lr(lr, m["lr_factor"])
            else:
                nm = m["norm_match"]
                tgt = nm["target"]
                if any(s["status"] != "OK" for s in status[tgt].values()):
                    raise RuntimeError(f"norm-match target {tgt} has non-OK runs")
                if nm["mode"] == "per_step":
                    lr = lrs[tgt]  # direction from full-space gradient; magnitude overwritten per step
                    step_norm_target = tgt
                elif nm["mode"] == "per_step_causal":
                    lr = lrs[tgt]  # direction only; magnitude comes from the lockstep replica of the target
                    causal_target = next(x for x in cfg["methods"] if x["name"] == tgt)
                elif nm["mode"] == "calibrated_global":
                    lr = _calibrate_global(ctx, m, next(x for x in cfg["methods"] if x["name"] == tgt),
                                           lrs[tgt], subspaces)
                else:
                    raise ConfigError(f"unknown norm_match mode {nm['mode']!r}")
            lrs[name] = lr
        except CapExceeded as e:
            dep_problem, dep_status = f"CapExceeded: {e}", "CAP_EXCEEDED"
        except Exception as e:  # noqa: BLE001
            dep_problem = f"{type(e).__name__}: {e}"
        for o in test_orders:
            cell = {"status": "NOT_RUN", "reason": None}
            if dep_problem:
                cell = {"status": dep_status, "reason": dep_problem}
            else:
                try:
                    norms, reference = None, None
                    if step_norm_target is not None:
                        if status[step_norm_target][o.name]["status"] != "OK":
                            raise RuntimeError("target run for this order is not OK")
                        norms = traces[step_norm_target][o.name]
                    if causal_target is not None:
                        reference = CausalNormReference(_make_adapter(ctx, causal_target, lr, subspaces))
                    a = _make_adapter(ctx, m, lr, subspaces, norms, reference)
                    res, info = replay_one(ctx, a, "test_stream", "test_holdout", o, test_eval, name, "test_time",
                                           regime_eval=test_regime)
                    if reference is not None:  # post-hoc audit only: the replica reproduced the target run
                        tgt_trace = traces[causal_target["name"]].get(o.name)
                        info["reference_grad_evals"] = reference.grad_evals
                        info["reference_matches_target_trace"] = (
                            tgt_trace is not None and len(tgt_trace) == len(reference.reference.trace)
                            and all(abs(x - r.update_norm) <= 1e-12
                                    for x, r in zip(tgt_trace, reference.reference.trace)))
                    results[name][o.name], infos[name][o.name] = res, info
                    traces[name][o.name] = per_step_norm_schedule(a.trace)
                    cards[name] = a.card.as_dict()
                    cell = {"status": "OK", "reason": None}
                    log.write(event="terminal", replicate=rep, method=name, order=o.name, **res.as_dict(), **info)
                except CapExceeded as e:
                    cell = {"status": "CAP_EXCEEDED", "reason": str(e)}
                except Exception as e:  # noqa: BLE001
                    cell = {"status": "FAILED", "reason": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()}
            status[name][o.name] = cell
            if cell["status"] != "OK":
                log.write(event="cell_status", replicate=rep, method=name, order=o.name, **cell)

    errors = {n: [results[n][o.name].error for o in test_orders if o.name in results[n]] for n in results}

    def table(key: str) -> Dict[str, List[float]]:
        return {n: [infos[n][o.name][key] for o in test_orders if o.name in infos[n]] for n in infos}

    metric_tables = {"terminal_error": errors, "online_error": table("online_error"),
                     "path_length": table("path_length"), "terminal_displacement": table("terminal_displacement")}
    if test_regime is not None:
        metric_tables["regime_end_error_mean"] = table("regime_end_error_mean")
        metric_tables["terminal_regime_error"] = table("terminal_regime_error")
    diagnostics = {n: {o: {k: infos[n][o][k] for k in ("online_class_distribution", "terminal_class_distribution",
                                                        "regime", "path_length", "mean_step_norm", "max_step_norm",
                                                        "state_checks", "gradient_evals", "wall_seconds", "n_steps")
                           + tuple(k for k in ("reference_grad_evals", "reference_matches_target_trace")
                                   if k in infos[n][o])}
                       for o in infos[n]} for n in infos}
    method_status = {n: ("OK" if all(c["status"] == "OK" for c in status[n].values()) else
                         sorted({c["status"] for c in status[n].values() if c["status"] != "OK"})[0])
                     for n in status}
    output_diff = {n: pairwise_output_difference([results[n][o.name] for o in test_orders if o.name in results[n]])
                   for n in results}
    return {"replicate": rep, "errors": errors, "metric_tables": metric_tables, "diagnostics": diagnostics,
            "method_status": method_status, "cells": status, "lrs": lrs,
            "tuned_lrs": tuned, "tuning_failed": tuning_failed, "cards": cards,
            "output_level_order_difference": output_diff,
            "orders": [{"name": o.name, "fingerprint": o.fingerprint()} for o in test_orders],
            "online_error": {n: {o: i["online_error"] for o, i in infos[n].items()} for n in infos},
            "terminal_displacement": {n: {o: i["terminal_displacement"] for o, i in infos[n].items()} for n in infos},
            "per_domain_error": {n: {o: r.per_domain_error for o, r in results[n].items()} for n in results},
            "prequential_label_reads": {n: {o: i["prequential_label_reads"] for o, i in infos[n].items()}
                                        for n in infos},
            "non_scorer_stream_label_reads": {n: sum(i["non_scorer_stream_label_reads"] for i in infos[n].values())
                                              for n in infos},
            "holdout_fingerprint": test_eval.fingerprint, "holdout_label_reads": test_eval.label_reads,
            "regime_holdout_label_reads": test_regime.label_reads if test_regime is not None else 0,
            "world_fingerprint": world.fingerprint(), "source_model_fingerprint": model.fingerprint(),
            "subspaces": {k: {"dim": v.dim, "fingerprint": v.fingerprint(), "provenance": v.provenance}
                          for k, v in subspaces.items()},
            "failed_subspaces": failed_subspaces}


def _calibrate_global(ctx: Context, m: dict, target_m: dict, target_lr: float,
                      subspaces: Dict[str, AdaptationSubspace]) -> float:
    """Match the target's mean terminal displacement on the calibration split (never the test split)."""
    nm = m["norm_match"]
    orders = _orders_for(ctx.cfg, ctx.world, "calibration", nm["calibration_order_seeds"])

    def mean_disp(adapter_factory) -> float:  # displacement only: no holdout, no labels
        ds = []
        for o in orders:
            a = adapter_factory()
            _, info = replay_one(ctx, a, "calibration", "none", o, None, m["name"], "calibration",
                                 log_steps=False)
            ds.append(info["terminal_displacement"])
        return sum(ds) / len(ds)

    target = mean_disp(lambda: _make_adapter(ctx, target_m, target_lr, subspaces))
    probe_m = {"name": m["name"], "kind": "entropy_sgd", "subspace": None}
    res = calibrate_lr_to_displacement(lambda lr: mean_disp(lambda: _make_adapter(ctx, probe_m, lr, subspaces)),
                                       target, nm["lr_lo"], nm["lr_hi"], nm["iters"])
    ctx.log.write(event="norm_calibration", replicate=ctx.replicate, method=m["name"], target_method=target_m["name"],
                  target_mean_displacement=target, lr=res.lr, achieved=res.achieved,
                  relative_mismatch=res.relative_mismatch, history=res.iterations)
    tol = nm.get("max_relative_mismatch", 0.05)
    if res.relative_mismatch > tol:
        raise RuntimeError(f"calibration mismatch {res.relative_mismatch:.3g} > {tol}")
    return res.lr


# ------------------------------------------------------------------------------------------ main

def science_status(run_type: str) -> str:
    return {"smoke": "SCIENCE_NOT_EVALUATED",
            "toy_pilot": "TOY_PILOT_ONLY_NOT_EVIDENCE_ABOUT_REAL_TTA",
            "toy_diagnostic": "TOY_DIAGNOSTIC_ONLY_NOT_EVIDENCE_ABOUT_REAL_TTA"}[run_type]


def run(config_path: str, out_root: str, run_id: Optional[str] = None,
        cpu_deadline: Optional[float] = None) -> str:
    """Run one config. ``cpu_deadline`` (absolute ``cpu_seconds_used()`` value) lets a stage driver share one
    CPU budget across several configs; otherwise ``resource_cap.max_cpu_seconds`` (if set) starts now."""
    with open(config_path) as f:
        cfg = json.load(f)
    warnings = validate_config(cfg)
    cpu_start = cpu_seconds_used()
    if cpu_deadline is None and "max_cpu_seconds" in cfg["resource_cap"]:
        cpu_deadline = cpu_start + cfg["resource_cap"]["max_cpu_seconds"]
    run_id = run_id or f"{os.path.splitext(os.path.basename(config_path))[0]}_{time.strftime('%Y%m%dT%H%M%S')}"
    git_before = git_info(REPO_ROOT)  # before any output file exists, so 'dirty' reflects code/config only
    out_dir = os.path.join(out_root, run_id)
    os.makedirs(out_dir, exist_ok=False)
    tracemalloc.start()
    started_wall = utc_now()
    started = time.perf_counter()
    log = RawLog(os.path.join(out_dir, "raw_log.jsonl"))
    ledger = CostLedger()
    log.write(event="run_start", run_id=run_id, config_path=config_path, config_sha256=canonical_sha256(cfg),
              warnings=warnings, utc=started_wall)
    reps = []
    cap_skipped: List[int] = []
    run_error = None
    try:
        for seed in cfg.get("replicate_seeds", [cfg["world"]["seed"]]):
            if cpu_deadline is not None and cpu_seconds_used() > cpu_deadline:
                cap_skipped.append(seed)  # every cell of this replicate is CAP_EXCEEDED, recorded below
                log.write(event="replicate_cap_exceeded", seed=seed)
                continue
            reps.append(run_replicate(cfg, seed, log, ledger, started, cpu_deadline))
    except Exception as e:  # noqa: BLE001 - preserved in manifest
        run_error = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
        log.write(event="run_failed", **run_error)
    wall = time.perf_counter() - started
    cpu_used = cpu_seconds_used() - cpu_start
    _, py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    log.write(event="run_end", wall_seconds=wall)
    log.close()

    dec_cfg = cfg["decision"]
    errors = {r["replicate"]: r["errors"] for r in reps}
    statuses: Dict[str, str] = {}
    for r in reps:
        for mth, s in r["method_status"].items():
            statuses[mth] = s if statuses.get(mth, "OK") == "OK" else statuses[mth]
    if cap_skipped:
        statuses = {m["name"]: "CAP_EXCEEDED" for m in cfg["methods"]}
    ok_methods = [mth for mth, s in statuses.items() if s == "OK"]
    summary_stats = summarize({rep: {mth: e[mth] for mth in ok_methods} for rep, e in errors.items()},
                              dec_cfg["no_adapt"]) if reps and dec_cfg["no_adapt"] in ok_methods else {}
    if dec_cfg.get("mode") == "seedwise_v2":
        # per-condition descriptive summary only; the stage driver applies the v2 rule across conditions
        arms = [m["name"] for m in cfg["methods"]]
        seedwise = {}
        for metric in (reps[0]["metric_tables"] if reps else {}):
            tab = {r["replicate"]: r["metric_tables"][metric] for r in reps}
            seedwise[metric] = seedwise_condition_summary(tab, arms, dec_cfg["candidate"], dec_cfg["no_adapt"],
                                                          [a for a in arms if a != dec_cfg["candidate"]])
        decision = {"label": "SEE_STAGE_DECISION" if not run_error else "INCOMPLETE", "mode": "seedwise_v2",
                    "seedwise": seedwise, "run_error": run_error}
    else:
        decision = decide(errors, statuses, dec_cfg) if reps and not run_error else {"label": "INCOMPLETE",
                                                                                       "run_error": run_error}
    summary = {
        "run_id": run_id, "run_type": cfg["run_type"],
        "software_status": "RUN_COMPLETED" if not run_error else "RUN_FAILED",
        "science_status": science_status(cfg["run_type"]),
        "decision": decision,
        "method_status": statuses,
        "cap_skipped_replicates": cap_skipped,
        "metrics": summary_stats,
        "replicates": reps,
        "cost_ledger": ledger.as_dict(),
        "notes": ["observed_worst_* are the worst of the sampled orders only, not a risk bound over all orders",
                  "cost caps are limits; actual spend per arm is in cost_ledger"],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    manifest = {
        "run_id": run_id, "command_config": config_path, "utc_start": started_wall, "utc_end": utc_now(),
        "wall_seconds": wall, "cpu_seconds": cpu_used, "cpu_deadline": cpu_deadline,
        "peak_python_heap_mib": py_peak / (1024 * 1024), "peak_rss_mib": peak_rss_mib(),
        "git": git_before, "config_sha256": canonical_sha256(cfg), "config_file_sha256": file_sha256(config_path),
        "data": {r["replicate"]: r["world_fingerprint"] for r in reps},
        "source_models": {r["replicate"]: r["source_model_fingerprint"] for r in reps},
        "subspaces": {r["replicate"]: r["subspaces"] for r in reps},
        "holdout": {r["replicate"]: r["holdout_fingerprint"] for r in reps},
        "orders": {r["replicate"]: r["orders"] for r in reps},
        "environment": environment(), "raw_log_events": log.n_events,
        "raw_log_sha256": file_sha256(os.path.join(out_dir, "raw_log.jsonl")),
        "summary_sha256": file_sha256(os.path.join(out_dir, "summary.json")),
        "run_error": run_error, "warnings": warnings,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    return out_dir


def dry_run(config_path: str) -> dict:
    """Validate config, build data and orders, check split disjointness and order integrity. No adaptation."""
    with open(config_path) as f:
        cfg = json.load(f)
    warnings = validate_config(cfg)
    plan = {"config_sha256": canonical_sha256(cfg), "run_type": cfg["run_type"], "warnings": warnings,
            "replicates": []}
    st = cfg["stream"]
    n_adapt = sum(1 for m in cfg["methods"] if m["kind"] != "no_adapt")
    for seed in cfg.get("replicate_seeds", [cfg["world"]["seed"]]):
        world = build_world(dict(cfg["world"], seed=seed))
        splits = {k: world.ids(k) for k in world.splits}
        assert_disjoint(splits)
        orders = _orders_for(cfg, world, "test_stream", st["order_seeds"])
        n_steps = 0
        for o in orders:
            loader = ReplayLoader(_samples(world, "test_stream"), world.ids("test_stream"), o, st["batch_size"])
            if st.get("regime_eval"):
                _block_ends(loader, _samples(world, "test_stream"))
            n_steps = max(n_steps, loader.num_batches)
        plan["replicates"].append({
            "seed": seed, "world_fingerprint": world.fingerprint(),
            "split_sizes": {k: len(v) for k, v in splits.items()},
            "orders": [{"name": o.name, "fingerprint": o.fingerprint()} for o in orders],
            "test_steps_per_order": n_steps,
            "test_time_adaptation_steps_total": n_steps * len(orders) * n_adapt})
    return plan


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "runs"))
    ap.add_argument("--run-id")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run(args.config), indent=1))
        return 0
    out = run(args.config, args.out, args.run_id)
    with open(os.path.join(out, "summary.json")) as f:
        s = json.load(f)
    print(json.dumps({"out_dir": out, "software_status": s["software_status"],
                      "science_status": s["science_status"], "decision": s["decision"]["label"],
                      "method_status": s["method_status"]}, indent=1))
    return 0 if s["software_status"] == "RUN_COMPLETED" else 1


__all__ = ["validate_config", "run", "dry_run", "main", "ConfigError", "IntegrityError"]

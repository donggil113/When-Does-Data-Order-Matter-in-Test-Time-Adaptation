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
from .analysis import decide, summarize
from .evaluator import TerminalHoldoutEvaluator, TerminalResult, pairwise_output_difference
from .manifest import canonical_sha256, environment, file_sha256, git_info, peak_rss_mib, utc_now
from .methods import (Adapter, EntropySGD, NoAdapt, calibrate_lr_to_displacement, per_step_norm_schedule,
                      small_lr)
from .replay import (IntegrityError, LabelVault, OrderSpec, PrequentialLabelOracle, ReplayLoader, Sample,
                     assert_disjoint, make_orders, sha256_json)
from .subspace import (AdaptationSubspace, CostLedger, CostRecord, MetaBatchStats, complete_basis,
                       full_space, gradient_pca_subspace, order_aware_subspace, random_subspace,
                       top_eigenvectors)
from .toy import LinearTTAModel, SyntheticWorld, build_world, train_source_model

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_SPLITS = ("test_stream", "test_holdout")
PREREG_KEYS = ("primary_metric", "spread_metric", "mie", "seeds", "tuning_range", "resource_cap", "splits")


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
    if cfg["run_type"] not in ("smoke", "toy_pilot"):
        raise ConfigError("run_type must be 'smoke' or 'toy_pilot'")
    if cfg["run_type"] != "smoke":
        pre = cfg.get("preregistration")
        if not pre:
            raise ConfigError("pilot configs must carry a 'preregistration' block fixed before running")
        for k in PREREG_KEYS:
            if k not in pre:
                raise ConfigError(f"preregistration missing {k!r}")
        if pre["mie"] != cfg["decision"]["mie"]:
            raise ConfigError("preregistration.mie and decision.mie differ")
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
        names.append(n)
    dec = cfg["decision"]
    for key in ("candidate", "baseline", "no_adapt"):
        if dec[key] not in names:
            raise ConfigError(f"decision.{key}={dec[key]!r} is not a method")
    for k in dec.get("stability_controls", []) + dec.get("dimension_controls", []):
        if k not in names:
            raise ConfigError(f"decision control {k!r} is not a method")
    tcfg = cfg.get("tuning", {})
    if tcfg.get("enabled"):
        for key, allowed in (("split", "development"), ("holdout", "dev_holdout"), ("metric", "mean_terminal_error")):
            if tcfg.get(key, allowed) != allowed:
                raise ConfigError(f"tuning.{key} must be {allowed!r} (the only supported value)")
        if not tcfg.get("lr_grid") or not tcfg.get("order_seeds"):
            raise ConfigError("tuning needs lr_grid and order_seeds")
    st = cfg["stream"]
    if st["order_family"] not in ("uniform_permutation", "domain_blocked"):
        raise ConfigError("stream.order_family must be uniform_permutation or domain_blocked")
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


def _samples(world: SyntheticWorld, split: str) -> Dict[str, Sample]:
    return {s.id: s for s in world.samples(split)}


def _orders_for(world: SyntheticWorld, split: str, family: str, seeds: Sequence[int],
                domain_sequences=None) -> List[OrderSpec]:
    samples = world.samples(split)
    return make_orders([s.id for s in samples], family, seeds, {s.id: s.domain for s in samples},
                       domain_sequences)


def _check_cap(ctx: Context, phase: str) -> None:
    """Wall-clock cap: whole run, every phase. Gradient-eval cap: test-time phase, per replicate."""
    cap = ctx.cfg["resource_cap"]
    if time.perf_counter() - ctx.started > cap["max_wall_seconds"]:
        raise CapExceeded(f"wall-clock cap {cap['max_wall_seconds']}s exceeded")
    if phase == "test_time" and ctx.test_time_grad_evals > cap["max_test_time_gradient_evals"]:
        raise CapExceeded(f"test-time gradient-eval cap {cap['max_test_time_gradient_evals']} exceeded")


def replay_one(ctx: Context, adapter: Adapter, split: str, holdout_split: str, order: OrderSpec,
               evaluator: TerminalHoldoutEvaluator, method: str, phase: str,
               log_steps: bool = True) -> Tuple[TerminalResult, dict]:
    """Adapt on one full order of ``split`` and evaluate terminally on ``holdout_split``."""
    world = ctx.world
    samples = _samples(world, split)
    loader = ReplayLoader(samples, world.ids(split), order, ctx.cfg["stream"]["batch_size"])
    stream_vault = LabelVault(world.labels(split), allowed_readers=[PrequentialLabelOracle.READER])
    oracle = PrequentialLabelOracle(stream_vault, order) if ctx.cfg.get("prequential", True) else None
    g0 = adapter.model.grad_evals
    t0 = time.perf_counter()
    online_wrong = 0
    for batch in loader:
        _check_cap(ctx, phase)
        if oracle is not None:  # predict-then-score; the adapter never sees the oracle
            preds = [max(range(len(p)), key=p.__getitem__) for p in adapter.predict_proba(batch.xs)]
            oracle.commit(batch.ids)
            online_wrong += sum(int(p != oracle.label(i)) for p, i in zip(preds, batch.ids))
        rec = adapter.step(batch)
        if log_steps:
            ctx.log.write(event="step", phase=phase, replicate=ctx.replicate, method=method, order=order.name,
                          ids_sha256=sha256_json(list(batch.ids)), **rec.as_dict())
    loader.verify_complete()
    result = evaluator.evaluate(adapter.predict_proba, loader)
    wall = time.perf_counter() - t0
    grads = adapter.model.grad_evals - g0
    if phase == "test_time":
        ctx.test_time_grad_evals += grads
    n = len(world.ids(split))
    info = {"online_error": online_wrong / n if oracle is not None else None,
            "terminal_displacement": adapter.displacement(),
            "theta_sha256": sha256_json([round(x, 12) for x in adapter.theta]),
            "prequential_label_reads": stream_vault.count(),
            "non_scorer_stream_label_reads": sum(1 for r, _ in stream_vault.access_log
                                                 if r != PrequentialLabelOracle.READER),
            "prequential_violations": len(oracle.violations) if oracle else 0,
            "gradient_evals": grads, "wall_seconds": wall, "n_steps": loader.num_batches}
    ctx.ledger.add(CostRecord(f"{phase}:{method}/{order.name}", split, n, sha256_json(world.ids(split)),
                              uses_labels=False, gradient_evals=grads, wall_seconds=wall,
                              notes=f"terminal holdout={holdout_split}; stream labels read only by prequential scorer"))
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
    for name, sc in ctx.cfg.get("subspaces", {}).items():
        try:
            if isinstance(sc.get("k"), str):  # "match:<fitted subspace>" -> dimension-matched control
                ref = sc["k"].split(":", 1)[1]
                if ref in failed:
                    raise RuntimeError(f"dimension reference {ref!r} failed to fit")
                if ref not in out:
                    raise ConfigError(f"subspace {name}: k refers to {ref!r}, which must be fitted earlier")
                sc = dict(sc, k=out[ref].dim)
            out[name] = _fit_one(ctx, name, sc)
        except Exception as e:  # noqa: BLE001 - preserved, dependent arms become NOT_RUN
            failed[name] = f"{type(e).__name__}: {e}"
            ctx.log.write(event="subspace_fit_failed", replicate=ctx.replicate, name=name, reason=failed[name],
                          traceback=traceback.format_exc())
    return out, failed


def _fit_one(ctx: Context, name: str, sc: dict) -> AdaptationSubspace:
    world, model = ctx.world, ctx.model
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
    else:
        raise ConfigError(f"unknown fitter {fitter!r}")
    wall = time.perf_counter() - t0
    rec = CostRecord(f"meta_training:{name}", "meta_train" if used_ids else "none", len(used_ids),
                     sha256_json(sorted(used_ids)), uses_labels, model.grad_evals - g0, n_hvp, n_jvp, wall,
                     notes=f"fitter={fitter}; HVP/JVP by central finite differences")
    ctx.ledger.add(rec)
    ctx.log.write(event="subspace_fit", replicate=ctx.replicate, name=name, fitter=fitter, dim=sub.dim,
                  fingerprint=sub.fingerprint(), provenance=sub.provenance, cost=rec.as_dict())
    return sub


def _make_adapter(ctx: Context, m: dict, lr: float, subspaces: Dict[str, AdaptationSubspace],
                  step_norms=None) -> Adapter:
    model = copy.deepcopy(ctx.model)
    model.grad_evals = 0
    if m["kind"] == "no_adapt":
        a = NoAdapt(model, m["name"])
    else:
        sub = subspaces[m["subspace"]] if m.get("subspace") else None
        a = EntropySGD(model, m["name"], lr, sub, step_norms, control_of=m.get("control_of"),
                       subspace_fit_uses_source_labels=bool(
                           sub and sub.provenance.get("fitter") == "order_aware_subspace"))
    return a


def run_replicate(cfg: dict, replicate_seed: int, log: RawLog, ledger: CostLedger, started: float) -> dict:
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
    ctx = Context(cfg, world, model, ledger, log, rep, started)
    log.write(event="replicate_start", replicate=rep, world_fingerprint=world.fingerprint(),
              source_model_fingerprint=model.fingerprint(), source_cost=src_cost)

    subspaces, failed_subspaces = fit_subspaces(ctx)
    st = cfg["stream"]
    fam = st["order_family"]
    test_orders = _orders_for(world, "test_stream", fam, st["order_seeds"], st.get("domain_sequences"))
    for o in test_orders:
        log.write(event="order", replicate=rep, split="test_stream", name=o.name, fingerprint=o.fingerprint())
    others = {k: v for k, v in all_splits.items() if k != "test_holdout"}
    test_eval = TerminalHoldoutEvaluator(world.samples("test_holdout"), world.labels("test_holdout"),
                                         others, st["eval_batch_size"])

    # ---- tuning on the development split (never on test) -------------------------------------
    tuned: Dict[str, float] = {}
    tuning_failed: Dict[str, str] = {}
    tcfg = cfg.get("tuning", {})
    if tcfg.get("enabled"):
        dev_orders = _orders_for(world, "development", fam, tcfg["order_seeds"], st.get("domain_sequences"))
        dev_eval = TerminalHoldoutEvaluator(world.samples("dev_holdout"), world.labels("dev_holdout"),
                                            {k: v for k, v in all_splits.items() if k != "dev_holdout"},
                                            st["eval_batch_size"])
        for m in cfg["methods"]:
            if m.get("lr") != "tune" or m.get("subspace") in failed_subspaces:
                continue
            try:
                scores = []
                for lr in tcfg["lr_grid"]:
                    errs = []
                    for o in dev_orders:
                        a = _make_adapter(ctx, m, lr, subspaces)
                        res, _ = replay_one(ctx, a, "development", "dev_holdout", o, dev_eval, m["name"],
                                            "tuning", log_steps=False)
                        errs.append(res.error)
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
        dep_problem = None
        step_norm_target = None
        try:
            if m.get("subspace") in failed_subspaces:
                raise RuntimeError(f"subspace {m['subspace']!r} failed to fit: {failed_subspaces[m['subspace']]}")
            if m["kind"] == "no_adapt":
                lr = 0.0
            elif "lr" in m:
                if m["lr"] == "tune" and name not in tuned:
                    raise RuntimeError(f"tuning did not produce an lr: {tuning_failed.get(name, 'not tuned')}")
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
                elif nm["mode"] == "calibrated_global":
                    lr = _calibrate_global(ctx, m, next(x for x in cfg["methods"] if x["name"] == tgt),
                                           lrs[tgt], subspaces)
                else:
                    raise ConfigError(f"unknown norm_match mode {nm['mode']!r}")
            lrs[name] = lr
        except Exception as e:  # noqa: BLE001
            dep_problem = f"{type(e).__name__}: {e}"
        for o in test_orders:
            cell = {"status": "NOT_RUN", "reason": None}
            if dep_problem:
                cell["reason"] = dep_problem
            else:
                try:
                    norms = None
                    if step_norm_target is not None:
                        if status[step_norm_target][o.name]["status"] != "OK":
                            raise RuntimeError("target run for this order is not OK")
                        norms = traces[step_norm_target][o.name]
                    a = _make_adapter(ctx, m, lr, subspaces, norms)
                    res, info = replay_one(ctx, a, "test_stream", "test_holdout", o, test_eval, name, "test_time")
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
    method_status = {n: ("OK" if all(c["status"] == "OK" for c in status[n].values()) else
                         sorted({c["status"] for c in status[n].values() if c["status"] != "OK"})[0])
                     for n in status}
    output_diff = {n: pairwise_output_difference([results[n][o.name] for o in test_orders if o.name in results[n]])
                   for n in results}
    return {"replicate": rep, "errors": errors, "method_status": method_status, "cells": status, "lrs": lrs,
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
            "world_fingerprint": world.fingerprint(), "source_model_fingerprint": model.fingerprint(),
            "subspaces": {k: {"dim": v.dim, "fingerprint": v.fingerprint(), "provenance": v.provenance}
                          for k, v in subspaces.items()},
            "failed_subspaces": failed_subspaces}


def _calibrate_global(ctx: Context, m: dict, target_m: dict, target_lr: float,
                      subspaces: Dict[str, AdaptationSubspace]) -> float:
    """Match the target's mean terminal displacement on the calibration split (never the test split)."""
    nm = m["norm_match"]
    world = ctx.world
    fam = ctx.cfg["stream"]["order_family"]
    orders = _orders_for(world, "calibration", fam, nm["calibration_order_seeds"],
                         ctx.cfg["stream"].get("domain_sequences"))
    # the calibration terminal evaluation reuses dev_holdout only to satisfy the terminal-eval protocol
    ev = TerminalHoldoutEvaluator(world.samples("dev_holdout"), world.labels("dev_holdout"),
                                  {"calibration": world.ids("calibration")}, ctx.cfg["stream"]["eval_batch_size"])

    def mean_disp(adapter_factory) -> float:
        ds = []
        for o in orders:
            a = adapter_factory()
            _, info = replay_one(ctx, a, "calibration", "dev_holdout", o, ev, m["name"], "calibration",
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
            "toy_pilot": "TOY_PILOT_ONLY_NOT_EVIDENCE_ABOUT_REAL_TTA"}[run_type]


def run(config_path: str, out_root: str, run_id: Optional[str] = None) -> str:
    with open(config_path) as f:
        cfg = json.load(f)
    warnings = validate_config(cfg)
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
    run_error = None
    try:
        for seed in cfg.get("replicate_seeds", [cfg["world"]["seed"]]):
            reps.append(run_replicate(cfg, seed, log, ledger, started))
    except Exception as e:  # noqa: BLE001 - preserved in manifest
        run_error = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
        log.write(event="run_failed", **run_error)
    wall = time.perf_counter() - started
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
    ok_methods = [mth for mth, s in statuses.items() if s == "OK"]
    summary_stats = summarize({rep: {mth: e[mth] for mth in ok_methods} for rep, e in errors.items()},
                              dec_cfg["no_adapt"]) if reps and dec_cfg["no_adapt"] in ok_methods else {}
    decision = decide(errors, statuses, dec_cfg) if reps and not run_error else {"label": "INCOMPLETE",
                                                                                   "run_error": run_error}
    summary = {
        "run_id": run_id, "run_type": cfg["run_type"],
        "software_status": "RUN_COMPLETED" if not run_error else "RUN_FAILED",
        "science_status": science_status(cfg["run_type"]),
        "decision": decision,
        "method_status": statuses,
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
        "wall_seconds": wall, "peak_python_heap_mib": py_peak / (1024 * 1024), "peak_rss_mib": peak_rss_mib(),
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
        orders = _orders_for(world, "test_stream", st["order_family"], st["order_seeds"], st.get("domain_sequences"))
        for o in orders:
            ReplayLoader(_samples(world, "test_stream"), world.ids("test_stream"), o, st["batch_size"])
        n_steps = -(-len(world.ids("test_stream")) // st["batch_size"])
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

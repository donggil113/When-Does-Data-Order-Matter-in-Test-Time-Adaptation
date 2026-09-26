"""Export manuscript tables and number macros from the recorded stage-2 run (read-only, stdlib only).

Usage (from the repository root):
    python3 paper/scripts/export_results.py

Inputs (never modified):   runs/stage2_order_penalty_0119a34/{stage_summary,stage_manifest}.json and
                           <condition>/{summary.json, raw_log.jsonl, manifest.json}
Outputs (regenerated):     paper/generated/numbers.tex    -- \\res{key} macros used by the text
                           paper/generated/results.json   -- every exported value with its source field
                           paper/tables/*.tex             -- tables included by the manuscript
                           paper/tables/display_names.tsv -- arm_id -> display name (run ids unchanged)
Every value in numbers.tex and the tables is computed here from the raw files; nothing is typed by hand.
"""

import json
import os
import statistics
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN = "runs/stage2_order_penalty_0119a34"
CONDS = {"stationary_fixed_batches": "stat", "drift_fixed_batches": "drift", "stationary_reshuffle": "reshuf"}
SEEDS = ["r0", "r1", "r2"]
ARMS = ["no_adapt", "tent_full", "tent_full_lr_x0.3", "tent_full_lr_x0.1", "tent_orderaware",
        "tent_utility_only", "norm_matched_per_step_causal", "norm_matched_global"]
# Display names only; arm ids in configs, logs and summaries are unchanged. The baseline is an
# entropy-minimisation toy (feature-wise affine on fixed source statistics, plain SGD), not Tent as published.
DISPLAY = {
    "no_adapt": "No adaptation",
    "tent_full": "EM-toy (full affine)",
    "tent_full_lr_x0.3": "EM-toy, lr $\\times$0.3",
    "tent_full_lr_x0.1": "EM-toy, lr $\\times$0.1",
    "tent_orderaware": "Order-penalized subspace (candidate)",
    "tent_utility_only": "Utility-only subspace",
    "norm_matched_per_step_causal": "Per-step norm-matched (causal)",
    "norm_matched_global": "Global norm-matched (calibrated)",
}
SHORT = {"no_adapt": "No adaptation", "tent_full": "EM-toy", "tent_full_lr_x0.3": "EM-toy lr$\\times$0.3",
         "tent_full_lr_x0.1": "EM-toy lr$\\times$0.1", "tent_orderaware": "Candidate",
         "tent_utility_only": "Utility-only", "norm_matched_per_step_causal": "Norm-matched (step)",
         "norm_matched_global": "Norm-matched (global)"}
KEY = {"no_adapt": "noad", "tent_full": "full", "tent_full_lr_x0.3": "lrA", "tent_full_lr_x0.1": "lrB",
       "tent_orderaware": "cand", "tent_utility_only": "util", "norm_matched_per_step_causal": "nstep",
       "norm_matched_global": "nglob"}
CONTROLS = ["tent_utility_only", "tent_full_lr_x0.3", "tent_full_lr_x0.1", "norm_matched_per_step_causal",
            "norm_matched_global", "tent_full"]

values = {}  # key -> {"tex": str, "value": float|str, "source": str}


def load(path):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)


def put(key, value, tex, source):
    values[key] = {"value": value, "tex": tex, "source": source}


def pct(x, sign=False):
    return (f"{100 * x:+.2f}" if sign else f"{100 * x:.2f}").replace("-", "\\ensuremath{-}")


def fmt(x, nd=2):
    return f"{x:.{nd}f}".replace("-", "\\ensuremath{-}")


def main():
    stage = load(f"{RUN}/stage_summary.json")
    smani = load(f"{RUN}/stage_manifest.json")
    summ = {c: load(f"{RUN}/{c}/summary.json") for c in CONDS}
    mani = {c: load(f"{RUN}/{c}/manifest.json") for c in CONDS}
    src_stage = f"{RUN}/stage_summary.json"

    # ---------------------------------------------------------------- decision and run facts
    dec = stage["decision"]
    put("decision", dec["label"], dec["label"].replace("_", "\\_"), f"{src_stage}:decision.label")
    put("nreasons", len(dec["reasons"]), str(len(dec["reasons"])), f"{src_stage}:decision.reasons")
    put("cpu", smani["cpu_seconds_total_incl_children"], f"{smani['cpu_seconds_total_incl_children']:.1f}",
        f"{RUN}/stage_manifest.json:cpu_seconds_total_incl_children")
    put("cpubudget", smani["cpu_budget_seconds"], f"{smani['cpu_budget_seconds']:,}".replace(",", "{,}"),
        f"{RUN}/stage_manifest.json:cpu_budget_seconds")
    put("rss", smani["peak_rss_mib_self"], f"{smani['peak_rss_mib_self']:.1f}",
        f"{RUN}/stage_manifest.json:peak_rss_mib_self")
    put("commit", smani["git"]["commit"][:7], smani["git"]["commit"][:7], f"{RUN}/stage_manifest.json:git.commit")
    ncells, nok = 0, 0
    for c, s in summ.items():
        for r in s["replicates"]:
            for arm, cells in r["cells"].items():
                for cell in cells.values():
                    ncells += 1
                    nok += cell["status"] == "OK"
    put("ncells", ncells, str(ncells), f"{RUN}/*/summary.json:replicates[].cells")
    put("ncellsok", nok, str(nok), f"{RUN}/*/summary.json:replicates[].cells[].status")

    # ---------------------------------------------------------------- world / protocol facts
    cfg = load("configs/stage2_stationary_fixed_batches.json")
    npc = cfg["world"]["n_per_class"]
    world = cfg["world"]
    put("dim", world["dim"], str(world["dim"]), "configs/stage2_stationary_fixed_batches.json:world.dim")
    put("ncls", world["n_classes"], str(world["n_classes"]), "configs/...:world.n_classes")
    put("ntestdom", world["n_test_domains"], str(world["n_test_domains"]), "configs/...:world.n_test_domains")
    nstream = world["n_test_domains"] * world["n_classes"] * npc["test_stream"]
    nhold = world["n_test_domains"] * world["n_classes"] * npc["test_holdout"]
    put("nstream", nstream, str(nstream), "configs/...:world.n_per_class.test_stream x domains x classes")
    put("nhold", nhold, str(nhold), "configs/...:world.n_per_class.test_holdout x domains x classes")
    put("batch", cfg["stream"]["batch_size"], str(cfg["stream"]["batch_size"]), "configs/...:stream.batch_size")
    put("norders", len(cfg["stream"]["order_seeds"]), str(len(cfg["stream"]["order_seeds"])),
        "configs/...:stream.order_seeds")
    put("nseeds", len(cfg["replicate_seeds"]), str(len(cfg["replicate_seeds"])), "configs/...:replicate_seeds")
    put("mieg", cfg["decision"]["mie"]["gain_abs"], pct(cfg["decision"]["mie"]["gain_abs"]),
        "configs/...:decision.mie.gain_abs")
    put("mies", cfg["decision"]["mie"]["spread_abs"], pct(cfg["decision"]["mie"]["spread_abs"]),
        "configs/...:decision.mie.spread_abs")
    put("tolg", cfg["decision"]["tolerance"]["gain_abs"], pct(cfg["decision"]["tolerance"]["gain_abs"]),
        "configs/...:decision.tolerance.gain_abs")
    grid = cfg["tuning"]["lr_grid"]
    put("lrgrid", grid, "\\{" + ", ".join(f"{g:g}" for g in grid) + "\\}", "configs/...:tuning.lr_grid")
    put("lrmax", max(grid), f"{max(grid):g}", "configs/...:tuning.lr_grid")
    steps = {c: sorted({d["n_steps"] for d in summ[c]["replicates"][0]["diagnostics"]["tent_full"].values()})[0]
             for c in CONDS}
    put("stepsstat", steps["stationary_fixed_batches"], str(steps["stationary_fixed_batches"]),
        f"{RUN}/stationary_fixed_batches/summary.json:replicates[0].diagnostics.tent_full[*].n_steps")
    put("stepsdrift", steps["drift_fixed_batches"], str(steps["drift_fixed_batches"]),
        f"{RUN}/drift_fixed_batches/summary.json:replicates[0].diagnostics.tent_full[*].n_steps")

    # ---------------------------------------------------------------- seedwise metrics
    metric_of = {"stat": "terminal_error", "reshuf": "terminal_error", "drift": "regime_end_error_mean"}
    tables = {}
    for c, ck in CONDS.items():
        metrics = [metric_of[ck]] + (["online_error", "terminal_error"] if ck == "drift" else [])
        for met in metrics:
            sw = summ[c]["decision"]["seedwise"][met]
            mk = {"terminal_error": "te", "regime_end_error_mean": "re", "online_error": "on"}[met]
            base = f"{RUN}/{c}/summary.json:decision.seedwise.{met}"
            for arm in ARMS:
                for sd in SEEDS:
                    row = sw["per_arm"][arm]["per_seed"][sd]
                    for field, fk, sign in (("mean", "mean", False), ("std_over_orders", "std", False),
                                            ("observed_max", "max", False), ("mean_gain_vs_no_adapt", "gain", True)):
                        k = f"{ck}.{mk}.{KEY[arm]}.{fk}.{sd}"
                        put(k, row[field], pct(row[field], sign), f"{base}.per_arm.{arm}.per_seed.{sd}.{field}")
                        tables[k] = row[field]
            for ctrl in [a for a in ARMS if a != "tent_orderaware"]:
                for sd in SEEDS:
                    row = sw["paired_vs_candidate"][ctrl]["per_seed"][sd]
                    for field, fk in (("mean_diff_c_minus_k", "dmean"), ("std_diff_c_minus_k", "dstd"),
                                      ("max_diff_c_minus_k", "dmax")):
                        k = f"{ck}.{mk}.pair.{KEY[ctrl]}.{fk}.{sd}"
                        put(k, row[field], pct(row[field], True),
                            f"{base}.paired_vs_candidate.{ctrl}.per_seed.{sd}.{field}")

    # stage-rule evidence
    ev = dec["evidence"]
    for i, sd in enumerate(SEEDS):
        put(f"ev.gain.{sd}", ev["stationary_gain_vs_no_adapt_per_seed"][i],
            pct(ev["stationary_gain_vs_no_adapt_per_seed"][i], True),
            f"{src_stage}:decision.evidence.stationary_gain_vs_no_adapt_per_seed[{i}]")
        put(f"ev.stdred.{sd}", ev["stationary_std_reduction_vs_baseline_per_seed"][i],
            pct(ev["stationary_std_reduction_vs_baseline_per_seed"][i], True),
            f"{src_stage}:decision.evidence.stationary_std_reduction_vs_baseline_per_seed[{i}]")
    for m, mk in (("regime_end_error", "re"), ("online_error", "on")):
        g = ev["drift_guard"][m]
        put(f"ev.drift.{mk}.gain", g["candidate_gain_vs_no_adapt"], pct(g["candidate_gain_vs_no_adapt"], True),
            f"{src_stage}:decision.evidence.drift_guard.{m}.candidate_gain_vs_no_adapt")
        put(f"ev.drift.{mk}.vsutil", g["candidate_minus_reference_mean"], pct(g["candidate_minus_reference_mean"], True),
            f"{src_stage}:decision.evidence.drift_guard.{m}.candidate_minus_reference_mean")

    # ---------------------------------------------------------------- per-seed lr, subspace, diagnostics
    identity_rows = []
    diag_rows = defaultdict(dict)
    for c, ck in CONDS.items():
        raw = [json.loads(line) for line in open(os.path.join(ROOT, RUN, c, "raw_log.jsonl"))]
        term = {(e["replicate"], e["method"], e["order"]): e for e in raw if e["event"] == "terminal"}
        stepn = defaultdict(list)
        for e in raw:
            if e["event"] == "step" and e["phase"] == "test_time":
                stepn[(e["replicate"], e["method"], e["order"])].append(e["update_norm"])
        for r in summ[c]["replicates"]:
            sd = r["replicate"]
            oa, uo = r["subspaces"]["order_aware"]["provenance"], r["subspaces"]["utility_only"]["provenance"]
            for arm in ("tent_full", "tent_orderaware", "tent_utility_only", "norm_matched_global"):
                put(f"{ck}.lr.{KEY[arm]}.{sd}", r["lrs"][arm], f"{r['lrs'][arm]:.3g}",
                    f"{RUN}/{c}/summary.json:replicates[{sd}].lrs.{arm}")
            put(f"{ck}.rank.{sd}", r["subspaces"]["order_aware"]["dim"], str(r["subspaces"]["order_aware"]["dim"]),
                f"{RUN}/{c}/summary.json:replicates[{sd}].subspaces.order_aware.dim")
            orders = [o["name"] for o in r["orders"]]
            same_theta = all(term[(sd, "tent_orderaware", o)]["theta_sha256"] ==
                             term[(sd, "tent_utility_only", o)]["theta_sha256"] for o in orders)
            same_steps = all(stepn[(sd, "tent_orderaware", o)] == stepn[(sd, "tent_utility_only", o)]
                             for o in orders)
            identity_rows.append({"cond": c, "seed": sd, "oa_idx": oa["chosen_pool_indices"],
                                  "uo_idx": uo["chosen_pool_indices"], "oa_stop": oa["stopped"],
                                  "same_index_set": sorted(oa["chosen_pool_indices"]) == sorted(uo["chosen_pool_indices"]),
                                  "shared_stats": uo.get("shares_meta_statistics_with") == "order_aware",
                                  "lr_oa": r["lrs"]["tent_orderaware"], "lr_uo": r["lrs"]["tent_utility_only"],
                                  "same_theta_all_orders": same_theta, "same_step_norms_all_orders": same_steps,
                                  "source_fp": r["source_model_fingerprint"][:12]})
            for arm in ARMS:
                d = r["diagnostics"][arm]
                diag_rows[(ck, arm)][sd] = {
                    "path": statistics.fmean(v["path_length"] for v in d.values()),
                    "disp": statistics.fmean(r["terminal_displacement"][arm][o] for o in d),
                    "maxshare": max(v["terminal_class_distribution"]["max_share"] for v in d.values()),
                    "ncls_min": min(v["terminal_class_distribution"]["n_classes_predicted"] for v in d.values())}
                for f in ("path", "maxshare"):
                    put(f"{ck}.diag.{KEY[arm]}.{f}.{sd}", diag_rows[(ck, arm)][sd][f],
                        fmt(diag_rows[(ck, arm)][sd][f], 2 if f == "path" else 3),
                        f"{RUN}/{c}/summary.json:replicates[{sd}].diagnostics.{arm}[*]")
    ranks = sorted({values[k]["value"] for k in values if ".rank." in k})
    put("ranks", ranks, ", ".join(str(x) for x in ranks), f"{RUN}/*/summary.json:replicates[].subspaces.order_aware.dim")
    n_identical = sum(1 for row in identity_rows if row["same_theta_all_orders"])
    put("nidentical", n_identical, str(n_identical), f"{RUN}/*/raw_log.jsonl:terminal.theta_sha256")
    put("nidentityrows", len(identity_rows), str(len(identity_rows)), f"{RUN}/*/summary.json:replicates")
    # grid-boundary count for the candidate
    nbound = sum(1 for c, ck in CONDS.items() for r in summ[c]["replicates"]
                 if abs(r["lrs"]["tent_orderaware"] - max(grid)) < 1e-12)
    put("ncandbound", nbound, str(nbound), f"{RUN}/*/summary.json:replicates[].lrs.tent_orderaware")

    # ---------------------------------------------------------------- per-arm actual costs (per condition)
    cost = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    for c, ck in CONDS.items():
        for rec in summ[c]["cost_ledger"]["records"]:
            phase, _, rest = rec["phase"].partition(":")
            arm = rest.split("/")[0] if rest else phase
            # gradient calls only: finite-difference HVPs are already counted as their two gradient calls,
            # forward-only JVPs are not gradient evaluations
            cost[(ck, phase)][arm][0] += rec["gradient_evals"]
            cost[(ck, phase)][arm][1] += rec["wall_seconds"]

    # ---------------------------------------------------------------- write outputs
    gen = os.path.join(ROOT, "paper", "generated")
    tab = os.path.join(ROOT, "paper", "tables")
    os.makedirs(gen, exist_ok=True)
    os.makedirs(tab, exist_ok=True)
    with open(os.path.join(gen, "numbers.tex"), "w") as f:
        f.write("% AUTO-GENERATED by paper/scripts/export_results.py from " + RUN + ". Do not edit.\n")
        for k in sorted(values):
            f.write(f"\\expandafter\\def\\csname res@{k}\\endcsname{{{values[k]['tex']}}}\n")
    with open(os.path.join(gen, "results.json"), "w") as f:
        json.dump({"run": RUN, "values": values, "identity_rows": identity_rows,
                   "costs": {f"{k[0]}:{k[1]}": v for k, v in cost.items()}}, f, indent=1, sort_keys=True)
    with open(os.path.join(tab, "display_names.tsv"), "w") as f:
        f.write("arm_id\tdisplay_name\tmacro_key\n")
        for a in ARMS:
            f.write(f"{a}\t{DISPLAY[a]}\t{KEY[a]}\n")

    def seed_cols(ck, mk, arm, fields):
        return " & ".join(f"\\res{{{ck}.{mk}.{KEY[arm]}.{fk}.{sd}}}" for sd in SEEDS for fk in fields)

    def main_table(ck, mk, caption, label, fields=("mean", "std", "gain"), heads=("Err", "SD", "Gain"),
                   placement="t"):
        n = len(fields)
        lines = ["% AUTO-GENERATED by paper/scripts/export_results.py. Do not edit.",
                 f"\\begin{{table}}[{placement}]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{3.2pt}",
                 f"\\caption{{{caption}}}", f"\\label{{{label}}}",
                 "\\begin{tabular}{l" + ("r" * n + "@{\\hspace{7pt}}") * 3 + "}", "\\toprule",
                 " & " + " & ".join(f"\\multicolumn{{{n}}}{{c}}{{World seed {sd}}}" for sd in SEEDS) + " \\\\",
                 " ".join(f"\\cmidrule(lr){{{2 + i * n}-{1 + (i + 1) * n}}}" for i in range(3)),
                 "Arm & " + " & ".join(h for _ in SEEDS for h in heads) + " \\\\", "\\midrule"]
        for arm in ARMS:
            lines.append(f"{DISPLAY[arm]} & {seed_cols(ck, mk, arm, fields)} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        return "\n".join(lines) + "\n"

    with open(os.path.join(tab, "tab_stationary.tex"), "w") as f:
        f.write(main_table("stat", "te",
                           "Primary stationary condition (fixed batch membership, batch order permuted). Err: mean "
                           "terminal common-holdout error over the 8 observed orders (\\%); SD: standard deviation "
                           "over those orders (percentage points, pp); Gain: no-adaptation error minus arm error "
                           "(pp; positive is better). Orders are nested in world seeds and are not independent "
                           "replicates.", "tab:stationary"))
    with open(os.path.join(tab, "tab_reshuffle.tex"), "w") as f:
        f.write(main_table("reshuf", "te",
                           "Secondary condition (samples reshuffled and re-batched per order). Same columns as "
                           "Table~\\ref{tab:stationary}. Not used by the decision rule.", "tab:reshuffle", placement="!h"))
    with open(os.path.join(tab, "tab_drift.tex"), "w") as f:
        f.write(main_table("drift", "re",
                           "Drift condition (single-domain batches fixed; orders permute the domain-block sequence "
                           "and batches within blocks). Err: mean over the four block ends of the current-regime "
                           "holdout error (\\%); SD over orders (pp); Gain vs.\\ no adaptation (pp).",
                           "tab:drift"))
    with open(os.path.join(tab, "tab_drift_online.tex"), "w") as f:
        f.write(main_table("drift", "on",
                           "Drift condition, predict-then-update online error over the stream (\\%), SD over "
                           "orders (pp) and gain vs.\\ no adaptation (pp).", "tab:drift-online", placement="!h"))
    with open(os.path.join(tab, "tab_drift_mixture.tex"), "w") as f:
        f.write(main_table("drift", "te",
                           "Drift condition, terminal error on the mixture holdout (\\%). Reported for "
                           "completeness only; it penalises adaptation to the latest regime and is not used by the "
                           "decision rule.", "tab:drift-mixture", placement="!h"))
    with open(os.path.join(tab, "tab_max.tex"), "w") as f:
        f.write(main_table("stat", "te", "Primary stationary condition: worst observed order (maximum terminal "
                           "error over the 8 sampled orders, \\%). This is not a bound over all orders.",
                           "tab:max", fields=("max",), heads=("Max",), placement="!h"))

    # paired differences
    lines = ["% AUTO-GENERATED by paper/scripts/export_results.py. Do not edit.", "\\begin{table}[t]",
             "\\centering", "\\small", "\\setlength{\\tabcolsep}{3.2pt}",
             "\\caption{Paired differences, candidate minus control, computed per world seed over the same 8 "
             "orders (pp; negative favours the candidate). $\\Delta$Err: mean terminal error difference; "
             "$\\Delta$SD: difference of across-order standard deviations. Primary stationary condition.}",
             "\\label{tab:paired}", "\\begin{tabular}{l" + "rr@{\\hspace{7pt}}" * 3 + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{World seed {sd}}}" for sd in SEEDS) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + i * 2}-{3 + i * 2}}}" for i in range(3)),
             "Control & " + " & ".join("$\\Delta$Err & $\\Delta$SD" for _ in SEEDS) + " \\\\", "\\midrule"]
    for ctrl in CONTROLS:
        cells = " & ".join(f"\\res{{stat.te.pair.{KEY[ctrl]}.{fk}.{sd}}}" for sd in SEEDS for fk in ("dmean", "dstd"))
        lines.append(f"{DISPLAY[ctrl]} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(tab, "tab_paired.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # identity of candidate and utility-only
    lines = ["% AUTO-GENERATED by paper/scripts/export_results.py. Do not edit.", "\\begin{table}[t]",
             "\\centering", "\\small", "\\setlength{\\tabcolsep}{3pt}",
             "\\caption{Candidate (order-penalized) versus utility-only subspace per condition and world seed. "
             "Pool indices of the selected directions (both fitters select from the same shared pool and meta "
             "statistics), tuned learning rates, and whether the terminal parameters (SHA-256 of $\\theta_T$) and "
             "every per-step update norm coincide on all 8 orders, read from the raw logs.}",
             "\\label{tab:identity}", "\\begin{tabular}{llllrrcc}", "\\toprule",
             "Condition & Seed & Cand.\\ idx & Util.\\ idx & lr cand. & lr util. & same $\\theta_T$ & same steps \\\\",
             "\\midrule"]
    cname = {"stationary_fixed_batches": "stationary", "drift_fixed_batches": "drift",
             "stationary_reshuffle": "reshuffle"}
    for row in identity_rows:
        lines.append(f"{cname[row['cond']]} & {row['seed']} & {row['oa_idx']} & {row['uo_idx']} & "
                     f"{row['lr_oa']:g} & {row['lr_uo']:g} & {'yes' if row['same_theta_all_orders'] else 'no'} & "
                     f"{'yes' if row['same_step_norms_all_orders'] else 'no'} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(tab, "tab_identity.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # diagnostics: lr, path length, max class share (appendix)
    lines = ["% AUTO-GENERATED by paper/scripts/export_results.py. Do not edit.", "\\begin{table}[!h]",
             "\\centering", "\\small", "\\setlength{\\tabcolsep}{3.6pt}",
             "\\caption{Primary stationary condition diagnostics per world seed: selected learning rate, mean "
             "update path length $\\sum_t\\|\\Delta\\theta_t\\|$ over orders, and the largest predicted-class "
             "share on the holdout over orders (collapse indicator; 1/3 is balanced for 3 classes).}",
             "\\label{tab:diag}", "\\begin{tabular}{l" + "rrr@{\\hspace{6pt}}" * 3 + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{World seed {sd}}}" for sd in SEEDS) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + i * 3}-{4 + i * 3}}}" for i in range(3)),
             "Arm & " + " & ".join("lr & path & max share" for _ in SEEDS) + " \\\\", "\\midrule"]
    sstat = summ["stationary_fixed_batches"]["replicates"]
    for arm in ARMS:
        cells = []
        for r in sstat:
            sd = r["replicate"]
            d = diag_rows[("stat", arm)][sd]
            cells += [f"{r['lrs'][arm]:.3g}", fmt(d["path"]), fmt(d["maxshare"], 3)]
        lines.append(f"{SHORT[arm]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(tab, "tab_diag.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # costs (appendix)
    lines = ["% AUTO-GENERATED by paper/scripts/export_results.py. Do not edit.", "\\begin{table}[!h]",
             "\\centering", "\\small", "\\setlength{\\tabcolsep}{3.5pt}",
             "\\caption{Actual spend per arm in the primary stationary condition, summed over 3 world seeds: "
             "gradient evaluations (finite-difference Hessian-vector products counted as their two gradient calls; "
             "forward-only Jacobian-vector products not counted) and wall-clock seconds by phase. Utility-only reuses the "
             "candidate's meta statistics; its meta-fit cost is selection only. The causal per-step control "
             "includes the lockstep reference replica. Source training (shared by all arms) is listed once.}",
             "\\label{tab:cost}", "\\begin{tabular}{lrrrrrr}", "\\toprule",
             "Arm & \\multicolumn{2}{c}{Meta fit} & \\multicolumn{2}{c}{Tuning / calib.} & "
             "\\multicolumn{2}{c}{Test time} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             " & evals & s & evals & s & evals & s \\\\", "\\midrule"]
    meta_key = {"tent_orderaware": "order_aware", "tent_utility_only": "utility_only"}
    for arm in ARMS:
        m = cost[("stat", "meta_training")].get(meta_key.get(arm, "-"), [0, 0.0])
        t = cost[("stat", "tuning")].get(arm, [0, 0.0])
        cal = cost[("stat", "calibration")].get(arm, [0, 0.0])
        tt = cost[("stat", "test_time")].get(arm, [0, 0.0])
        lines.append(f"{SHORT[arm]} & {m[0]} & {m[1]:.2f} & {t[0] + cal[0]} & {t[1] + cal[1]:.2f} & "
                     f"{tt[0]} & {tt[1]:.2f} \\\\")
    src = cost[("stat", "source_training")]["source_training"]
    lines += ["\\midrule", f"Source training (shared) & \\multicolumn{{6}}{{l}}{{{src[0]} full-batch epochs, "
              f"{src[1]:.2f} s}} \\\\", "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(tab, "tab_cost.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"exported {len(values)} values, {len(identity_rows)} identity rows; "
          f"identical candidate/utility trajectories in {n_identical}/{len(identity_rows)} (condition, seed) pairs")


if __name__ == "__main__":
    main()

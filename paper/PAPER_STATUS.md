# PAPER_STATUS: Working Draft v0

| field | value |
|---|---|
| title | Order Sensitivity and Adaptation Utility in a Controlled Test-Time Adaptation Model |
| status | **INTERNAL WORKING DRAFT v0**: not submitted, not under review, anonymous format, **HUMAN_REVIEW_PENDING** |
| format | official ICLR 2027 style ZIP (sha256 `0d940dfa…`), copied unmodified; running head overridden in `main.tex`; title footnote marks not-submitted status |
| build | **BUILT**: `paper/main.pdf`, 14 pages. Main text is pages 1–8 (limit 9). No overfull boxes, no undefined refs or citations. Local TeX Live 2023 unpacked from signed Ubuntu debs, not installed system-wide (see `BUILD.md`) |
| evidence | one recorded run `runs/stage2_order_penalty_0119a34/` (commit 0119a34). All numbers are macros exported by `paper/scripts/export_results.py` |
| scientific status | **ORDER_PENALTY_ADDED_UTILITY_NOT_SUPPORTED** in this synthetic implementation; candidate **ON_HOLD**; no real-data evidence; no novelty claim |

## Content state

| section | state |
|---|---|
| Abstract, Introduction, Related Work, Problem and Experimental Setting, Method under Study, Experiments and Observations, Discussion and Limitations, Conclusion | full prose; numbers from macros |
| AI use statement | full; human review marked pending |
| Reproducibility statement | full |
| Appendix A: Planned Experiments (NOT RUN) | full |
| Appendix B: Additional tables | generated from raw results |
| Appendix C: Derivation details | full; standard estimate, explicitly not a new result |
| Appendix D: Information access | full |
| Appendix E: Reproducibility details and deviations | full |
| Appendix F: Formula checks | full; numerical checks, not proofs |

## Work done in this pass (P1 paper task)

- **Display names.** `paper/tables/display_names.tsv` maps arm ids to display names; arm and run ids are
  unchanged.
  - `tent_full` is shown as "EM-toy" because it differs from published Tent: fixed source statistics, input
    affine parameters, plain SGD.
  - The subspace arms use labelled meta data, and the manuscript does not call them source-free.
- **Condition separation.** Results are exported separately for primary stationary (fixed batches), secondary
  (re-batched) and drift (current-regime holdout plus online error).
  - Exported per seed and arm: mean, SD over orders, observed maximum, gain vs no adaptation, update path
    length, largest predicted-class share, learning rate and actual cost.
  - Paired differences are exported against utility-only, both small-LR controls and both norm controls.
- **r0/r2 identity check.** The raw logs were compared for candidate vs utility-only: pool indices, projector
  (same index set from the same pool), learning rate, the source-model/θ0 state checks, terminal θ hashes and
  per-step update norms. All coincide in 6 of 9 (condition, seed) pairs, and the manuscript explains the
  implementation reason. r1 is not used to claim efficacy.
- **Formula checks** (`paper/scripts/check_two_step_identity.py`):
  - the exact two-step identity held in 200/200 rational quadratic cases;
  - the stated remainder bound held in 200/200 quartic cases, with maximum ratio 0.996.
- **STATUS.md cross-check.** The stage-2 stationary table in STATUS.md (96 cells) matches the raw values. No
  correction was needed.

## Not done / open

- Human review of every claim (`paper/claim_evidence.tsv`), every citation (sub-agent verified; two rechecked
  by metadata only) and the AI use statement.
- Real-image TTA, a new environment in which adaptation helps, more seeds and continuous pools: listed under
  Planned Experiments (NOT RUN).
- The candidate stays ON_HOLD. There will be no λ, learning-rate, seed or model sweep.

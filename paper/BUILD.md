# Building the manuscript

## Inputs

- `paper/main.tex`, `paper/sections/*.tex`, `paper/appendix.tex`, `paper/references.bib`
- `paper/tables/*.tex` and `paper/generated/numbers.tex`: **generated**; do not edit by hand
- `paper/iclr2027/`: official ICLR 2027 style files, **unmodified** (see below)

## Regenerate numbers and tables from raw results

```sh
python3 paper/scripts/run_capped.py analysis 600 -- python3 paper/scripts/export_results.py
```

The script reads only `runs/stage2_order_penalty_0119a34/` and the frozen configs. `run_capped.py` applies
`RLIMIT_CPU` and `RLIMIT_AS = 3 GiB` to the child process. It appends actual CPU time, peak RSS and wall time
to `paper/generated/budget_log.tsv`.

## Official style files

| field | value |
|---|---|
| guidelines page | https://iclr.cc/Conferences/2027/AuthorGuidelines (fetched 2026-09-26, HTTP 200) |
| ZIP | https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip |
| ZIP sha256 | `0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e` |
| ZIP contents | `iclr2027/` with `iclr2027_conference.{sty,bst,tex,bib}`, `math_commands.tex`, `fancyhdr.sty`, `natbib.sty` (sty and tex dated 2026-07-28 in the ZIP) |
| copied, unmodified | `iclr2027_conference.sty`, `iclr2027_conference.bst`, `fancyhdr.sty`, `natbib.sty`, `math_commands.tex` (hashes in `paper/iclr2027/SHA256SUMS`) |

**Running-head override.** With `\iclrfinalcopy` unset, the official style prints the running head "Under
review as a conference paper at ICLR 2027". This draft has not been submitted. `main.tex` therefore resets the
left head after `\maketitle` to "Internal working draft v0 --- not submitted to any venue". The style file is
unchanged. `\iclrfinalcopy` is not used, and there is no submission ID or anonymous URL.

## Compile

Status: see `PAPER_STATUS.md` (build section).

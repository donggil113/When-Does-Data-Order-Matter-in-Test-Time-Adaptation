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

**Anonymous author block.** The official anonymous block also prints "Paper under double-blind review". We do
not alter the style, so a title footnote states that the draft is not submitted and not under review, and that
this line comes from the template.

## Compile (what was actually done)

There was no LaTeX engine in the environment, and GitHub release downloads (e.g. tectonic) are blocked by the
session policy. We therefore used Ubuntu's signed archive **without installing anything system-wide**.

1. `apt-get download` fetched 14 packages (46.1 MB, 6.6 s wall; download cost recorded separately from the
   build budget):
   - `texlive-binaries`, `texlive-base`, `texlive-latex-base`, `texlive-latex-recommended`,
     `texlive-fonts-recommended`, `tex-common`
   - the libraries `libkpathsea6`, `libptexenc1`, `libsynctex2`, `libtexlua53-5`, `libteckit0`,
     `libzzip-0-13t64`, `libpotrace0`, `libpaper1`

   Versions are TeX Live 2023.20240207-1 and binaries 2023.20230311.66589-9build3. apt verified each
   package against the signed index. The sha256 of every .deb was also recorded in the session scratchpad
   (`texdeb/debs.sha256`).
2. The packages were unpacked with `dpkg-deb -x` into a scratch directory (`$TEX_LOCAL/root`). Nothing was
   written to `/usr`, `/etc` or the system package database.
3. A local `texmf.cnf` override (`$TEX_LOCAL/local/web2c/texmf.cnf`) points kpathsea at the unpacked tree.
4. The `pdflatex` format was built with `pdftex -ini -etex pdflatex.ini`. Hyphenation is US English
   (`hyphen.tex`), because Debian generates `language.dat` only at install time.
5. `pdftex.map` was assembled from the shipped map files (amsfonts, `utm`, `uhv`, `ucr`) plus Adobe-name
   aliases to the metric-compatible URW Type 1 fonts (`ptmr8r` -> `utmr8a.pfb`, and so on). This is what
   `updmap` does with `LW35=URWkb`. All 20 fonts in the PDF are embedded Type 1 (checked with `pdffonts`).
6. The manuscript was compiled with `TEX_LOCAL=<dir> sh paper/scripts/build_pdf.sh`: pdflatex, bibtex, then
   pdflatex until cross-references settle, under
   `python3 paper/scripts/run_capped.py build 600 -- ...` with RLIMIT_CPU and RLIMIT_AS = 3 GiB.
7. For page inspection only, `poppler-utils` 24.02.0-1ubuntu9 (212 kB) was downloaded and unpacked the same
   way. It provided `pdfinfo`, `pdffonts`, `pdftoppm` and `pdftotext`; the matching 9.8 build returned 404
   from the mirror. It is not needed for building.

The unpacked tree lives in the session scratchpad and is not committed. To rebuild elsewhere, use any TeX Live
2023+ with `pdflatex` and `bibtex` and run the three commands in `build_pdf.sh` with the same `TEXINPUTS` and
`BSTINPUTS`.

## Result of the final build

- `paper/main.pdf` (copy of `paper/build/main.pdf`): 14 pages, US letter.
- Main text runs from page 1 to the end of the Conclusion on page 8, which is within the 9-page limit. The AI
  use and reproducibility statements, which the official template says do not count toward the limit, the
  references (pages 8--9) and the appendix (pages 10--14) follow.
- `paper/build/main.log`: 0 overfull boxes, 0 undefined references, 0 undefined citations.
- Budget log in `paper/generated/budget_log.tsv`:
  - build and static checks: 14 capped runs, 79.21 CPU-s of 600. This includes two failed builds (an
    unescaped underscore; a missing font map), kept in the log, and a full unit-test run.
  - analysis: 10 runs, 2.41 CPU-s of 600.
  - peak RSS ≤ 41.1 MiB.

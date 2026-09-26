#!/usr/bin/env sh
# Compile paper/main.tex with pdflatex + bibtex into paper/build/.
# Requires TEX_LOCAL=<dir> holding an extracted (not installed) TeX Live tree at $TEX_LOCAL/root and the local
# kpathsea override + format at $TEX_LOCAL/local (see paper/BUILD.md for how that tree was produced).
set -eu
cd "$(dirname "$0")/.."
: "${TEX_LOCAL:?set TEX_LOCAL (see paper/BUILD.md)}"
R="$TEX_LOCAL/root"; B="$TEX_LOCAL/local"
export PATH="$R/usr/bin:$PATH" LD_LIBRARY_PATH="$R/usr/lib/x86_64-linux-gnu"
export TEXMFCNF="$B/web2c:$R/usr/share/texlive/texmf-dist/web2c"
export TEXINPUTS=".:./iclr2027//:" BSTINPUTS=".:./iclr2027:" BIBINPUTS=".:"
mkdir -p build
run() { pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex; }
run > build/pass1.out
(cd build && BIBINPUTS="..:" BSTINPUTS="../iclr2027:" bibtex.original main > bibtex.out)
for i in 2 3 4 5; do
  run > "build/pass$i.out"
  grep -q "Rerun to get" build/main.log || break
done
if grep -q "Rerun to get" build/main.log; then echo "cross-references did not settle" >&2; exit 2; fi
echo "built build/main.pdf after $i passes"

#!/usr/bin/env sh
# Stdlib-only checks: unit tests, then config dry-runs. No downloads, no installs, CPU only.
set -eu
cd "$(dirname "$0")/.."
python3 -m unittest discover -s tests -t . -v
for c in configs/*.json; do
  PYTHONPATH=src python3 -m ordertta.run --config "$c" --dry-run > /dev/null
  echo "dry-run OK: $c"
done

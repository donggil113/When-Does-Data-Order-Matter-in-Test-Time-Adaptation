#!/usr/bin/env sh
# Stdlib-only checks: unit tests, then config dry-runs. No downloads, no installs, CPU only.
set -eu
cd "$(dirname "$0")/.."
python3 -m unittest discover -s tests -t . -v
for c in configs/*.json; do
  if grep -q '"conditions"' "$c"; then
    PYTHONPATH=src python3 -m ordertta.stage --stage "$c" --validate-only > /dev/null
    echo "stage validate OK: $c"
  else
    PYTHONPATH=src python3 -m ordertta.run --config "$c" --dry-run > /dev/null
    echo "dry-run OK: $c"
  fi
done

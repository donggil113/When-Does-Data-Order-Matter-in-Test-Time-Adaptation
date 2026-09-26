"""Entry point: ``PYTHONPATH=src python -m ordertta.run --config configs/cpu_smoke.json``."""

import sys

from .runner import main

if __name__ == "__main__":
    sys.exit(main())

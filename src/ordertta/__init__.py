"""ordertta: CPU-only instruments for studying data-order dependence in test-time adaptation.

The package deliberately uses only the Python standard library so that the core
checks run in the bare project environment (numpy/torch are not installed there;
see STATUS.md for the dependency blocker).
"""

__version__ = "0.1.0"

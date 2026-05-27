"""
Map_builder.py  (CLI shim)
==========================
Command-line entry-point for map generation.
The actual implementation lives in ``OptiLine.map_builder``.

Run from the project root:
    python tests/Map_builder.py --n 5 --seed 42
    python tests/Map_builder.py --n 1000 --seed 0 --start-index 26
    python tests/Map_builder.py --n 10 --variable-width
    python tests/Map_builder.py --n 20 --maps-dir /path/to/output
"""

import sys
import os

# Allow running directly without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from OptiLine.map_builder import _cli

if __name__ == "__main__":
    _cli()

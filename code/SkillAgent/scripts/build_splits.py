#!/usr/bin/env python3
"""Build the BIRD training split and independent SynSQL validation split."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skillagent.data import build_cli


if __name__ == "__main__":
    build_cli()

#!/usr/bin/env python3
"""Run LunaRad-PEEK validation test suite."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lunarad_peek.validation.tests import run_all_tests

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

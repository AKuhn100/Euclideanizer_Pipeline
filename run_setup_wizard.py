#!/usr/bin/env python3
"""
Setup wizard entry point. Guides from raw data to a converted NPZ file ready for pipeline input.
Requires ANTHROPIC_API_KEY in the environment. See specs/SETUP_WIZARD.md.
"""
from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from src.wizard import main

if __name__ == "__main__":
    main()

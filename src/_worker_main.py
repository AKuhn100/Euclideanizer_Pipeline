"""
Multi-GPU worker entry. Sets CUDA_VISIBLE_DEVICES before importing torch so each
process sees only one GPU, avoiding instability when multiple processes use CUDA.
Implemented as a separate module so the child process can set the environment
before importing the main pipeline.
"""
from __future__ import annotations

import os
import sys


def main(device_id: int, task_list: list, log_path: str, shared_args: dict) -> None:
    # Ensure this process sees only one GPU before any CUDA/torch use.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    # Stagger worker startup slightly to reduce contention when multiple workers start.
    if device_id > 0:
        import time
        time.sleep(device_id * 0.5)
    # Enable faulthandler so low-level crashes produce a traceback (e.g. in pytest output).
    try:
        import faulthandler
        faulthandler.enable(all_threads=True)
    except Exception:
        pass
    # Import run (and thus torch) only after env is set; this process will only see one GPU as cuda:0.
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _pipeline_dir = os.path.dirname(_script_dir)  # pipeline root (parent of src/)
    if _pipeline_dir not in sys.path:
        sys.path.insert(0, _pipeline_dir)
    import run as run_mod
    run_mod._worker(0, task_list, log_path, shared_args)

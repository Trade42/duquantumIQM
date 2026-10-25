"""Run the complete preserved experiment audit in its original relative paths."""
from pathlib import Path
import os
import runpy
import sys

if __name__ == "__main__":
    archive = Path(__file__).resolve().parent
    sys.path.insert(0, str(archive.parent))
    os.chdir(archive)
    runpy.run_module("challenge.replay_rounds", run_name="__main__")

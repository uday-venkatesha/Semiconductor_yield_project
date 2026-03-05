"""
main.py — Pipeline Orchestrator
────────────────────────────────
Runs all four Python phases in sequence.  Each phase saves its outputs to
data/artifacts/ so the next phase can load them independently.

Usage:
    python main.py
"""

import subprocess
import sys
import os

def run_script(script_path: str) -> None:
    print(f"─── Running {script_path} {'─' * (50 - len(script_path))}")
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,   # stream stdout/stderr directly to terminal
    )
    if result.returncode != 0:
        print(f"\n❌  Error in {script_path}. Pipeline halted.")
        sys.exit(1)
    print(f"✅  {script_path} completed\n")

if __name__ == "__main__":
    print("🚀  Predictive Quality & Yield Optimization Pipeline — Starting...\n")

    # Ensure all required directories exist before any script runs
    for directory in ['data/raw', 'data/artifacts', 'data/db']:
        os.makedirs(directory, exist_ok=True)

    scripts = [
        "src/01_preprocess.py",
        "src/02_model_training.py",
        "src/03_root_cause.py",
        "src/04_database_pipeline.py",
    ]

    for script in scripts:
        run_script(script)

    print("🎉  Pipeline complete!")
    print("    SQLite database : data/db/manufacturing_yield.db")
    print("    Artifacts       : data/artifacts/")
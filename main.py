import subprocess
import sys
import os

def run_script(script_path):
    print(f"--- Running {script_path} ---")
    # sys.executable ensures it uses your active virtual environment's Python
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ Error executing {script_path}. Pipeline halted.")
        sys.exit(1)
    print(f"✅ Successfully completed {script_path}\n")

if __name__ == "__main__":
    print("🚀 Starting Predictive Quality & Yield Optimization Pipeline...\n")
    
    # Ensure the database directory exists before running the pipeline
    os.makedirs("data/db", exist_ok=True)
    
    scripts_to_run = [
        "src/01_preprocess.py",
        "src/02_model_training.py",
        "src/03_root_cause.py",
        "src/04_database_pipeline.py"
    ]
    
    for script in scripts_to_run:
        run_script(script)
        
    print("🎉 Pipeline executed successfully! The SQLite database is ready at data/db/manufacturing_yield.db")
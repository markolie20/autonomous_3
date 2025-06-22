# (autonomous_3/run_fast_test.py)
import sys
from pathlib import Path

# Add project root to sys.path to allow `from src import ...`
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src import main as warlords_main_runner

if __name__ == "__main__":
    print("--- Starting FAST TEST run for Warlords ---")
    warlords_main_runner.run_experiments(config_module_name="src.config_fast_test")
    print("--- FAST TEST run for Warlords Finished ---")
"""Run the whole demo end-to-end (press Enter between steps)."""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).parent
STEPS = ["step1_ingest.py", "step2_train_and_track.py",
         "step3_query_and_promote.py", "step4_serve_and_monitor.py"]

auto = "--no-pause" in sys.argv
sys.path.insert(0, str(HERE))
for i, script in enumerate(STEPS):
    if i and not auto:
        input("\n[Enter] next step...")
    runpy.run_path(str(HERE / script), run_name="__main__")

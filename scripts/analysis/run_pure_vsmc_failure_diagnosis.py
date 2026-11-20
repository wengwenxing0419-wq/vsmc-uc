import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.pure_vsmc_failure_lib import run_all
run_all()

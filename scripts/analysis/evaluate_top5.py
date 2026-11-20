import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.pure_vsmc_failure_lib import discover_experiments, evaluate_top5
evaluate_top5(discover_experiments())

"""
Minimal demonstration of GEPA + GenericOpenEvolveAdapter
========================================================

This script shows how to optimise a Python strategy for OpenEvolve’s
*Cant-Be-Late* task **without** writing a custom dataset loader.  The
`GenericOpenEvolveAdapter` wraps the evaluator located at:

    openevolve/examples/ADRS/cant-be-late/evaluator.py

The adapter ignores the ``batch`` parameter, so we pass a single ``None`` as
the “training” and “validation” sets.

Run:

    python -m gepa.adapters.generic_openevolve_adapter.example

Environment variables
---------------------
The OpenEvolve evaluator and the adapter will respect any API-key settings you
place in *openevolve/config.yaml* (e.g., ``${OPENAI_API_KEY}``).  No further
configuration is required.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

from gepa import optimize
from gepa.adapters.generic_openevolve_adapter import GenericOpenEvolveAdapter

INITIAL_PROGRAM = """
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class AlwaysOnDemandStrategy(Strategy):
    \"\"\"Extremely simple baseline: run on ON_DEMAND the whole time.\"\"\"

    NAME = "always_on_demand"

    def __init__(self, args):
        super().__init__(args)

    def reset(self, env, task):
        super().reset(env, task)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Guarantee deadline by always choosing expensive but reliable instances.
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)
""".lstrip()

PROJECT_ROOT = Path(__file__).resolve().parents[5]
PARENT = Path(__file__).resolve().parent
EVALUATION_FILE = PARENT / "openevolve" / "evaluator.py"

RUN_DIR = (
    Path(os.getenv("GEPA_RUN_DIR"))
    if os.getenv("GEPA_RUN_DIR")
    else PROJECT_ROOT / "gepa" / "runs" / "generic_openevolve" / datetime.now().strftime("%Y%m%d-%H%M%S")
)
RUN_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Saving artifacts to: {RUN_DIR}")

adapter = GenericOpenEvolveAdapter(evaluation_file=str(EVALUATION_FILE))

DUMMY_BATCH: List[None] = [None]

result = optimize(
    seed_candidate={"program": INITIAL_PROGRAM},
    trainset=DUMMY_BATCH,
    valset=DUMMY_BATCH,
    adapter=adapter,
    # Reflection LM is embedded in the adapter – GEPA doesn’t need its own.
    max_metric_calls=int(os.getenv("GEPA_MAX_METRIC_CALLS", "10")),
    run_dir=str(RUN_DIR),
    display_progress_bar=True
)

best = result.best_candidate
print("\n===== BEST PROGRAM =====\n")
print(best["program"])
print("\n===== METRICS =====")
print(json.dumps(result.to_dict(), indent=2))

# Save best program & metadata
(best_path := RUN_DIR / "best_program.py").write_text(best["program"], encoding="utf-8")
(RUN_DIR / "result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
print(f"[INFO] Results written to {RUN_DIR}")

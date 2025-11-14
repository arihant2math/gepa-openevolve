from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List
import tempfile

from gepa import optimize
from generic_evolve_adapter import EvolveAdapter

INITIAL_PROGRAM = Path(__file__).resolve().parent / "openevolve" / "initial_greedy.py"
adapter = EvolveAdapter(path=Path(__file__).resolve().parent / "openevolve")

DUMMY_BATCH: List[None] = [None]
RUN_DIR = Path(tempfile.mkdtemp())

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

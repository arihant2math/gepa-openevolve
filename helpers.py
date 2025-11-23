import datetime
import dataclasses
import os
import json
from pathlib import Path
from typing import Optional, Callable


DEFAULT_TRACE_RATIO = 0.30

def _resolve_run_dir_factory(name: str) -> Callable[Path]:
    def _resolve_run_dir():
        run_dir_env = os.environ.get("GEPA_RUN_DIR")
        if run_dir_env:
            run_dir = Path(run_dir_env)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir = Path("runs") / name / timestamp
    
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    return _resolve_run_dir


def _write_checkpoints(
    run_dir: Path,
    gepa_result,
    base_score: Optional[float],
    optimized_score: Optional[float],
    best_candidate: dict[str, str],
):
    # Serialize the full GEPA result for later inspection
    result_path = run_dir / "gepa_result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(gepa_result.to_dict(), f, indent=2)

    # Write the best program as a Python file
    best_program_path = run_dir / "best_program.py"
    best_program_path.write_text(best_candidate["program"], encoding="utf-8")

    # Record test metrics for quick reference
    metrics_path = run_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_test_score": base_score,
                "optimized_test_score": optimized_score,
                "best_candidate_index": gepa_result.best_idx,
            },
            f,
            indent=2,
        )

    # Snapshot every candidate for manual analysis
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(exist_ok=True)
    for idx, candidate in enumerate(gepa_result.candidates):
        program_path = candidates_dir / f"candidate_{idx:03d}.py"
        program_path.write_text(candidate["program"], encoding="utf-8")


@dataclasses.dataclass
class Config:
    max_traces: Optional[int] = None
    max_metric_calls: int = 20
    trace_ratio: float = DEFAULT_TRACE_RATIO
    skip_test: bool = False
 
 
def get_config() -> Config:
    config = Config()
    config.max_traces = int(os.environ.get("GEPA_MAX_TRACES")) if os.environ.get("GEPA_MAX_TRACES") else None
    config.max_metric_calls = int(os.environ.get("GEPA_MAX_METRIC_CALLS", "20"))
    config.trace_ratio = float(os.environ.get("GEPA_TRACE_RATIO", DEFAULT_TRACE_RATIO))
    config.skip_test = os.environ.get("GEPA_SKIP_TEST", "0") == "1"
    return config

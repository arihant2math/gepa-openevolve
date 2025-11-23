from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any
import tempfile
import math

from gepa import optimize
from gepa import EvaluationBatch
from generic_evolve_adapter import EvolveAdapter

from helpers import DEFAULT_TRACE_RATIO


TRACE_SAMPLE_IDS: list[int] = [
    0,
    8,
    9,
    20,
    21,
    33,
    42,
    51,
    61,
    70,
    99,
    107,
    117,
    126,
    135,
    145,
    154,
    163,
    172,
    182,
    191,
    219,
    228,
    238,
    247,
    256,
    266,
    275,
    284,
    294,
]

# Overheads that have been extracted in the reference archive.
TRACE_OVERHEADS: list[float] = [0.02, 0.20, 0.40]


def _is_random_start_trace(path: Path) -> bool:
    return path.is_file() and path.suffix == ".json" and "traces" in path.parts


def _list_all_traces(root: Path) -> list[str]:
    trace_paths = {
        str(path.resolve())
        for path in root.glob("**/traces/random_start/*.json")
        if _is_random_start_trace(path)
    }
    if not trace_paths:
        raise FileNotFoundError(f"No trace files found under {root}")
    return sorted(trace_paths)


def _list_sample_traces(root: Path, sample_ids: list[int]) -> list[str]:
    id_set = {str(i) for i in sample_ids}
    trace_paths: set[str] = set()

    for overhead in TRACE_OVERHEADS:
        overhead_root = root / f"ddl=search+task=48+overhead={overhead:.2f}" / "real"
        if not overhead_root.exists():
            continue
        for trace_path in overhead_root.glob("*/traces/random_start/*.json"):
            if not _is_random_start_trace(trace_path):
                continue
            if trace_path.stem in id_set:
                trace_paths.add(str(trace_path.resolve()))

    if not trace_paths:
        # Fallback: filter across the entire archive by trace ID
        trace_paths = {
            str(path.resolve())
            for path in root.glob("**/traces/random_start/*.json")
            if _is_random_start_trace(path) and path.stem in id_set
        }

    return sorted(trace_paths)

def load_trace_dataset(
    dataset_root: str,
    split_config=None,
    seed: int = 0,
    max_traces_per_split: int | None = None,
    trace_ratio: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build train/val/test splits from an extracted cant-be-late trace archive.

    Train/val share a fixed set of trace IDs (30 by default) covering all
    overheads/environments. Test uses the full archive unless
    ``max_traces_per_split`` is provided (useful for smoke tests).
    """

    del split_config  # Unused; retained for backward compatibility
    del seed  # Deterministic sampling based on TRACE_SAMPLE_IDS

    root_path = Path(dataset_root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset root {dataset_root} does not exist")

    sample_ids = list(TRACE_SAMPLE_IDS)

    ratio = DEFAULT_TRACE_RATIO if trace_ratio is None else trace_ratio
    ratio = min(1.0, max(ratio, DEFAULT_TRACE_RATIO))
    desired_count = max(1, math.ceil(len(sample_ids) * ratio))
    sample_ids = sample_ids[:desired_count]

    if max_traces_per_split is not None:
        sample_ids = sample_ids[:max_traces_per_split]

    sample_traces = _list_sample_traces(root_path, sample_ids)
    if not sample_traces:
        raise FileNotFoundError("No traces found for the sampled trace IDs")

    all_traces = _list_all_traces(root_path)
    if max_traces_per_split is not None or ratio < 1.0:
        test_traces = _list_sample_traces(root_path, sample_ids)
    else:
        test_traces = all_traces

    return {
        "train": [{"trace_files": sample_traces}]
        if sample_traces
        else [],
        "val": [{"trace_files": sample_traces}]
        if sample_traces
        else [],
        "test": [{"trace_files": test_traces}]
        if test_traces
        else [],
    }

DATASET_ROOT = Path(__file__).resolve().parent / "cant_be_late_data" / "exp" / "real"

OPENEVOLVE_ROOT = Path(__file__).resolve().parent / "cant_be_late"
INITIAL_PROGRAM = open(OPENEVOLVE_ROOT / "initial_greedy.py", "r").read()

def output_extractor(eval_out):
    trace_cost_json = eval_out.artifacts["trace_costs_json"]
    trace_cost = json.loads(trace_cost_json)
    scores = []
    for key, results in trace_cost.items():
        for result in results:
            scores.append(result["cost"])
    return EvaluationBatch(scores=scores, outputs=trace_cost, trajectories=eval_out.artifacts)

def reflect(batch: EvaluationBatch) -> list:
    # TODO: Fix valset stuff first
    return []

adapter = EvolveAdapter(path=OPENEVOLVE_ROOT, output_extractor=output_extractor, reflect=reflect)

RUN_DIR = Path(tempfile.mkdtemp())


def load_dataset(
    max_traces_per_split: int | None = None,
    trace_ratio: float | None = None,
    include_test: bool = True,
):
    """Load train/val/test splits from extracted from traces."""

    splits = load_trace_dataset(
        dataset_root=str(DATASET_ROOT),
        max_traces_per_split=max_traces_per_split,
        trace_ratio=trace_ratio,
    )
    train_set = splits["train"]
    val_set = splits["val"]
    test_set = splits["test"] if include_test else []
    return train_set, val_set, test_set

def _resolve_run_dir() -> Path:
    import os

    run_dir_env = os.environ.get("GEPA_RUN_DIR")
    if run_dir_env:
        run_dir = Path(run_dir_env)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path("runs") / "cant_be_late" / timestamp

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_checkpoints(
    run_dir: Path,
    gepa_result,
    base_score: Optional[float],
    optimized_score: Optional[float],
    best_candidate: dict[str, str],
):
    max_traces_env = os.environ.get("GEPA_MAX_TRACES")
    max_traces = int(max_traces_env) if max_traces_env else None
    max_metric_calls_env = os.environ.get("GEPA_MAX_METRIC_CALLS")
    max_metric_calls = int(max_metric_calls_env) if max_metric_calls_env else 20
    trace_ratio_env = os.environ.get("GEPA_MAX_TRACES")
    trace_ratio = DEFAULT_TRACE_RATIO
    if trace_ratio_env:
        try:
            trace_ratio = float(trace_ratio_env)
        except ValueError:
            print(
                f"Invalid GEPA_MAX_TRACES='{trace_ratio_env}', falling back to {DEFAULT_TRACE_RATIO:.2f}",
                flush=True,
            )
    trace_ratio = min(1.0, max(trace_ratio, DEFAULT_TRACE_RATIO))
    skip_test = os.environ.get("GEPA_SKIP_TEST", "0") == "1"

    run_dir = _resolve_run_dir()
    print(f"GEPA artifacts will be saved to: {run_dir}")
    print(f"Using {trace_ratio:.0%} of traces for train/val evaluation", flush=True)
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

if __name__ == "__main__":
    import os

    max_traces_env = os.environ.get("GEPA_MAX_TRACES")
    max_traces = int(max_traces_env) if max_traces_env else None
    max_metric_calls_env = os.environ.get("GEPA_MAX_METRIC_CALLS")
    max_metric_calls = int(max_metric_calls_env) if max_metric_calls_env else 20
    trace_ratio_env = os.environ.get("GEPA_TRACE_RATIO")
    trace_ratio = DEFAULT_TRACE_RATIO
    if trace_ratio_env:
        try:
            trace_ratio = float(trace_ratio_env)
        except ValueError:
            print(
                f"Invalid GEPA_TRACE_RATIO='{trace_ratio_env}', falling back to {DEFAULT_TRACE_RATIO:.2f}",
                flush=True,
            )
    trace_ratio = min(1.0, max(trace_ratio, DEFAULT_TRACE_RATIO))
    skip_test = os.environ.get("GEPA_SKIP_TEST", "0") == "1"

    run_dir = _resolve_run_dir()
    # Load from train and test set
    train_set, val_set, test_set = load_dataset(
        max_traces_per_split=max_traces,
        trace_ratio=trace_ratio,
        include_test=not skip_test,
    )
    
    if skip_test:
        base_score: Optional[float] = None
        print("Base program score: skipped (GEPA_SKIP_TEST=1)")
    else:
        output_base = adapter.evaluate(test_set, {"program": INITIAL_PROGRAM})
        base_score = sum(output_base.scores)
        print(f"Base program score: {base_score}")

    gepa_result = optimize(
        seed_candidate={"program": INITIAL_PROGRAM},
        trainset=train_set,
        valset=val_set,
        adapter=adapter,
        max_metric_calls=int(os.getenv("GEPA_MAX_METRIC_CALLS", "10")),
        run_dir=str(RUN_DIR),
        display_progress_bar=True
    )
    best_candidate = gepa_result.best_candidate
    print(f"Best program from optimization: {best_candidate['program']}")

    if skip_test:
        optimized_score: Optional[float] = None
        print("Optimized program score: skipped (GEPA_SKIP_TEST=1)")
    else:
        output_optimized = adapter.evaluate(test_set, best_candidate)
        optimized_score = sum(output_optimized.scores)
        print(f"Optimized program score: {optimized_score}")

    _write_checkpoints(run_dir, gepa_result, base_score, optimized_score, best_candidate)
    print(f"Checkpoint artifacts written under {run_dir}")

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Optional, Sequence

import yaml
from gepa import EvaluationBatch, GEPAAdapter
from openevolve.evaluation_result import EvaluationResult
from openevolve_proposal_signature import (
    OpenEvolveProposalSignature,
)


def _process_evaluation_result(result: Any) -> EvaluationResult:
    """
    Process evaluation result to handle both dict and EvaluationResult returns

    Args:
        result: Raw result from evaluation function

    Returns:
        EvaluationResult instance
    """
    if isinstance(result, dict):
        # Backward compatibility - wrap dict in EvaluationResult
        return EvaluationResult.from_dict(result)
    elif isinstance(result, EvaluationResult):
        # New format - use directly
        return result
    else:
        # Error case - return error metrics
        logging.warning(f"Unexpected evaluation result type: {type(result)}")
        return EvaluationResult(metrics={"error": 0.0})


def _passes_threshold(metrics: Dict[str, float], threshold: float) -> bool:
    """
    Check if metrics pass a threshold

    Uses 'combined_score' if available (for consistency with evolution),
    otherwise falls back to averaging all numeric metrics except 'error'

    Args:
        metrics: Dictionary of metric name to score
        threshold: Threshold to pass

    Returns:
        True if metrics pass threshold
    """
    if not metrics:
        return False

    # Use combined_score if available - this is what evolution uses
    if "combined_score" in metrics:
        score = metrics.get("combined_score")
        if isinstance(score, (int, float)):
            return float(score) >= threshold

    # Fallback: average all numeric metrics except 'error'
    # This maintains backward compatibility
    valid_metrics = []
    for name, value in metrics.items():
        # Skip 'error' keys and ensure values are numeric
        if name != "error" and isinstance(value, (int, float)):
            try:
                valid_metrics.append(float(value))
            except (TypeError, ValueError):
                logging.warning(f"Skipping non-numeric metric: {name}={value}")
                continue

    if not valid_metrics:
        return False

    avg_score = sum(valid_metrics) / len(valid_metrics)
    return avg_score >= threshold


class EvaluationStrategy:
    def evaluate(self, program_path: str) -> list:
        raise NotImplementedError


class DefaultEvaluationStrategy(EvaluationStrategy):
    def __init__(self, path: Path):
        self.path = path
        spec = importlib.util.spec_from_file_location("evaluator", self.path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.path.stem] = module
        spec.loader.exec_module(module)
        if not hasattr(module, "evaluate"):
            raise AttributeError(f"evaluate function not found in {self.path}")
        self.module = getattr(module, "evaluate")

    def evaluate(self, program_path: str) -> list:
        try:
            result = self.module(program_path)
            eval_result = _process_evaluation_result(result)
        except Exception as e:
            logging.error(f"Error evaluating {program_path}: {e}")
            error_context = {}
            return EvaluationResult(
                metrics={"passed": 0.0, "error": 0.0},
                artifacts={
                    "stderr": str(e),
                    "traceback": traceback.format_exc(),
                    **error_context,
                },
            )
        return eval_result


class CascadeEvaluationStrategy(EvaluationStrategy):
    def __init__(self, path: Path, cascade_thresholds: list[float]):
        self.path = path
        self.stages = self.load_stages()
        self.cascade_thresholds = cascade_thresholds

    def load_stages(self) -> dict[str, Callable]:
        required_stages = ["evaluate_stage1"]
        possible_stages = ["evaluate_stage2", "evaluate_stage3"]
        stages = {}
        # import
        spec = importlib.util.spec_from_file_location("evaluator", self.path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.path.stem] = module
        spec.loader.exec_module(module)
        for stage in required_stages:
            if hasattr(module, stage):
                stages[stage] = getattr(module, stage)
            else:
                raise AttributeError(f"Required stage {stage} not found in {self.path}")
        for stage in possible_stages:
            if hasattr(module, stage):
                stages[stage] = getattr(module, stage)
        return stages

    def evaluate(self, program_path: str) -> EvaluationResult:
        # TODO: implement cascading logic based on thresholds
        # TODO: use asyncio to do run with timeout like openevolve
        try:
            stage1 = self.stages["evaluate_stage1"]
            stage1_result = stage1(program_path)
            stage1_eval_result = _process_evaluation_result(stage1_result)
        except Exception as e:
            logging.error(f"Error in stage 1 evaluation: {str(e)}")
            # Capture stage 1 failure with enhanced context
            # TODO: Add error context
            error_context = {}
            return EvaluationResult(
                metrics={"stage1_passed": 0.0, "error": 0.0},
                artifacts={
                    "stderr": str(e),
                    "traceback": traceback.format_exc(),
                    **error_context,
                },
            )

        # Check threshold
        if not _passes_threshold(
            stage1_eval_result.metrics, self.cascade_thresholds[0]
        ):
            logging.warning(f"Stage 1 evaluation failed to meet threshold")
            # TODO: Should we return?

        if not "evaluate_stage2" in self.stages:
            logging.warning(f"Stage 2 evaluation not configured")
            return stage1_eval_result

        try:
            stage2 = self.stages["evaluate_stage2"]
            stage2_result = stage2(program_path)
            stage2_eval_result = _process_evaluation_result(stage2_result)
        except Exception as e:
            logging.error(f"Error in stage 2 evaluation: {str(e)}")
            # Capture stage 2 failure with enhanced context
            # TODO: Add error context
            error_context = {}
            stage1_eval_result.metrics["stage2_passed"] = 0.0
            return stage1_eval_result

        merged_metrics = {}
        # Convert all values to float to avoid type errors
        for name, value in stage1_eval_result.metrics.items():
            if isinstance(value, (int, float)) and name != "error":
                merged_metrics[name] = float(value)

        for name, value in stage2_eval_result.metrics.items():
            if isinstance(value, (int, float)) and name != "error":
                merged_metrics[name] = float(value)

        # Merge artifacts
        merged_artifacts = {}
        merged_artifacts.update(stage1_eval_result.artifacts)
        merged_artifacts.update(stage2_eval_result.artifacts)

        merged_result = EvaluationResult(
            metrics=merged_metrics, artifacts=merged_artifacts
        )

        # Check threshold for stage 3
        if len(self.cascade_thresholds) < 2 or not _passes_threshold(
            merged_result.metrics, self.cascade_thresholds[1]
        ):
            return merged_result

        # Stage 3
        if not "evaluate_stage3" in self.stages:
            return merged_result

        try:
            stage3 = self.stages["evaluate_stage3"]
            stage3_result = stage3(program_path)
            stage3_eval_result = _process_evaluation_result(stage3_result)
        except Exception as e:
            logging.error(f"Error in stage 3: {e}")
            # Capture stage 3 failure, but keep previous results
            merged_result.artifacts.update(
                {
                    "stage3_stderr": str(e),
                    "stage3_traceback": traceback.format_exc(),
                    "failure_stage": "stage3",
                }
            )
            merged_result.metrics["stage3_passed"] = 0.0
            return merged_result

        # Merge stage 3 results
        for name, value in stage3_eval_result.metrics.items():
            if isinstance(value, (int, float)) and name != "error":
                merged_result.metrics[name] = float(value)

        merged_result.artifacts.update(stage3_eval_result.artifacts)

        # Merge stage 3 results
        for name, value in stage3_eval_result.metrics.items():
            if isinstance(value, (int, float)) and name != "error":
                merged_result.metrics[name] = float(value)

        merged_result.artifacts.update(stage3_eval_result.artifacts)

        return merged_result


class EvolveAdapter(GEPAAdapter):
    def __init__(
        self,
        path: Path,
        output_extractor: Callable[EvaluationResult, EvaluationBatch],
        reflect: Callable[EvaluationBatch, dict],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.path = path
        self.config = yaml.safe_load(open(path / "config.yaml", "r"))
        self.cascade = self.config["evaluator"].get("cascade_evaluation", False)
        self.evaluator_path = path / "evaluator.py"
        self.temp_env_path = Path(tempfile.mkdtemp())
        self.output_extractor = output_extractor
        self.reflect = reflect
        self.evaluation_strategy = (
            CascadeEvaluationStrategy(
                self.evaluator_path, self.config["evaluator"]["cascade_thresholds"]
            )
            if self.cascade
            else DefaultEvaluationStrategy(self.evaluator_path)
        )

        # If caller did not explicitly set the number of workers, inherit it from the
        # configuration (falls back to the evaluator.parallel_evaluations field).
        max_litellm_workers = self.config["evaluator"]["parallel_evaluations"]

        # Pick the model with the highest weight (first if already sorted)
        if "models" in self.config["llm"]:
            primary = sorted(
                self.config["llm"]["models"], key=lambda m: m["weight"], reverse=True
            )[0]
            model_name = primary["name"]
            api_key = primary["api_key"] or self.config["llm"]["api_key"]
            api_base = primary["api_base"] or self.config["llm"]["api_base"]
        else:
            # Sensible fall-back
            model_name = (
                getattr(self.config["llm"], "primary_model", None) or "gpt-3.5-turbo"
            )
            api_key = self.config["llm"]["api_key"]
            api_base = self.config["llm"]["api_base"]

        import litellm  # type: ignore

        self.litellm = litellm

        def _call_lm(prompt: str) -> str:
            completion = self.litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                base_url=api_base,
                temperature=primary.temperature,
                top_p=primary.top_p,
                max_tokens=primary.max_tokens,
            )
            return completion.choices[0].message.content or ""

        self.reflection_lm = _call_lm

    def evaluate(
        self,
        batch: list,
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        # candidate = {'code': '# Evolve-Block -Start ... # Evolve-Block end'}
        # write the code to a temporary file
        tmp_code_path = self.temp_env_path / "temp_code.py"
        # Delete file if it exists
        if tmp_code_path.exists() and tmp_code_path.is_file():
            tmp_code_path.unlink()
        elif tmp_code_path.exists():
            tmp_code_path.rmdir()
        with open(tmp_code_path, "w") as f:
            f.write(candidate["program"])
        # run the code
        # run the evaluate method with the temporary file
        eval_out = self.evaluation_strategy.evaluate(str(tmp_code_path))
        output = self.output_extractor(eval_out)
        return {"program": output}

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if "program" not in components_to_update:
            return {}

        dataset = self.reflect(eval_batch)
        return dataset

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        new_texts: dict[str, str] = {}
        for name in components_to_update:
            base_instruction = candidate[name]
            dataset_with_feedback = reflective_dataset.get(name, [])

            new_texts[name] = OpenEvolveProposalSignature.run(
                lm=self.reflection_lm,
                input_dict={
                    "current_instruction_doc": base_instruction,
                    "dataset_with_feedback": dataset_with_feedback,
                },
            )["new_program"]

        return new_texts

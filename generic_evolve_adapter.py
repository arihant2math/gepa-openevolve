from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
from types import ModuleType
from typing import Any, Callable, Optional, TypedDict
from pathlib import Path
import importlib

from gepa import EvaluationBatch, GEPAAdapter
import yaml

class EvolveDataInst(TypedDict):
    """Input data instance for evolution tasks."""
    problem_description: str
    additional_context: dict[str, Any]

class EvolveTrajectory(TypedDict):
    data: EvolveDataInst
    evaluation_result: dict[str, Any]
    program_path: str

class EvolveRolloutOutput(TypedDict):
    metrics: dict[str, float]
    artifacts: dict[str, Any]
    program_path: str

class EvaluationStrategy:
    def evaluate(self, program_path: str) -> list:
        raise NotImplementedError

class DefaultEvaluationStrategy(EvaluationStrategy):
    def __init__(self, evaluator_path: Path, temp_env_path: Path):
        self.evaluator_path = evaluator_path
        self.temp_env_path = temp_env_path
        self.evaluate_fn = self._load_evaluate_function()
    
    def _load_evaluate_function(self) -> Callable[[str], dict[str, Any]]:
        spec = importlib.util.spec_from_file_location(
            "evaluator", self.evaluator_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load evaluator from {self.evaluator_path}")
        
        module = importlib.util.module_from_spec(spec)
        
        original_path = sys.path.copy()
        try:
            sys.path.insert(0, str(self.temp_env_path))
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = original_path
    
        if not hasattr(module, "evaluate"):
            raise AttributeError(f"evaluate function not found in {self.path}")
        return getattr(module, "evaluate")

    def evaluate(self, program_path: str) -> dict[str, Any]:
        """Evaluate the program using the loaded evaluate function."""
        result = self.evaluate_fn(program_path)
        return result

class CascadeEvaluationStrategy(EvaluationStrategy):
    """Cascade evaluation strategy using multiple evaluation stages."""
    
    def __init__(
        self, 
        evaluator_path: Path, 
        temp_env_path: Path,
        cascade_thresholds: list[float]
    ):
        self.evaluator_path = evaluator_path
        self.temp_env_path = temp_env_path
        self.stages = self.load_stages()
        self.cascade_thresholds = cascade_thresholds

    def _load_stages(self) -> dict[str, Callable]:
        required_stages = ["evaluate_stage1"]
        optional_stages = ["evaluate_stage2", "evaluate_stage3"]
        stages = {}

        spec = importlib.util.spec_from_file_location(
            "evaluator", self.evaluator_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load evaluator from {self.evaluator_path}")
        
        module = importlib.util.module_from_spec(spec)
        original_path = sys.path.copy()
        try:
            sys.path.insert(0, str(self.temp_env_path))
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = original_path
        
        for stage in required_stages:
            if hasattr(module, stage):
                stages[stage] = getattr(module, stage)
            else:
                raise AttributeError(f"Required stage {stage} not found in {self.evaluator_path}")
        
        for stage in optional_stages:
            if hasattr(module, stage):
                stages[stage] = getattr(module, stage)
        return stages

    def evaluate(self, program_path: str) -> dict[str, Any]:
        """Evaluate in cascading stages using the thresholds."""
        # TODO: Explore more creative methods for combining the results from the cascading stages.  
        result_stage1 = self.stages["evaluate_stage1"](program_path)
       
        # Merge results from all stages that pass thresholds
        merged_result = result_stage1.copy()
        score = result_stage1.get("combined_score", 0.0)
        
        # Run stage 2 if threshold is met
        if "evaluate_stage2" in self.stages:
            if score >= self.cascade_thresholds[0]:
                result_stage2 = self.stages["evaluate_stage2"](program_path)
                merged_result.update(result_stage2)
                score = merged_result.get("combined_score", score)
        
        # Run stage 3 if threshold is met
        if "evaluate_stage3" in self.stages:
            if score >= self.cascade_thresholds[1]:
                result_stage3 = self.stages["evaluate_stage3"](program_path)
                merged_result.update(result_stage3)
                
        return merged_result

class EvolveAdapter(GEPAAdapter):
    def __init__(
        self, 
        path: Path, 
        failure_score: float = 0.0,
    ):   
        self.path = Path(path)
        self.failure_score = failure_score

        # Load configuration
        config_path = self.path / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Load evaluator
        self.evaluator_path = self.path / "evaluator.py"
        if not self.evaluator_path.exists():
            raise FileNotFoundError(f"Evaluator file not found at {self.evaluator_path}")
        
        # Set up temporary environment
        self.temp_env_path = Path(tempfile.mkdtemp(prefix="evolve_adapter_"))
        self._setup_temp_environment()

       # Initialize evaluation strategy
        cascade_evaluation = self.config.get("evaluator", {}).get("cascade_evaluation", False)
        if cascade_evaluation:
            cascade_thresholds = self.config.get("evaluator", {}).get("cascade_thresholds", [0.5, 0.75])
            self.evaluation_strategy = CascadeEvaluationStrategy(
                self.evaluator_path, self.temp_env_path, cascade_thresholds
            )
        else:
            self.evaluation_strategy = DefaultEvaluationStrategy(self.evaluator_path, self.temp_env_path)

    def _setup_temp_environment(self):
        # Initialize uv environment
        result = os.system(f"uv init .", cwd=self.temp_env_path)
        if result != 0:
            raise RuntimeError(f"Failed to initialize uv environment at {self.temp_env_path}")
        
        # Install requirements if they exist
        requirements_path = self.path / "requirements.txt"
        if requirements_path.exists():
            result = os.system(
                f"uv pip install -r {requirements_path}",
                cwd=self.temp_env_path
            )
            if result != 0:
                raise RuntimeError(
                    f"Failed to install requirements from {requirements_path}"
                )
    def __del__(self):
        """Clean up temporary environment on deletion."""
        if hasattr(self, "temp_env_path") and self.temp_env_path.exists():
            shutil.rmtree(self.temp_env_path, ignore_errors=True)
        
    def evaluate(
        self,
        batch,
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[EvolveTrajectory, EvolveRolloutOutput]:
        outputs: list[EvolveRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[EvolveTrajectory] | None = [] if capture_traces else None
        # TODO: Current implementation is most likely not correct fundamentally
        # Elephant in the room is what do do about batch, since GEPAAdapter is 
        # supposed to return EvaluationBatch containing fine-grained per-data-instance scores

        # Write candidate code to temporary file
        temp_code_path = "temp_code.py"
        with open(temp_code_path, "w") as f:
            f.write(candidate["code"])
        
        # Run evaluation
        for data in batch:
            try:
                # TODO: Handle the different data types eval_result could be (dict, EvaluationResult)
                eval_result = self.evaluation_strategy.evaluate(temp_code_path)
                
                # Extract metrics and artifacts
                metrics = eval_result.get("metrics", eval_result)
                artifacts = eval_result.get("artifacts", {})
                combined_score = eval_result.get("combined_score", 0.0)
                
                # Create output
                output = {
                    "metrics": metrics,
                    "artifacts": artifacts,
                    "program_path": temp_code_path,
                }
                outputs.append(output)
                scores.append(combined_score)

                if capture_traces:
                    trajectory = {
                        "data": data,
                        "evaluation_result": eval_result,
                        "program_path": temp_code_path,
                    }
                    trajectories.append(trajectory)

            except Exception as e:
                # Handle evaluation failure
                output = {
                    "metrics": {"combined_score": 0.0},
                    "artifacts": {"error": str(e)},
                    "program_path": temp_code_path,
                }
                outputs.append(output)
                scores.append(self.failure_score)
                
                if capture_traces:
                    trajectory = {
                        "data": data,
                        "evaluation_result": {"error": str(e)},
                        "program_path": temp_code_path,
                    }
                    trajectories.append(trajectory)
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[EvolveTrajectory, EvolveRolloutOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        # TODO: 
        ret_d: dict[str, list[dict[str, Any]]] = {}
        items: list[dict[str, Any]] = []
        
        enable_artifacts = self.config.get("evaluator", {}).get("enable_artifacts", False)
        
        if not enable_artifacts:
            # TODO: Figure out default behavior when enable_artifacts is not enabled, as
            # well as what else OpenEvolve uses that should be passed into propose_new_texts
            return ret_d
        
        # When enable_artifacts is True, use artifacts
        for component in components_to_update:
            items: list[dict[str, Any]] = []
            for trajectory, output, score in zip(
                eval_batch.trajectories, 
                eval_batch.outputs, 
                eval_batch.scores
            ):
                # Use artifacts as feedback
                artifacts = output.get("artifacts", {})
                items.append(artifacts)
            ret_d[component] = items
        
        return ret_d

    def propose_new_texts(self, candidate: dict, inputs: list, trajectories: list) -> dict:
        return super().propose_new_texts(candidate, inputs, trajectories)
        # Use the llm config from config.yaml to propose new texts
        # Use the prompt.system_prompt from config.yaml to propose new texts

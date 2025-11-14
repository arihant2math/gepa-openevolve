from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
from types import ModuleType
from typing import Any, Callable, Optional
from pathlib import Path
import importlib
import subprocess

from gepa import EvaluationBatch, GEPAAdapter
import yaml

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
        return self.module.evaluate(program_path)

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

    def evaluate(self, program_path: str) -> list:
        stage1 = self.stages["evaluate_stage1"]
        merged_results = stage1(program_path)
        # TODO: implement cascading logic based on thresholds

class EvolveAdapter(GEPAAdapter):
    def __init__(self, path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path
        self.config = yaml.safe_load(open(path / "config.yaml", "r"))
        self.cascade = self.config["evaluator"].get("cascade_evaluation", False)
        self.evaluator_path = path / "evaluator.py"
        self.temp_env_path = Path(tempfile.mkdtemp())
        
        self.evaluation_strategy = CascadeEvaluationStrategy(self.evaluator_path, self.config["evaluator"]["cascade_thresholds"]) if self.cascade else DefaultEvaluationStrategy(self.evaluator_path)

    def evaluate(self, candidate: dict, inputs: list) -> list:
        # candidate = {'code': '# Evolve-Block -Start ... # Evolve-Block end'}
        # write the code to a temporary file
        with open("temp_code.py", "w") as f:
            f.write(candidate['code'])
        # run the code
        # run the evaluate method with the temporary file
        eval_out = self.evaluation_strategy.evaluate(candidate['code'])
        # Post process this eval_out to return a list of scores and feedback
        scores = [eval_out['score'] for eval_out in eval_out]
        return scores

    def make_reflective_dataset(self, candidate: dict, inputs: list, trajectories: list) -> list:
        if not self.config['evaluator']['enable_artifacts']:
            return super().make_reflective_dataset(candidate, inputs, trajectories)
        else: 
            # TODO: replicate openevolve behavior
            return super().make_reflective_dataset(candidate, inputs, trajectories)

    def propose_new_texts(self, candidate: dict, inputs: list, trajectories: list) -> dict:
        return super().propose_new_texts(candidate, inputs, trajectories)
        # Use the llm config from config.yaml to propose new texts
        # Use the prompt.system_prompt from config.yaml to propose new texts

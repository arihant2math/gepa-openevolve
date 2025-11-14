from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
from types import ModuleType
from typing import Any, Callable, Optional

from gepa import EvaluationBatch, GEPAAdapter

# Fallback constant.  Some OpenEvolve evaluators expose FAILED_SCORE, but we
# define one here to stay self-contained.
_GENERIC_FAILED_SCORE = -1.0


class GenericOpenEvolveAdapter(GEPAAdapter[Any, Any, Any]):
    """
    Generic adapter that lets GEPA optimise code using *any* OpenEvolve
    evaluator file.

    Usage
    -----
    >>> adapter = GenericOpenEvolveAdapter(
    ...     evaluation_file="openevolve/examples/ADRS/cant-be-late/evaluator.py"
    ... )
    >>> optimize(..., adapter=adapter)  # GEPA optimise call

    Notes
    -----
    • The ``batch`` argument received by :pymeth:`evaluate` is *ignored*.
      The chosen evaluator is expected to discover its own traces / inputs
      internally (exactly how OpenEvolve examples already work).  You may
      therefore pass ``[None]`` or any dummy list to GEPA.

    • Reflection-LM support is optional.  If ``model`` is:

        – ``callable`` ‑ used verbatim as an LLM;
        – ``str``      ‑ interpreted as a LiteLLM model name;
        – ``None``     ‑ fall back to the primary model in
                         ``openevolve/config.yaml`` (if available).

      Reflection is *not* required for evaluation, but GEPA’s mutation
      pipeline may attempt to call :pymeth:`propose_new_texts`, so a minimal
      echo-LM is provided when none is configured.
    """

    def __init__(
        self,
        *,
        evaluation_file: str,
        model: str | Callable[[str], str] | None = None,
        config_path: str | os.PathLike | None = None,
        failure_score: float = _GENERIC_FAILED_SCORE,
    ) -> None:
        self.evaluation_file = os.fspath(evaluation_file)
        self.failure_score = failure_score

        # ------------------------------------------------------------------ #
        # Reflection LLM initialisation (optional)
        # ------------------------------------------------------------------ #
        if callable(model):
            self.reflection_lm = model
        else:
            import litellm  # type: ignore

            self._litellm = litellm

            if isinstance(model, str):
                # Explicit model name provided by caller.
                model_name = model
                api_key = None
                base_url = None
                temperature = top_p = max_tokens = None
            else:
                # No explicit model → inspect OpenEvolve config
                from openevolve.config import load_config

                if config_path is None:
                    # Default to evaluator directory /config.yaml if present,
                    # else fall back to package-root config (safe default).
                    candidate_cfg = os.path.join(os.path.dirname(self.evaluation_file), "config.yaml")
                    cfg_path = candidate_cfg if os.path.exists(candidate_cfg) else None
                else:
                    cfg_path = os.fspath(config_path)

                oe_cfg = load_config(cfg_path) if cfg_path else load_config()

                if oe_cfg.llm.models:
                    primary = sorted(oe_cfg.llm.models, key=lambda m: m.weight, reverse=True)[0]
                    model_name = primary.name
                    api_key = primary.api_key or oe_cfg.llm.api_key
                    base_url = primary.api_base or oe_cfg.llm.api_base
                    temperature = primary.temperature
                    top_p = primary.top_p
                    max_tokens = primary.max_tokens
                else:
                    # Very old configs: just use GPT-3.5 with OpenAI defaults.
                    model_name = "gpt-3.5-turbo"
                    api_key = None
                    base_url = None
                    temperature = top_p = max_tokens = None

            def _call_lm(prompt: str) -> str:
                """Wrap LiteLLM completion with sane defaults."""
                completion = self._litellm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                # LiteLLM always returns a content string; be defensive anyway.
                return getattr(completion.choices[0].message, "content", "") or ""

            self.reflection_lm = _call_lm

        # ------------------------------------------------------------------ #
        # Pre-load evaluator module to speed up repeated calls
        # ------------------------------------------------------------------ #
        self._evaluator: ModuleType = self._load_evaluator_module(self.evaluation_file)

        # Extract FAILED_SCORE from module when available (task-specific).
        self.failure_score = getattr(self._evaluator, "FAILED_SCORE", failure_score)

        # Track latest tmpdir for clean-up.
        self._last_tmpdir: Optional[str] = None

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #

    @staticmethod
    def _load_evaluator_module(path: str) -> ModuleType:
        """Dynamically import an evaluation file as a Python module."""
        spec = importlib.util.spec_from_file_location("oe_eval_module", path)
        if spec is None or spec.loader is None:  # pragma: no cover
            raise ImportError(f"Could not load evaluator at {path}")
        module = importlib.util.module_from_spec(spec)
        # Ensure the evaluator's directory is import-able so that it can
        # resolve its own relative modules (e.g., `sim_worker`).
        eval_dir = os.path.dirname(os.path.abspath(path))
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        return module

    # --------------------------------------------------------------------- #
    # GEPAAdapter abstract-method implementations
    # --------------------------------------------------------------------- #

    def build_program(self, candidate: dict[str, str]) -> tuple[Optional[dict[str, str]], Optional[Any]]:
        """
        Write candidate code to a temporary file and run *stage-1* validation
        when the evaluator exposes an ``evaluate_stage1`` function.
        """
        code = candidate["program"]

        tmpdir = tempfile.mkdtemp(prefix="oe_generic_")
        program_path = os.path.join(tmpdir, "candidate.py")

        with open(program_path, "w", encoding="utf-8") as fh:
            fh.write(code)

        # If the evaluator has a quick stage-1, run it.
        if hasattr(self._evaluator, "evaluate_stage1"):
            try:
                stage1_out = self._evaluator.evaluate_stage1(program_path)
            except Exception as exc:  # pragma: no cover
                stage1_out = {
                    "runs_successfully": 0.0,
                    "error": str(exc),
                }

            if stage1_out.get("runs_successfully", 0.0) < 1.0:
                # Stage-1 failed – mark as failure and clean up.
                stage1_out.setdefault("score", self.failure_score)
                stage1_out.setdefault("combined_score", self.failure_score)
                shutil.rmtree(tmpdir, ignore_errors=True)
                return None, stage1_out

        # Build succeeded.
        self._last_tmpdir = tmpdir
        return {"program_path": program_path, "tmpdir": tmpdir}, None

    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        batch,
        candidate: dict[str, str],
        capture_traces: bool = False,  # unused, for signature parity
    ) -> EvaluationBatch:
        """
        Run the evaluator’s main function (``evaluate_stage2`` if present, else
        ``evaluate``) and shape the result into a GEPA :class:`EvaluationBatch`.
        """
        build_res, feedback = self.build_program(candidate)
        batch_size = len(batch) if batch else 1  # allow dummy batch == []

        if build_res is None:
            # Build failed → propagate identical scores/outputs
            return EvaluationBatch(
                outputs=[feedback] * batch_size,
                scores=[self.failure_score] * batch_size,
                trajectories=feedback,
            )

        program_path = build_res["program_path"]
        tmpdir = build_res["tmpdir"]

        try:
            if hasattr(self._evaluator, "evaluate_stage2"):
                result = self._evaluator.evaluate_stage2(program_path)
            else:
                # Fallback to single-stage evaluator
                if not hasattr(self._evaluator, "evaluate"):
                    raise AttributeError(
                        f"Evaluator '{self.evaluation_file}' defines neither 'evaluate_stage2' nor 'evaluate'"
                    )
                result = self._evaluator.evaluate(program_path)
        finally:
            # Clean up temp directory.
            shutil.rmtree(tmpdir, ignore_errors=True)
            if self._last_tmpdir == tmpdir:
                self._last_tmpdir = None

        # Extract score
        if isinstance(result, dict):
            score_val = result.get("combined_score") or result.get("score") or self.failure_score
        else:
            # OpenEvolve.EvaluationResult
            score_val = getattr(result, "metrics", {}).get("combined_score") or getattr(result, "metrics", {}).get(
                "score", self.failure_score
            )

        scores = [score_val] * batch_size
        outputs = [result] * batch_size

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=result)

    # ------------------------------------------------------------------ #
    # Optional: provide no-op reflection to satisfy GEPA’s interface.
    # ------------------------------------------------------------------ #

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ):
        # Generic adapter has no task-specific feedback parsing.
        return {}

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        # Fallback: echo original program.
        return {name: candidate[name] for name in components_to_update}

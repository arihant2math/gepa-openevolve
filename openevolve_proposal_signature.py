# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from gepa.proposer.reflective_mutation.base import Signature
from typing import Optional
import re
import logging
import os

# Set up logging
logger = logging.getLogger(__name__)

class OpenEvolveProposalSignature(Signature):
    # Adapt from full_rewrite_prompt_template in openEvolve
    def __init__(self, config):
        self.prompt_template = config["prompt"]["system_message"]

    input_keys = ["current_instruction_doc", "dataset_with_feedback"]
    output_keys = ["new_program"]

    @classmethod
    def prompt_renderer(cls, input_dict: dict[str, str]) -> str:
        # TODO(shu): we don't need many datasets with feedback here 
        # TODO(shu): we just need to provide the original program instructions + execution feedback trace 
        def format_samples(samples):
            formatted = []
            for i, sample in enumerate(samples, 1):
                s = f"Example {i}:\n"
                for key, val in sample.items():
                    s += f"- {key}: {val}\n"
                formatted.append(s.strip())
            return "\n\n".join(formatted)

        prompt = cls.prompt_template
        prompt = prompt.replace("<curr_program>", input_dict["current_instruction_doc"])
        prompt = prompt.replace("<inputs_outputs_feedback>", format_samples(input_dict["dataset_with_feedback"]))
        
        # Log the full prompt if verbose logging is enabled
        if os.environ.get("GEPA_LOG_PROMPTS", "0") == "1":
            logger.info("="*80)
            logger.info("LLM PROMPT:")
            logger.info("="*80)
            logger.info(prompt)
            logger.info("="*80)
        
        return prompt


    @classmethod
    def _parse_full_rewrite(cls, llm_response: str, language: str = "python") -> Optional[str]:
        """
        Extract a full rewrite from an LLM response

        Args:
            llm_response: Response from the LLM
            language: Programming language

        Returns:
            Extracted code or None if not found
        """
        code_block_pattern = r"```" + language + r"\n(.*?)```"
        matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Fallback to any code block
        code_block_pattern = r"```(.*?)```"
        matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Fallback to plain text
        return llm_response

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        # Log the full LLM response if verbose logging is enabled
        if os.environ.get("GEPA_LOG_PROMPTS", "0") == "1":
            logger.info("="*80)
            logger.info("LLM RESPONSE:")
            logger.info("="*80)
            logger.info(lm_out)
            logger.info("="*80)
        
        # TODO(shu): just add some config.language for other languages
        new_program = cls._parse_full_rewrite(lm_out, language="python")

        return {"new_program": new_program}

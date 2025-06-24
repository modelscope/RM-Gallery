import copy
import random
from typing import List

from loguru import logger
from pydantic import Field
from retry import retry

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.message import format_messages
from rm_gallery.core.reward.principle.generator import (
    BaseGeneratorTemplate,
    PrincipleGenerator,
)
from rm_gallery.core.reward.template import BasePromptTemplate


class JustificationTempalte(BasePromptTemplate):
    winner: str = Field(default=..., description="the id of winning completion")

    @classmethod
    def format(
        cls,
        instruction: str,
        completions: List[str],
        preference: int,
        **kwargs,
    ) -> str:
        completion_str = ""
        for i, completion in enumerate(completions):
            completion_str += (
                f"<completion_{i + 1}>\n{completion}\n</completion_{i + 1}>\n\n"
            )

        return f"""You are tasked with analyzing the completions respond to the instruction.
Based on the content, please provide a detailed explanation of why the groud truth might have preferred the winning completion.
Please consider aspects such as clarity, coherence, helpfulness, tone, and overall quality.

## Instruction
{instruction}

## Completions
{completion_str}

## Winning Completion
Completion {preference} is better than others

## Output Format Requirements
{cls.schema(**kwargs)}
"""


class ExtractionTemplate(BaseGeneratorTemplate):
    @classmethod
    def format(
        cls,
        reason: str,
        **kwargs,
    ) -> str:
        return f"""Based on the following reasoning about why completion with assistant winner is better, extract any principle-like statements implied by the reasoning that indicate this preference.
Principle-like statements should be able to be judged objectively and deterministically.
Below are a few examples of principle-like statements:
Validate Assumptions Adequately: The assistant’s responses should validate any assumptions made with sufficient context and examples.
Avoid Repetition: The assistant’s responses should not simply restate information provided by the user as its answer.
Satifaction To User: The assistant’s responses should have a structure that satisfies the user’s request.
## Reasoning
{reason}

## Output Format Requirements
{cls.schema(**kwargs)}
"""


class MergeTemplate(BaseGeneratorTemplate):
    @classmethod
    def format(cls, principles, **kwargs) -> str:
        return f"""Below is a large list of principle-like statements regarding the behavior of an AI assistant.
Some of these principles might be duplicates or very similar in meaning.
Please merge them so that there are no duplicates or principles with very similar meanings.

## Output Format Requirements
{cls.schema(**kwargs)}
"""


class AutoRuleGenerator(PrincipleGenerator):
    def justify(self, sample: DataSample) -> DataSample:
        instruction: str = format_messages(sample.input)
        # Process completions and identify best one
        completions = [
            (output.answer.label["preference"], output.answer.content)
            for output in sample.output
        ]
        random.shuffle(completions)
        for i, (label, completion) in enumerate(completions):
            if label == "chosen":
                best = i + 1
        completions = [completion for _, completion in completions]

        prompt = JustificationTempalte.format(
            instruction=instruction,
            completions=completions,
            preference=best,
            enable_thinking=self.llm.enable_thinking,
        )

        @retry(tries=self.max_retries, delay=1.0)
        def call():
            logger.info(f"prompt: {prompt}")
            response = self.llm.simple_chat(
                prompt,
                sys_prompt="You are a professional assistant skilled in step-by-step justifying and reasoning.",
            )
            result = ExtractionTemplate.parse(response)
            sample.input[-1].additional_kwargs["justification"] = result.model_dump()
            return sample

        try:
            sample = call()
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
        return sample

    def extract(self, sample: DataSample) -> DataSample:
        try:
            reason = sample.input[-1].additional_kwargs["justification"]["reason"]
        except:
            return sample

        prompt = ExtractionTemplate.format(
            reason=reason, enable_thinking=self.llm.enable_thinking
        )

        @retry(tries=self.max_retries, delay=1.0)
        def call():
            logger.info(f"prompt: {prompt}")
            response = self.llm.simple_chat(
                prompt,
                sys_prompt="You are a professional assistant skilled in extracting key insights and summarizing information.",
            )
            result = ExtractionTemplate.parse(response)
            sample.input[-1].additional_kwargs["extraction"] = result.model_dump()
            return sample

        try:
            sample = call()
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
        return sample

    def generate(self, sample: DataSample):
        sample = copy.deepcopy(sample)
        sample = self.justify(sample)
        sample = self.extract(sample)
        return sample

    def cluster(self, samples: List[DataSample]):
        # Build example strings from sample principles
        principles = []
        for i, sample in enumerate(samples):
            try:
                if "extraction" not in sample.input[-1].additional_kwargs:
                    continue

                for key, value in (
                    sample.input[-1]
                    .additional_kwargs["extraction"]["principles"]
                    .items()
                ):
                    principles.append(f"{key}: {value}")
            except:
                continue

        logger.info(f"===RAW PRINCIPLES===\n{principles}")

        # Get clustered principles from LLM
        @retry(tries=self.max_retries, delay=1.0)
        def call():
            response = self.llm.simple_chat(
                MergeTemplate.format(principles=principles),
                sys_prompt="You are a skilled professional assistant focusing on induction and summarization.",
            )
            result = MergeTemplate.parse(response)
            logger.info("===CLUSTER RESULT===\n" + result.model_dump_json())
            return result.principles

        try:
            principles = call()
        except Exception as e:
            principles = {}
            logger.error(f"API call failed: {str(e)}")
        return principles

import copy
import json
import random
import re
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from typing import Dict, List

from loguru import logger
from pydantic import BaseModel, Field

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.base import BaseLLM
from rm_gallery.core.model.message import format_messages
from rm_gallery.core.reward.template import BasePromptTemplate


class BaseGeneratorTemplate(BasePromptTemplate):
    principles: Dict[str, str] = Field(
        default=...,
        description="""```json
{
    "{phrase}": "{description}",
    ...
}
```""",
    )

    @classmethod
    def parse(cls, text: str):
        contents = cls._parse(text)

        json_pattern = r"```json(.*?)```"
        json_dict = re.findall(json_pattern, contents["principles"], re.DOTALL)
        json_dict = json_dict[0] if len(json_dict) > 0 else "{}"

        try:
            parsed_dict = json.loads(json_dict)
        except json.JSONDecodeError:
            pattern = r'"(.*?)"\s*:\s*"(.*?)"'
            matches = re.findall(pattern, json_dict)
            parsed_dict = {key: value for key, value in matches}

        return cls(
            think=contents["think"],
            principles=parsed_dict,
        )


class PrincipleGenerateTempalte(BaseGeneratorTemplate):
    @classmethod
    def format(
        cls,
        desc: str,
        instruction: str,
        completions: List[str],
        preference: str | int,
        **kwargs,
    ) -> str:
        completion_str = ""
        for i, completion in enumerate(completions):
            completion_str += f"### Completion {i + 1}\n{completion}\n\n\n"

        return f"""## Overview
Please propose some concise principles about why one completion is superior to the others in the scenario.
Another assistant will evaluate the output based on these principles.

## Requirements for Principles:
(1) The principles should target some general standards of the "scenario" that may revolve around key points of the instruction.
(2) Principles are presented from most important to least important.
(3) The principles should be as critical as possible.
(4) Each principle should consist of a brief phrase accompanied by a single sentence description.
(5) The number of principles should be LESS THAN OR EQUAL TO 10.

## Input
### Scenario
{desc}

### Instruction:
{instruction}

{completion_str}
### Preference:
Completion {preference}

## Output Format Requirements
{cls.schema(**kwargs)}
"""


class PrincipleClusterTemplate(BaseGeneratorTemplate):
    @classmethod
    def format(cls, principles: str, desc: str, **kwargs) -> str:
        return f"""## Overview
Please summarize some concise principles from the candicates that tackle the scenario.
Another assistant will evaluate the completion based on these principles.

## Requirements for Principles:
(1) The principles should **specifically** target some general standards of the "scenario".
(2) Principles are presented from most important to least important.
(3) The principles should be as critical as possible.
(4) Each principle should consist of a brief phrase accompanied by a single sentence description.
(5) The number of principles should be LESS THAN OR EQUAL TO 10.
(6) Focus on summarizing recurring candidate principles.

## Input
### Scenario
{desc}

{principles}

## Output Format Requirements
{cls.schema(**kwargs)}
"""


class PrincipleGenerator(BaseModel):
    llm: BaseLLM = Field(default=..., description="llm client")
    desc: str = Field(default=..., description="description")

    def generate(self, sample: DataSample):
        sample = copy.deepcopy(sample)
        instructioin: str = format_messages(sample.input)
        completions = [
            (output.answer.label["preference"], output.answer.content)
            for output in sample.output
        ]
        random.shuffle(completions)
        for i, (label, completion) in enumerate(completions):
            if label == "chosen":
                best = i + 1
        completions = [completion for _, completion in completions]

        prompt = PrincipleGenerateTempalte.format(
            instruction=instructioin,
            completions=completions,
            preference=best,
            enable_thinking=self.llm.enable_thinking,
            desc=self.desc,
        )

        logger.info(f"prompt: {prompt}")
        response = self.llm.simple_chat(
            prompt,
            sys_prompt="You are a professional assistant skilled in extracting key insights and summarizing information.",
        )
        result = PrincipleGenerateTempalte.parse(response)
        sample.input[-1].additional_kwargs["generate"] = result.model_dump()
        return sample

    def cluster(self, samples: List[DataSample]):
        str_principles = []
        for sample in samples:
            for key, value in (
                sample.input[-1].additional_kwargs["generate"]["principles"].items()
            ):
                str_principles.append(f"{key}: {value}")

        principles = "\n".join(str_principles)
        logger.info("===RAW PRINCIPLES===\n" + principles)

        response = self.llm.simple_chat(
            PrincipleClusterTemplate.format(
                desc=self.desc,
                principles=principles,
                enable_thinking=self.llm.enable_thinking,
            ),
            sys_prompt="You are a skilled professional assistant focusing on induction and summarization.",
        )
        result = PrincipleClusterTemplate.parse(response)

        logger.info("===CLUSTER RESULT===\n" + result.model_dump_json())
        return result.principles

    def run_batch(
        self, samples: List[DataSample], thread_pool: ThreadPoolExecutor
    ) -> Dict[str, str]:
        futures = [thread_pool.submit(self.generate, sample) for sample in samples]
        wait(futures, return_when=ALL_COMPLETED)
        samples = [future.result() for future in futures]
        return self.cluster(samples)

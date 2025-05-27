


from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, List, Type, TypeVar

from pydantic import Field
from src.base_module import BaseModule
from src.data.base import BaseDataSet
from src.data.schema import DataSample, ChatMessage, Step
from src.model.base import LLMClient
from src.rm.template import BaseTemplate


class BaseRewardModule(BaseModule):
    name: str = Field(default=...)

    @abstractmethod
    def _run(self, **kwargs):
        ...

    @abstractmethod
    def run(self, sample: DataSample, thread_pool: ThreadPoolExecutor):
        ...

    def run_batch(self, samples: BaseDataSet, thread_pool: ThreadPoolExecutor):
        for sample in samples:
            thread_pool.submit(self.run, sample=sample, thread_pool=thread_pool)


class StepRewardModule(BaseRewardModule):
    @abstractmethod
    def _run(self, input: List[ChatMessage], output: Step, step: Step) -> dict:
        ...

    def run(self, sample: DataSample, thread_pool: ThreadPoolExecutor):
        for output in sample.output:
            assert isinstance(output.steps, list)
            for step in output.steps:
                thread_pool.submit(self._run, input=sample.input, output=output, step=step)


class PointRewardModule(BaseRewardModule):
    @abstractmethod
    def _run(self, input: List[ChatMessage], output: Step) -> dict:
        ...

    def run(self, sample: DataSample, thread_pool: ThreadPoolExecutor):
        for output in sample.output:
            thread_pool.submit(self._run, input=sample.input, output=output)


class ListRewardModule(BaseRewardModule):
    @abstractmethod
    def _run(self, sample: DataSample):
        ...
    
    def run(self, sample: DataSample, thread_pool: ThreadPoolExecutor):
        self._run(sample=sample)


T = TypeVar("T", StepRewardModule, PointRewardModule, ListRewardModule)


class LLMRewardModule(Generic[T]):
    llm: LLMClient = Field(default=..., description="llm client")
    desc: str | None = Field(default=None, description="evaluation task description")
    template: Type[BaseTemplate] = Field(default=BaseTemplate, description="prompt template")

    def format(self, **kwargs) -> str:
        """
        format prompt
        """
        return self.template.format(desc=self.desc, **kwargs)

    def _run(self, **kwargs):
        query = self.format(**kwargs)
        response = self.llm.simple_chat(query=query)
        output = self.template.parse(response)
        return output

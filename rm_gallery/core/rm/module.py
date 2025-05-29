


from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Type

from pydantic import Field

from rm_gallery.core.base_module import BaseModule
from rm_gallery.core.data.base import BaseDataSet
from rm_gallery.core.data.schema import DataOutput, DataSample
from rm_gallery.core.model.base import BaseLLM
from rm_gallery.core.rm.schema import DimensionRank, DimensionScore, ModuleResult
from rm_gallery.core.rm.template import BaseTemplate
from rm_gallery.core.utils.registry import ModuleRegistry


class BaseRewardModule(BaseModule):
    """
    This class is used to define a basic reward module, inheriting from BaseModule.
    """

    name: str = Field(default=..., description="The name of the reward module")

    @abstractmethod
    def _run(self, sample: DataSample, **kwargs) -> ModuleResult:
        """
        Main processing function, intended to be implemented by subclasses.
        This method accepts arbitrary keyword arguments and its specific behavior is determined by the subclass implementation.
        """
        ...

    @abstractmethod
    def run(
        self,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> DataSample:
        """
        This abstract method is designed to handle the execution of a single data sample using the provided thread pool.
        Subclasses must implement this method to define the specific behavior of the task execution.

        Parameters:
        - sample: DataSample type, representing the data sample to be processed.
        - thread_pool: ThreadPoolExecutor type, representing the thread pool used for task execution.
        """
        ...

    def run_batch(
        self,
        dataset: BaseDataSet,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> BaseDataSet:
        """
        Execute a batch of samples.
        """
        samples = []
        for sample in dataset.get_data_samples():
            samples.append(self.run(sample=sample, thread_pool=thread_pool))

        return BaseDataSet(datas=samples)


class StepModule(BaseRewardModule):
    """
    The StepModule class, derived from BaseModule, is designed to calculate rewards for each step.
    """

    @abstractmethod
    def _run(self, sample: DataSample, **kwargs) -> ModuleResult[DimensionScore]:
        """
        Abstract method _run, intended to be implemented by subclasses for processing each step's reward calculation.
        """
        ...

    def run(
        self,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> DataSample:
        """
        Method run processes the reward calculation for each step of the sample's output.
        """
        # TODO parallel
        for output in sample.output:
            assert isinstance(output.steps, list)
            for step in output.steps:
                result = self._run(
                    sample=DataSample(
                        unique_id=sample.unique_id,
                        input=sample.input,
                        output=[DataOutput(answer=output.answer, steps=[step])],
                    )
                )
                step.reward.details.extend(result.reward_details)
                step.additional_kwargs[self.name] = result.extra_data
        return sample


class PointModule(BaseRewardModule):
    @abstractmethod
    def _run(self, sample: DataSample, **kwargs) -> ModuleResult[DimensionScore]:
        """
        This method is responsible for processing a list of chat messages and updating the step output based on the processing results.
        It is an internal method, indicated by the leading underscore, suggesting it should not be called directly from outside the class.
        """
        ...

    def run(
        self, sample: DataSample, thread_pool: ThreadPoolExecutor | None = None
    ) -> DataSample:
        """
        Starts processing a data sample by submitting tasks to a thread pool.

        For each output in the data sample, a task is submitted to the thread pool for processing.
        This method does not wait for the tasks to complete.

        Parameters:
        - sample: DataSample - The data sample to be processed, containing input and a list of outputs.
        - thread_pool: ThreadPoolExecutor - The thread pool executor used to execute the tasks.
        """
        # TODO parallel
        for output in sample.output:
            result = self._run(
                sample=DataSample(
                    unique_id=sample.unique_id, input=sample.input, output=[output]
                )
            )
            output.answer.reward.details += result.reward_details
            output.answer.additional_kwargs[self.name] = result.extra_data
        return sample


class ListModule(BaseRewardModule):
    """
    This class is a subclass of BaseModule, designed to process data samples and compute rewards.
    It is an abstract class that requires subclasses to implement the _run method to specify the specific reward calculation logic.
    """

    @abstractmethod
    def _run(self, sample: DataSample, **kwargs) -> ModuleResult[DimensionRank]:
        """
        Abstract method _run, intended to be overridden by subclasses.
        This method's purpose is to process a given data sample to compute a reward.

        """
        ...

    def run(
        self,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> DataSample:
        """
        Executes the _run method in a multi-threaded environment.
        This method is intended to be called by external code, which provides a data sample and a thread pool executor.

        """
        result = self._run(sample=sample, thread_pool=thread_pool, **kwargs)
        for reward in result.reward_details:
            for i, output in enumerate(sample.output):
                output.answer.reward.details.append(reward[i])

        sample.input[-1].additional_kwargs[self.name] = result.extra_data
        return sample


class LLMModule(BaseRewardModule):
    """
    A generic class for LLM-based reward modules, providing a framework for interaction with LLMs and handli
    """

    llm: BaseLLM = Field(default=..., description="llm client")
    template: Type[BaseTemplate] = Field(
        default=BaseTemplate, description="prompt template"
    )

    def _before_call(self, **kwargs) -> dict:
        """
        Abstract method to be implemented by subclasses for preparing parameters before making a call to the LLM.
        """
        return {}

    def _call(self, **kwargs) -> BaseTemplate:
        """
        Abstract method to be implemented by subclasses for making a call to the LLM using the provided parameters.
        """
        return self.template.call(llm=self.llm, **kwargs)

    def _after_call(self, response: BaseTemplate, **kwargs) -> ModuleResult:
        """
        Abstract method to be implemented by subclasses for processing the response from the LLM and setting the reward.
        """
        return ModuleResult(
            module_name=self.name, reward_details=[], extra_data=response.model_dump()
        )

    def _run(self, **kwargs) -> ModuleResult:
        """
        Method to execute the full cycle of preparing the call, making the call to the LLM, and processing the response.
        """
        # Prepare parameters before making the call to the LLM.
        params = self._before_call(**kwargs)

        # Make the call to the LLM using the prepared parameters.
        response = self._call(**params)

        # Process the response from the LLM and set the reward.
        result = self._after_call(response=response, **kwargs)
        return result


class SequenceModule(LLMModule, BaseRewardModule):
    dimensions: List[Dict[str, Any] | BaseRewardModule] = Field(
        default_factory=list, description="dimension"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(len(self.dimensions)):
            if isinstance(self.dimensions[i], dict):
                if isinstance(self.dimensions[i]["cls"], str):
                    self.dimensions[i] = ModuleRegistry.get(self.dimensions[i]["cls"])(
                        llm=self.llm, **self.dimensions[i]["params"]
                    )
                elif issubclass(self.dimensions[i]["cls"], BaseRewardModule):
                    self.dimensions[i] = self.dimensions[i]["cls"](
                        llm=self.llm, **self.dimensions[i]["params"]
                    )
                else:
                    raise ValueError(f"Invalid dimension: {self.dimensions[i]}")

    def run(
        self, sample: DataSample, thread_pool: ThreadPoolExecutor | None = None
    ) -> DataSample:
        for dimension in self.dimensions:
            dimension.run(sample, thread_pool)

        return sample

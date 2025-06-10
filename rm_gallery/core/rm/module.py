


from abc import abstractmethod
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from typing import Dict, List, Type

import numpy as np
from loguru import logger
from pydantic import Field

from rm_gallery.core.base_module import BaseModule
from rm_gallery.core.data.schema import DataOutput, DataSample
from rm_gallery.core.model.base_llm import BaseLLM
from rm_gallery.core.rm.schema import (
    RewardDimensionWithRank,
    RewardDimensionWithScore,
    RewardResult,
)
from rm_gallery.core.rm.template import (
    BasePromptTemplate,
    PrincipleListWiseTemplate,
    PrinciplePointWiseTemplate,
)


class BaseReward(BaseModule):
    """
    Base class for reward modules that provides fundamental evaluation interfaces.

    Attributes:
        name (str): Identifier for the reward module
    """

    name: str = Field(default=..., description="The name of the reward module")

    @abstractmethod
    def _evaluate(self, sample: DataSample, **kwargs) -> RewardResult:
        """
        Core evaluation logic to be implemented by subclasses.

        Processes a single data sample and generates reward metrics.

        Parameters:
            sample (DataSample): Input data sample containing prompts and responses
            **kwargs: Additional implementation-specific parameters

        Returns:
            RewardResult: Computed reward metrics and metadata
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> DataSample:
        """
        Executes evaluation on a single data sample.

        Provides thread-safe execution capability through optional thread pool.

        Parameters:
            sample (DataSample): Data sample to evaluate
            thread_pool (ThreadPoolExecutor | None): Optional executor for parallel processing
            **kwargs: Additional parameters for evaluation logic

        Returns:
            DataSample: Processed sample with reward metrics populated
        """
        ...

    def evaluate_batch(
        self,
        samples: List[DataSample],
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> List[DataSample]:
        """
        Processes multiple data samples in parallel or sequentially.

        Uses provided thread pool for concurrent execution when available.

        Parameters:
            samples (List[DataSample]): Batch of samples to process
            thread_pool (ThreadPoolExecutor | None): Optional executor for parallel processing
            **kwargs: Parameters passed to individual evaluations

        Returns:
            List[DataSample]: Processed samples with reward metrics
        """
        if thread_pool:
            futures = [
                thread_pool.submit(self.evaluate, sample, **kwargs)
                for sample in samples
            ]
            wait(futures, return_when=ALL_COMPLETED)
            samples = [future.result() for future in futures]
        else:
            for i, sample in enumerate(samples):
                samples[i] = self.evaluate(sample=sample, thread_pool=thread_pool)

        return samples

    def best_of_n(
        self,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        n: int = 1,
        **kwargs,
    ) -> DataSample:
        """
        Selects top-n responses based on reward scores.

        Evaluates sample responses and retains those with highest scores.

        Parameters:
            sample (DataSample): Input sample containing multiple responses
            thread_pool (ThreadPoolExecutor | None): Optional executor for parallel processing
            n (int): Number of top responses to retain
            **kwargs: Parameters passed to evaluation

        Returns:
            DataSample: Filtered sample containing top-n responses
        """
        sample = self.evaluate(sample=sample, thread_pool=thread_pool, **kwargs)
        indices = np.argsort(
            np.array([output.answer.reward.score for output in sample.output])
        )[-n:]
        sample.output = [sample.output[i] for i in indices]
        return sample


class BaseStepWiseReward(BaseReward):
    """
    Reward module for step-wise evaluation of multi-step reasoning processes.

    Processes each reasoning step independently to assess quality progression.
    """

    @abstractmethod
    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Step-level evaluation logic to be implemented by subclasses.

        Parameters:
            sample (DataSample): Single-step data sample for evaluation
            **kwargs: Additional parameters for evaluation logic

        Returns:
            RewardResult[RewardDimensionWithScore]: Step-specific reward metrics
        """
        ...

    def evaluate(
        self,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> DataSample:
        """
        Processes all reasoning steps in a data sample.

        Applies step-wise evaluation to each step in the response chain.

        Parameters:
            sample (DataSample): Multi-step reasoning sample to evaluate
            thread_pool (ThreadPoolExecutor | None): Optional executor for parallel processing
            **kwargs: Parameters passed to step-wise evaluation

        Returns:
            DataSample: Sample with step-level reward metrics populated
        """
        sample = sample.model_copy(deep=True)
        futures = []
        for i, output in enumerate(sample.output):
            assert isinstance(output.steps, list)
            for j, step in enumerate(output.steps):
                subsample = DataSample(
                    unique_id=sample.unique_id,
                    input=sample.input,
                    output=[DataOutput(answer=output.answer, steps=[step])],
                )

                if thread_pool:
                    futures.append(
                        (
                            i,
                            j,
                            thread_pool.submit(
                                self._evaluate,
                                sample=subsample,
                                thread_pool=thread_pool,
                            ),
                        )
                    )
                else:
                    result = self._evaluate(
                        sample=subsample,
                        thread_pool=thread_pool,
                    )
                    step.reward.details.extend(result.details)
                    step.additional_kwargs[self.name] = result.extra_data
        if thread_pool:
            wait([future[-1] for future in futures], return_when=ALL_COMPLETED)
            for i, j, future in futures:
                result = future.result()
                step = sample.output[i].steps[j]
                step.reward.details.extend(result.details)
                step.additional_kwargs[self.name] = result.extra_data
        return sample


class BasePointWiseReward(BaseReward):
    """
    Point-wise reward module for individual response evaluation.

    Evaluates each response independently without considering relative ranking.
    """

    @abstractmethod
    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Processes a single response to generate reward metrics.

        Parameters:
            sample (DataSample): Single-response data sample
            **kwargs: Evaluation parameters

        Returns:
            RewardResult[RewardDimensionWithScore]: Response-specific reward metrics
        """
        ...

    def evaluate(
        self, sample: DataSample, thread_pool: ThreadPoolExecutor | None = None
    ) -> DataSample:
        """
        Evaluates all responses in a data sample independently.

        Processes responses either in parallel (with thread pool) or sequentially.

        Parameters:
            sample (DataSample): Sample containing multiple responses
            thread_pool (ThreadPoolExecutor | None): Optional executor for parallel processing

        Returns:
            DataSample: Responses with point-wise reward metrics
        """
        sample = sample.model_copy(deep=True)
        futures = []
        for i, output in enumerate(sample.output):
            subsample = DataSample(
                unique_id=sample.unique_id, input=sample.input, output=[output]
            )

            if thread_pool:
                futures.append(
                    (
                        i,
                        thread_pool.submit(
                            self._evaluate, sample=subsample, thread_pool=thread_pool
                        ),
                    )
                )
            else:
                result = self._evaluate(
                    sample=subsample,
                    thread_pool=thread_pool,
                )
                output.answer.reward.details += result.details
                output.answer.additional_kwargs[self.name] = result.extra_data

        if thread_pool:
            wait([future[-1] for future in futures], return_when=ALL_COMPLETED)
            for i, future in futures:
                result = future.result()
                output = sample.output[i]
                output.answer.reward.details += result.details
                output.answer.additional_kwargs[self.name] = result.extra_data

        return sample


class BaseListWiseReward(BaseReward):
    """
    List-wise reward module for comparative evaluation of multiple responses.

    Evaluates responses as a group to determine relative rankings.
    """

    @abstractmethod
    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithRank]:
        """
        Group evaluation logic to determine response rankings.

        Parameters:
            sample (DataSample): Multi-response sample for comparative evaluation
            **kwargs: Evaluation parameters

        Returns:
            RewardResult[RewardDimensionWithRank]: Relative ranking metrics
        """
        ...

    def evaluate(
        self,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> DataSample:
        """
        Executes list-wise evaluation on a group of responses.

        Applies ranking logic to all responses in the sample.

        Parameters:
            sample (DataSample): Multi-response sample to evaluate
            thread_pool (ThreadPoolExecutor | None): Optional executor for parallel processing
            **kwargs: Parameters for evaluation logic

        Returns:
            DataSample: Responses with ranking information populated
        """
        sample = sample.model_copy(deep=True)
        result = self._evaluate(sample=sample, thread_pool=thread_pool, **kwargs)
        for reward in result.details:
            for i, output in enumerate(sample.output):
                output.answer.reward.details.append(reward[i])

        sample.input[-1].additional_kwargs[self.name] = result.extra_data
        return sample


class BasePairWiseReward(BaseListWiseReward):
    """
    Pair-wise comparison reward module.

    Compares responses in pairs to determine relative preferences.
    """

    def evaluate(
        self,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> DataSample:
        """
        Performs all pairwise comparisons between responses.

        Evaluates every possible pair of responses to build comparative metrics.

        Parameters:
            sample (DataSample): Multi-response sample for pairwise evaluation
            thread_pool (ThreadPoolExecutor | None): Optional executor for parallel processing
            **kwargs: Evaluation parameters

        Returns:
            DataSample: Responses with pairwise comparison metrics
        """
        sample = sample.model_copy(deep=True)
        for i, output_i in enumerate(sample.output):
            for j, output_j in enumerate(sample.output, start=i + 1):
                subsample = DataSample(
                    unique_id=sample.unique_id,
                    input=sample.input,
                    output=[output_i, output_j],
                )
                result = self._evaluate(
                    sample=subsample, thread_pool=thread_pool, **kwargs
                )
                for reward in result.details:
                    output_i.answer.reward.details.append(reward[0])
                    output_j.answer.reward.details.append(reward[1])
        return sample


class BaseLLMReward(BaseReward):
    """
    Base class for LLM-based reward modules.

    Provides framework for prompt-based interaction with language models.
    """

    llm: BaseLLM = Field(default=..., description="llm client")
    template: Type[BasePromptTemplate] = Field(
        default=BasePromptTemplate, description="prompt template"
    )
    to_format: bool = Field(default=False, description="evaluate or format")

    def _before_evaluate(self, **kwargs) -> dict:
        """
        Prepares parameters for prompt generation.

        Returns:
            dict: Parameters for prompt template formatting
        """
        return {}

    def _after_evaluate(self, response: BasePromptTemplate, **kwargs) -> RewardResult:
        """
        Processes LLM response into reward metrics.

        Parameters:
            response (BasePromptTemplate): Parsed LLM response

        Returns:
            RewardResult: Structured reward metrics
        """
        return RewardResult(
            name=self.name, details=[], extra_data=response.model_dump()
        )

    def _format(self, **kwargs):
        """
        Generates prompt without executing LLM call.

        Returns:
            RewardResult: Contains generated prompt in extra_data
        """
        params = self._before_evaluate(**kwargs)
        prompt = self.template.format(**params)
        logger.info(f"prompt: {prompt}")
        return RewardResult(name=self.name, details=[], extra_data={"prompt": prompt})

    def _evaluate(self, **kwargs) -> RewardResult:
        """
        Full LLM evaluation cycle: prepare, execute, process.

        Handles errors during LLM interaction gracefully.

        Returns:
            RewardResult: Evaluation results with metrics and metadata
        """
        if self.to_format:
            return self._format(**kwargs)

        try:
            params = self._before_evaluate(**kwargs)
            prompt = self.template.format(
                enable_thinking=self.llm.enable_thinking, **params
            )
            logger.info(f"prompt: {prompt}")

            response = self.llm.simple_chat(query=prompt)
            response = self.template.parse(response)
            logger.info(f"response: {response}")

            result = self._after_evaluate(response=response, **kwargs)
            result.extra_data["prompt"] = prompt
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            result = RewardResult(
                name=self.name, details=[], extra_data={"error": str(e)}
            )
        return result


class BasePrincipleReward(BaseLLMReward):
    """
    Principle-based reward module using LLM evaluation.

    Evaluates responses against defined ethical/principle guidelines.
    """

    principles: List[str] = Field(default=..., description="principles")
    examples: List[str] = Field(default=[], description="examples")
    template: Type[BasePromptTemplate] = Field(
        default=PrinciplePointWiseTemplate, description="harmfulnessTemplate"
    )
    desc: str = Field(default=..., description="task desc")

    def _before_evaluate(self, sample: DataSample, **kwargs) -> dict:
        """
        Prepares principle evaluation parameters.

        Parameters:
            sample (DataSample): Sample containing query to evaluate

        Returns:
            dict: Parameters for principle-based prompt generation
        """

        principles_str = ""
        for i, principle in enumerate(self.principles):
            principles_str += f"{i + 1}. {principle}\n"

        return {
            "desc": self.desc,
            "principles": principles_str,
            "examples": "\n".join(self.examples),
            "query": sample.input[-1].content,
        }


class BasePointWisePrincipleReward(BasePrincipleReward, BasePointWiseReward):
    """
    Point-wise principle evaluation using LLM.

    Evaluates each response individually against ethical principles.
    """

    def _before_evaluate(self, sample: DataSample, **kwargs) -> Dict:
        """
        Adds response content to evaluation parameters.

        Parameters:
            sample (DataSample): Sample containing response to evaluate

        Returns:
            Dict: Parameters including response content
        """
        params = super()._before_evaluate(sample=sample, **kwargs)
        params["answer"] = sample.output[0].answer.content
        return params

    def _after_evaluate(
        self, response: PrinciplePointWiseTemplate, sample: DataSample, **kwargs
    ) -> RewardResult:
        """
        Converts LLM response to point-wise reward metrics.

        Parameters:
            response (PrinciplePointWiseTemplate): Parsed LLM evaluation

        Returns:
            RewardResult: Violation score with explanation
        """
        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithRank(
                    name=self.name, reason=response.reason, rank=response.violation
                )
            ],
        )


class BaseListWisePrincipleReward(BasePrincipleReward, BaseListWiseReward):
    """
    List-wise principle evaluation using LLM.

    Compares responses against each other based on ethical principles.
    """

    template: Type[PrincipleListWiseTemplate] = PrincipleListWiseTemplate

    def _before_evaluate(self, sample: DataSample, **kwargs) -> Dict:
        """
        Prepares list-wise evaluation parameters.

        Parameters:
            sample (DataSample): Multi-response sample to evaluate

        Returns:
            Dict: Parameters including all responses for comparison
        """
        params = super()._before_evaluate(sample=sample, **kwargs)
        answers = [output.answer.content for output in sample.output]
        params["answers"] = answers
        return params

    def _after_evaluate(
        self, response: PrincipleListWiseTemplate, sample: DataSample, **kwargs
    ) -> RewardResult:
        """
        Converts LLM response to list-wise ranking metrics.

        Parameters:
            response (PrincipleListWiseTemplate): Parsed LLM comparison

        Returns:
            RewardResult: Relative ranking of responses
        """
        scores = [0 for i in range(len(sample.output))]
        scores[response.best - 1] = 1
        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithRank(
                    name=self.name, reason=response.reason, rank=scores
                )
            ],
        )

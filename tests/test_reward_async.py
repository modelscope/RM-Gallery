#!/usr/bin/env python3
"""
Test asynchronous reward evaluation functionality and compare performance with synchronous functionality
"""

import asyncio
import time
from typing import List

import pytest
from pydantic import Field

from rm_gallery.core.data.schema import DataOutput, DataSample, Step
from rm_gallery.core.model.message import ChatMessage
from rm_gallery.core.reward.base import BaseReward
from rm_gallery.core.reward.composition import SimpleComposition
from rm_gallery.core.reward.schema import RewardDimensionWithScore, RewardResult
from rm_gallery.gallery.rm.general import AccuracyReward


class MockReward(BaseReward):
    """Mock reward module for testing asynchronous functionality"""

    delay: float = Field(default=0.1, description="Delay time for simulation")

    def _evaluate(self, sample: DataSample, **kwargs) -> RewardResult:
        """Mock evaluation process with delay"""
        time.sleep(self.delay)  # Simulate computation time

        # Simple evaluation logic
        score = len(sample.output[0].answer.content) / 100.0
        score = min(score, 1.0)  # Limit maximum score to 1.0

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name,
                    score=score,
                    reason=f"Content length based score: {score:.2f}",
                )
            ],
        )

    def _parallel(self, func, sample, thread_pool=None, **kwargs):
        """Synchronous parallel processing"""
        sample = sample.model_copy(deep=True)
        result = func(sample=sample, **kwargs)

        for output in sample.output:
            output.answer.reward.details.extend(result.details)
            output.answer.additional_kwargs[self.name] = result.extra_data

        return sample

    async def _async_parallel(self, func, sample, semaphore=None, **kwargs):
        """Asynchronous parallel processing"""
        sample = sample.model_copy(deep=True)

        # Use asyncio.to_thread to wrap synchronous functions
        if semaphore:
            async with semaphore:
                result = await asyncio.to_thread(func, sample=sample, **kwargs)
        else:
            result = await asyncio.to_thread(func, sample=sample, **kwargs)

        for output in sample.output:
            output.answer.reward.details.extend(result.details)
            output.answer.additional_kwargs[self.name] = result.extra_data

        return sample


@pytest.fixture
def test_samples():
    """Fixture to create test data samples"""

    def create_test_samples(num_samples: int = 10) -> List[DataSample]:
        samples = []

        for i in range(num_samples):
            sample = DataSample(
                unique_id=f"test_sample_{i}",
                input=[
                    ChatMessage(
                        role="user",
                        content=f"This is test question {i}. Please provide a detailed answer.",
                    )
                ],
                output=[
                    DataOutput(
                        answer=Step(
                            role="assistant",
                            content=f"This is a test answer for question {i}. "
                            * (i + 1),
                            label={"reference": f"Reference answer {i}"},
                        )
                    )
                ],
            )
            samples.append(sample)

        return samples

    return create_test_samples


@pytest.fixture
def accuracy_test_sample():
    """Fixture to create accuracy test sample"""
    return DataSample(
        unique_id="accuracy_test",
        input=[ChatMessage(role="user", content="What is 2+2?")],
        output=[
            DataOutput(
                answer=Step(role="assistant", content="4", label={"reference": "4"})
            )
        ],
    )


@pytest.fixture
def mock_reward():
    """Fixture to create mock reward module"""
    return MockReward(name="mock_reward", delay=0.2)


@pytest.fixture
def composition_reward():
    """Fixture to create composition reward module"""
    return SimpleComposition(
        name="test_composition",
        rewards={
            "mock1": MockReward(name="mock1", delay=0.1 * 5),
            "mock2": MockReward(name="mock2", delay=0.15 * 5),
            "mock3": MockReward(name="mock3", delay=0.12 * 5),
        },
        weights={"mock1": 0.5, "mock2": 0.3, "mock3": 0.2},
        is_parallel=True,
    )


@pytest.mark.asyncio
async def test_accuracy_reward(accuracy_test_sample):
    """Test AccuracyReward's synchronous and asynchronous functionality"""
    print("Testing AccuracyReward...")

    reward = AccuracyReward()

    # Synchronous test
    start_time = time.time()
    result_sync = reward.evaluate(accuracy_test_sample)
    sync_time = time.time() - start_time

    # Asynchronous test
    start_time = time.time()
    result_async = await reward.async_evaluate(accuracy_test_sample)
    async_time = time.time() - start_time

    print(
        f"AccuracyReward - Sync time: {sync_time:.4f}s, Async time: {async_time:.4f}s"
    )
    print(f"Sync score: {result_sync.output[0].answer.reward.score}")
    print(f"Async score: {result_async.output[0].answer.reward.score}")

    # Verify results
    assert (
        result_sync.output[0].answer.reward.score
        == result_async.output[0].answer.reward.score
    )
    # Check that reward details have content
    assert len(result_sync.output[0].answer.reward.details) > 0
    assert len(result_async.output[0].answer.reward.details) > 0
    print()


def test_mock_reward_batch(mock_reward, test_samples):
    """Test mock reward module's batch processing"""
    print("Testing MockReward batch processing...")

    # Create test samples
    samples = test_samples(5)

    start_time = time.time()
    results_async = mock_reward.evaluate_batch(samples, max_workers=5)
    async_time = time.time() - start_time

    print(f"MockReward Batch - sync time: {async_time:.4f}s")
    print(f"sync results count: {len(results_async)}")

    # Verify results
    assert len(results_async) == 5
    for result in results_async:
        # Check score in reward details
        assert len(result.output[0].answer.reward.details) > 0
        assert result.output[0].answer.reward.details[0].score > 0
    print()


@pytest.mark.asyncio
async def test_composition_reward(composition_reward, test_samples):
    """Test composition reward module's synchronous and asynchronous functionality"""
    print("Testing SimpleComposition...")

    # Create test sample
    sample = test_samples(1)[0]

    # Synchronous test
    start_time = time.time()
    result_sync = composition_reward.evaluate(sample)
    sync_time = time.time() - start_time

    # Asynchronous test
    start_time = time.time()
    result_async = await composition_reward.async_evaluate(sample)
    async_time = time.time() - start_time

    print(
        f"SimpleComposition - Sync time: {sync_time:.4f}s, Async time: {async_time:.4f}s"
    )
    print(f"Speedup: {sync_time / async_time:.2f}x")
    print(f"Sync final score: {result_sync.output[0].answer.reward.score:.4f}")
    print(f"Async final score: {result_async.output[0].answer.reward.score:.4f}")

    # Verify results
    assert (
        abs(
            result_sync.output[0].answer.reward.score
            - result_async.output[0].answer.reward.score
        )
        < 0.001
    )
    # Check that reward details have content
    assert len(result_sync.output[0].answer.reward.details) > 0
    assert len(result_async.output[0].answer.reward.details) > 0
    assert async_time < sync_time  # Async should be faster
    print()


def test_large_batch_performance(test_samples):
    """Test performance comparison for large batch data"""
    print("Testing large batch performance...")

    # Create large batch test samples
    samples = test_samples(20)

    # Create composition of multiple reward modules
    composition = SimpleComposition(
        name="large_batch_test",
        rewards={
            "reward1": MockReward(name="reward1", delay=0.05),
            "reward2": MockReward(name="reward2", delay=0.08),
            "reward3": MockReward(name="reward3", delay=0.06),
        },
        weights={"reward1": 0.4, "reward2": 0.4, "reward3": 0.2},
        is_parallel=True,
    )

    start_time = time.time()
    results_sync = composition.evaluate_batch(samples)
    sync_time = time.time() - start_time

    print(f"Large Batch ({len(samples)} samples) - Sync time: {sync_time:.4f}s")
    print(f"Sync results count: {len(results_sync)}")

    # Verify results
    assert len(results_sync) == 20
    for result in results_sync:
        # Check that reward details have content
        assert (
            len(result.output[0].answer.reward.details) >= 3
        )  # Should have details from 3 rewards
        # Check score for each detail
        for detail in result.output[0].answer.reward.details:
            assert detail.score > 0
    print()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "max_workers", [1, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]
)
async def test_concurrency_scaling(max_workers, test_samples):
    """Test performance with different concurrency levels"""
    print(f"Testing concurrency scaling with max_workers={max_workers}...")

    samples = test_samples(15)
    reward = MockReward(name="concurrency_test", delay=0.1)

    start_time = time.time()
    results = await reward._async_evaluate_batch(samples, max_workers=max_workers)
    execution_time = time.time() - start_time

    print(f"Max workers: {max_workers}, Time: {execution_time:.4f}s")

    # Verify results
    assert len(results) == 15
    for result in results:
        # Check that reward details have content
        assert len(result.output[0].answer.reward.details) > 0
        assert result.output[0].answer.reward.details[0].score > 0


@pytest.mark.performance
def test_performance_comparison(composition_reward, test_samples):
    """Performance comparison test"""
    sample = test_samples(1)[0]

    # Run multiple times and take average
    sync_times = []
    async_times = []

    for _ in range(3):
        # Synchronous test
        start_time = time.time()
        result_sync = composition_reward.evaluate(sample)
        sync_times.append(time.time() - start_time)

        # Asynchronous test
        async def async_test():
            start_time = time.time()
            result_async = await composition_reward.async_evaluate(sample)
            return time.time() - start_time

        async_times.append(asyncio.run(async_test()))

    avg_sync_time = sum(sync_times) / len(sync_times)
    avg_async_time = sum(async_times) / len(async_times)
    speedup = avg_sync_time / avg_async_time

    print(f"Average sync time: {avg_sync_time:.4f}s")
    print(f"Average async time: {avg_async_time:.4f}s")
    print(f"Average speedup: {speedup:.2f}x")

    # Verify that async version is indeed faster
    assert (
        speedup > 1.0
    ), f"Async should be faster than sync, but got speedup: {speedup}"

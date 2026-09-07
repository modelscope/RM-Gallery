# -*- coding: utf-8 -*-
"""Cancellation must stop work before releasing the caller's resource slot."""
import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import psutil
import pytest

from openjudge.graders.agentic_grader import AgenticGrader
from openjudge.graders.schema import Checkpoint, GraderScore, Rubric
from openjudge.harness.base import BaseHarness, HarnessResult
from openjudge.runner.resource_executor.semaphore_resource_executor import (
    SemaphoreResourceExecutor,
)


async def _wait_until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout=5)


def _rubrics():
    return [Rubric(name="r", checkpoints=[Checkpoint(id="c", description="criterion")])]


class RunningHarness(BaseHarness):
    def __init__(self, output_dir):
        super().__init__(timeout_s=30)
        self.output_dir = output_dir
        self.sandboxes = {}

    @property
    def default_binary(self):
        return sys.executable

    def build_command(self, sandbox_dir, prompt, model):
        tag = "first" if "<query>first</query>" in prompt else "second"
        self.sandboxes[tag] = sandbox_dir
        script = (
            "import os,pathlib,time\n"
            f"pathlib.Path({str(self.output_dir / (tag + '.pid'))!r}).write_text(str(os.getpid()))\n"
            f"while not pathlib.Path({str(self.output_dir / (tag + '.finish'))!r}).exists(): time.sleep(0.01)\n"
            "pathlib.Path('_judge_result.json').write_text('{\"c\": {\"passed\": true}}')\n"
        )
        return [self.binary, "-c", script]


@pytest.mark.unit
async def test_cancelling_one_run_cleans_up_without_cancelling_another(tmp_path):
    harness = RunningHarness(tmp_path)
    grader = AgenticGrader(harness=harness, rubrics=_rubrics())
    tasks = [asyncio.create_task(grader.aevaluate(query=tag, transcript=[])) for tag in ["first", "second"]]
    try:
        await _wait_until(lambda: all((tmp_path / (tag + ".pid")).exists() for tag in ["first", "second"]))
        first = psutil.Process(int((tmp_path / "first.pid").read_text()))
        second = psutil.Process(int((tmp_path / "second.pid").read_text()))
        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(tasks[0], timeout=5)
        assert not first.is_running()
        assert not harness.sandboxes["first"].exists()
        assert second.is_running()
        assert not tasks[1].done()
        (tmp_path / "second.finish").touch()
        result = await asyncio.wait_for(tasks[1], timeout=5)
        assert isinstance(result, GraderScore)
        assert result.score == 1.0
        assert not harness.sandboxes["second"].exists()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class DelayedCleanupHarness(BaseHarness):
    def __init__(self):
        super().__init__()
        self.started = Event()
        self.stopping = Event()
        self.allow_cleanup = Event()
        self.sandbox_dir = None

    @property
    def default_binary(self):
        return "unused"

    def build_command(self, sandbox_dir, prompt, model):
        return []

    def run(self, sandbox_dir, prompt, schema, model=None, cancel_event=None):
        self.sandbox_dir = sandbox_dir
        self.started.set()
        assert cancel_event.wait(5), "The caller must notify the worker of cancellation"
        self.stopping.set()
        assert self.allow_cleanup.wait(5), "The test must eventually release cleanup"
        return HarnessResult(available=False)


@pytest.mark.unit
async def test_repeated_cancellation_keeps_resource_slot_until_cleanup():
    harness = DelayedCleanupHarness()
    grader = AgenticGrader(harness=harness, rubrics=_rubrics())
    executor = SemaphoreResourceExecutor(max_concurrency=1)
    evaluation = asyncio.create_task(grader.aevaluate(executor=executor, transcript=[]))
    next_started = asyncio.Event()

    async def next_job():
        next_started.set()

    next_task = None
    try:
        await _wait_until(harness.started.is_set)
        evaluation.cancel()
        await _wait_until(harness.stopping.is_set)
        next_task = asyncio.create_task(executor.submit(next_job))
        evaluation.cancel()
        await asyncio.sleep(0.05)
        assert not evaluation.done()
        assert not next_started.is_set()
        assert Path(harness.sandbox_dir).exists()
        harness.allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(evaluation, timeout=5)
        assert not harness.sandbox_dir.exists()
        await asyncio.wait_for(next_task, timeout=5)
        assert next_started.is_set()
    finally:
        harness.allow_cleanup.set()
        evaluation.cancel()
        await asyncio.gather(evaluation, return_exceptions=True)
        if next_task is not None:
            next_task.cancel()
            await asyncio.gather(next_task, return_exceptions=True)


@pytest.mark.unit
async def test_cancelling_queued_work_does_not_wait_for_other_jobs(tmp_path):
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    release_blocker = Event()
    blocker_started = Event()

    def block_worker():
        blocker_started.set()
        release_blocker.wait(5)

    blocker = loop.run_in_executor(None, block_worker)
    harness = RunningHarness(tmp_path)
    grader = AgenticGrader(harness=harness, rubrics=_rubrics())
    evaluation = None
    try:
        await _wait_until(blocker_started.is_set)
        evaluation = asyncio.create_task(grader.aevaluate(transcript=[]))
        await asyncio.sleep(0.05)
        started = time.monotonic()
        evaluation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await evaluation
        assert time.monotonic() - started < 1
        assert not harness.sandboxes
    finally:
        release_blocker.set()
        await blocker
        if evaluation is not None:
            evaluation.cancel()
            await asyncio.gather(evaluation, return_exceptions=True)
        executor.shutdown(wait=True)

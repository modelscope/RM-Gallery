# -*- coding: utf-8 -*-
"""
Agentic grader implementation: agent-as-judge backed by an external
coding-agent CLI harness (Claude Code / Codex / Cursor CLI / ...).

Unlike a single LLM call (LLMGrader) or the in-process ReAct tool-calling
loop this module used to implement, AgenticGrader shells out to a real
external coding-agent CLI inside an isolated sandbox: the agent can read a
candidate's workspace and/or execution transcript, actually execute code
to verify checkpoints, and write its verdict into a fixed result file
inside the sandbox. AgenticGrader never depends on parsing a CLI's own
stdout/`--output-format` schema -- only the sandboxed result file (see
`openjudge.harness.base` for the shared protocol).
"""
import asyncio
import math
import time
from threading import Event
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import ValidationError

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import (
    CheckpointResult,
    GraderError,
    GraderMode,
    GraderScore,
    Rubric,
    RubricResult,
)
from openjudge.harness.base import BaseHarness
from openjudge.harness.sandbox import ProcessSandbox

__all__ = ["AgenticGrader"]


def _validate_rubrics(rubrics: List[Rubric]) -> None:
    """Reject empty rubrics or ambiguous identifiers before invoking a judge."""
    if not rubrics:
        raise ValueError("rubrics is required for AgenticGrader and must contain at least one Rubric.")
    rubric_names = set()
    checkpoint_ids = set()
    for rubric in rubrics:
        if not math.isfinite(rubric.weight) or rubric.weight < 0:
            raise ValueError(f"Rubric {rubric.name!r} must have a finite, non-negative weight.")
        if rubric.name in rubric_names:
            raise ValueError(f"Duplicate rubric name: {rubric.name!r}. Rubric names must be unique.")
        rubric_names.add(rubric.name)
        for checkpoint in rubric.checkpoints:
            if not math.isfinite(checkpoint.weight) or checkpoint.weight < 0:
                raise ValueError(f"Checkpoint {checkpoint.id!r} must have a finite, non-negative weight.")
            if checkpoint.id in checkpoint_ids:
                raise ValueError(
                    f"Duplicate checkpoint ID: {checkpoint.id!r}. Checkpoint IDs must be unique across all rubrics."
                )
            checkpoint_ids.add(checkpoint.id)


def _build_output_schema(rubrics: List[Rubric]) -> Dict[str, Any]:
    """Build the flat `{checkpoint_id: {}}` schema description passed to the harness."""
    schema: Dict[str, Any] = {}
    for rubric in rubrics:
        for checkpoint in rubric.checkpoints:
            schema[checkpoint.id] = {}
    return schema


def _build_prompt(query: str, response: str, rubrics: List[Rubric]) -> str:
    """Render rubrics/checkpoints plus query/response into full judge instructions."""
    lines = [
        "You are an evidence-based judge running inside a sandbox directory. "
        "Only read from this directory and its subdirectories; never access any path outside it.",
        "Evidence available in this directory: "
        "./workspace/ holds the candidate's produced artifacts, if any; "
        "./transcript.jsonl holds the candidate's execution transcript "
        "(one JSON message per line, in order), if any.",
        "Full instructions and the required output_schema are also available in "
        "./_judge_spec.json in this directory.",
        "For each checkpoint below: if its content looks like executable code or a test script, actually run "
        "it inside this sandbox to verify (and record exactly what you ran); if it is a natural-language "
        "judging criterion, judge it against the available evidence and the response text below.",
        "When done, write your verdict strictly following output_schema to ./_judge_result.json in this directory.",
        "The result file must be flat JSON whose top-level keys are the checkpoint ids themselves "
        '(not wrapped in "dimensions" or any other envelope). Each checkpoint value must be an object of the '
        'form {"passed": <true/false>, "reason": "...", "execution_log": "<commands + output you actually ran, '
        'omit this key if not applicable>"}.',
        "Every checkpoint's reason must cite concrete evidence (a file path, or a transcript line number and "
        "excerpt); if you cannot find evidence for a checkpoint, mark it failed and say so in reason.",
        "",
        f"<query>{query}</query>",
        f"<response>{response}</response>",
        "",
    ]
    for rubric in rubrics:
        lines.append(f"## Rubric: {rubric.name} (weight={rubric.weight})")
        if rubric.description:
            lines.append(rubric.description)
        for checkpoint in rubric.checkpoints:
            lines.append(f"- [{checkpoint.id}] (weight={checkpoint.weight}) {checkpoint.description}")
            if checkpoint.content:
                lines.append("  content:")
                lines.append("  ```")
                lines.append(f"  {checkpoint.content}")
                lines.append("  ```")
        lines.append("")
    return "\n".join(lines)


def _normalize_sample(parsed: Dict[str, Any], all_checkpoint_ids: List[str]) -> Dict[str, CheckpointResult]:
    """Normalize one harness sample into `{checkpoint_id: CheckpointResult}`, dropping unknown/malformed entries."""
    normalized: Dict[str, CheckpointResult] = {}
    for checkpoint_id in all_checkpoint_ids:
        val = parsed.get(checkpoint_id)
        if not isinstance(val, dict):
            continue
        try:
            normalized[checkpoint_id] = CheckpointResult.model_validate({**val, "checkpoint_id": checkpoint_id})
        except ValidationError:
            continue
    return normalized


def _aggregate_rubric(rubric: Rubric, checkpoint_results: Dict[str, CheckpointResult]) -> RubricResult:
    """Combine a rubric's checkpoint results into a RubricResult.

    Plain weighted mean of all checkpoints' pass/fail outcomes (each checkpoint
    contributes `weight * (1.0 if passed else 0.0)`); a checkpoint missing from
    `checkpoint_results` (the harness never returned a verdict for it) counts as
    failed rather than being excluded, so a rubric with no evidence at all scores 0.0
    instead of vacuously scoring 1.0. A rubric with zero checkpoints (or zero total
    weight) scores 0.0.

    Note: this is a v1 simplification — there is no `must_have` AND-gate here.
    See "Deferred to a follow-up iteration" at the end of this plan.
    """
    scale = max((cp.weight for cp in rubric.checkpoints), default=0.0)
    if scale == 0:
        score = 0.0
    else:
        # Scaling preserves the mean without overflowing for large finite weights.
        total_weight = sum(cp.weight / scale for cp in rubric.checkpoints)
        weighted = sum(
            cp.weight / scale * (1.0 if checkpoint_results.get(cp.id) and checkpoint_results[cp.id].passed else 0.0)
            for cp in rubric.checkpoints
        )
        score = weighted / total_weight

    results = [checkpoint_results[cp.id] for cp in rubric.checkpoints if cp.id in checkpoint_results]
    return RubricResult(name=rubric.name, score=score, checkpoint_results=results)


def _aggregate_overall(rubrics: List[Rubric], rubric_results: List[RubricResult]) -> float:
    """Weighted mean of `rubric_results` by each rubric's `weight`."""
    scale = max((r.weight for r in rubrics), default=0.0)
    if scale == 0:
        return 0.0
    total_weight = sum(r.weight / scale for r in rubrics)
    by_name = {r.name: r for r in rubric_results}
    weighted = sum(r.weight / scale * by_name[r.name].score for r in rubrics if r.name in by_name)
    return weighted / total_weight


class AgenticGrader(BaseGrader):
    """Agent-as-judge grader backed by an external coding-agent CLI harness.

    Unlike LLMGrader (a single LLM call) or the in-process tool-calling loop
    this class used to implement, AgenticGrader shells out to a real external
    coding-agent CLI (Claude Code / Codex / Cursor CLI) inside an isolated
    sandbox, so the judge can read a candidate's workspace/transcript and
    actually execute code to verify checkpoints rather than guessing.

    Attributes:
        harness: The BaseHarness implementation used to invoke the external CLI.
        rubrics: Default list of Rubric dimensions to evaluate against (can be
            overridden per-call via the `rubrics` keyword argument to `aevaluate()`).
        model: Optional model name passed through to the harness.

    Note:
        This is a v1: each evaluation runs exactly one sandboxed harness sample and
        aggregates checkpoints with a plain weighted mean — there is no k-sample
        majority voting, no `agreement` reliability gate, and no `must_have` AND-gate.
        See "Deferred to a follow-up iteration" at the end of this plan for why, and
        for the shape a later `k`/`min_agreement`/`must_have` addition would take.

    Example:
        >>> from openjudge.harness import ClaudeCodeHarness
        >>> from openjudge.graders.schema import Checkpoint, Rubric
        >>> harness = ClaudeCodeHarness(timeout_s=90)
        >>> rubrics = [Rubric(name="correctness", checkpoints=[
        ...     Checkpoint(id="c1", description="Output matches expected format"),
        ... ])]
        >>> grader = AgenticGrader(harness=harness, rubrics=rubrics)
        >>> result = await grader.aevaluate(
        ...     query="...", response="...", workspace_path="/tmp/candidate",
        ... )
    """

    def __init__(
        self,
        harness: BaseHarness,
        rubrics: List[Rubric],
        name: str = "agentic_grader",
        mode: GraderMode = GraderMode.POINTWISE,
        description: str = "Agent-as-judge grader backed by an external coding-agent harness",
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize AgenticGrader.

        Args:
            harness: Pre-constructed BaseHarness (e.g. `ClaudeCodeHarness()`, required.
            rubrics: Default rubrics to evaluate against (required, non-empty). Can be
                overridden per-call via the `rubrics` keyword to `aevaluate()`.
                Rubric names and checkpoint IDs must be unique across the evaluation.
            name: Grader name.
            mode: Only POINTWISE is supported.
            description: Grader description.
            model: Optional model name passed through to `harness.run(..., model=model)`.
            **kwargs: Additional keyword arguments forwarded to `BaseGrader.__init__`.

        Raises:
            ValueError: If the mode is not POINTWISE, `harness` is None, or rubrics are invalid.
        """
        if mode != GraderMode.POINTWISE:
            raise ValueError("AgenticGrader only supports POINTWISE mode.")
        super().__init__(name=name, mode=mode, description=description, **kwargs)
        if harness is None:
            raise ValueError(
                "harness is required for AgenticGrader. Construct one first, e.g. harness = ClaudeCodeHarness()."
            )
        _validate_rubrics(rubrics)
        self.harness = harness
        self.rubrics = rubrics
        self.model = model

    async def _aevaluate(
        self,
        query: str = "",
        response: str = "",
        workspace_path: Optional[str] = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> Union[GraderScore, GraderError]:
        """Evaluate a candidate by running a single sandboxed harness sample.

        Builds a judge prompt + output schema from `rubrics` (or the `rubrics`
        keyword override), runs one `ProcessSandbox` + harness invocation, then
        aggregates the returned checkpoint verdicts into a weighted score.

        Args:
            query: The original task/query being judged.
            response: The candidate's response text being judged.
            workspace_path: Path to the candidate's produced artifacts directory.
                At least one of `workspace_path`/`transcript` must be provided.
            transcript: The candidate's execution transcript (path to a JSONL file,
                or an already-parsed list of message dicts).
            **kwargs: May include `rubrics: List[Rubric]` to override `self.rubrics`
                for this call only.

        Returns:
            GraderScore on success. GraderError if rubrics are invalid, no evidence
            was given, or the harness sample failed/was unavailable. In the latter case,
            `GraderError.metadata` carries harness-level diagnostics (`exit_code`,
            `timed_out`, `duration`, `raw_stderr`, `setup_error` if the sandbox
            itself could not be built, or `harness_error` if the harness raised
            during execution or result parsing) so callers can tell an infrastructure
            failure apart from "the agent judged checkpoints as failing" without
            reaching into private internals.
        """
        rubrics: List[Rubric] = kwargs.pop("rubrics", self.rubrics)
        try:
            _validate_rubrics(rubrics)
        except ValueError as exc:
            return GraderError(name=self.name, error="invalid_rubrics", reason=str(exc))
        if workspace_path is None and transcript is None:
            return GraderError(
                name=self.name,
                error="no_evidence",
                reason="At least one of workspace_path or transcript must be provided.",
            )

        all_checkpoint_ids = [cp.id for rubric in rubrics for cp in rubric.checkpoints]
        schema = _build_output_schema(rubrics)
        prompt = _build_prompt(query, response, rubrics)

        start_time = time.time()
        checkpoint_results, diagnostics = await self._run_sample(
            prompt, schema, workspace_path, transcript, all_checkpoint_ids
        )
        if checkpoint_results is None:
            return GraderError(
                name=self.name,
                error="unavailable",
                reason="The harness sample failed or the harness CLI is unavailable.",
                metadata={"harness_type": type(self.harness).__name__, **diagnostics},
            )

        rubric_results = [_aggregate_rubric(rubric, checkpoint_results) for rubric in rubrics]
        overall_score = _aggregate_overall(rubrics, rubric_results)
        reasons = [f"{r.name}: score={r.score:.2f}" for r in rubric_results]

        return GraderScore(
            name=self.name,
            score=overall_score,
            reason="; ".join(reasons),
            metadata={
                "rubric_results": [r.model_dump() for r in rubric_results],
                "harness_type": type(self.harness).__name__,
                "total_time": time.time() - start_time,
                **diagnostics,
            },
        )

    async def _run_sample(
        self,
        prompt: str,
        schema: Dict[str, Any],
        workspace_path: Optional[str],
        transcript: Optional[Any],
        all_checkpoint_ids: List[str],
    ) -> Tuple[Optional[Dict[str, CheckpointResult]], Dict[str, Any]]:
        """Run a single sandboxed harness invocation.

        `ProcessSandbox`/`BaseHarness.run` are both blocking, so the call runs on
        the default thread pool executor to avoid blocking the event loop.
        Cancellation signals the worker and waits for cleanup before propagating
        `CancelledError`, preserving the caller's concurrency limit.

        Returns:
            A `(checkpoint_results, diagnostics)` pair. `checkpoint_results` is
            `{checkpoint_id: CheckpointResult}` if the harness came back `available`,
            else `None`. `diagnostics` is always returned (even on success) and its
            caller-facing purpose is to let a `GraderError`/`GraderScore.metadata`
            consumer distinguish an infrastructure failure (CLI crashed/timed out/
            misconfigured path) from "the agent judged the checkpoints as failing":

            - If sandbox setup itself raised (e.g. a bad `workspace_path`/`transcript`
              value) before any subprocess ever ran: `{"setup_error": "<ExceptionType>: <msg>"}`.
            - If `harness.run()` raised after sandbox setup succeeded:
              `{"harness_error": "<ExceptionType>: <msg>"}`.
            - Otherwise: `{"exit_code", "timed_out", "duration"}` straight from the
              underlying `HarnessResult`, plus `"raw_stderr"` when non-empty.
        """
        loop = asyncio.get_running_loop()
        cancel_event = Event()
        worker_started = Event()

        def _run_one() -> Tuple[Optional[Dict[str, CheckpointResult]], Dict[str, Any]]:
            worker_started.set()
            try:
                with ProcessSandbox(
                    workspace_path=workspace_path, transcript=transcript, cancel_event=cancel_event
                ) as sandbox_dir:
                    if cancel_event.is_set():
                        return None, {}
                    try:
                        result = self.harness.run(
                            sandbox_dir, prompt, schema, model=self.model, cancel_event=cancel_event
                        )
                    except Exception as exc:
                        return None, {"harness_error": f"{type(exc).__name__}: {exc}"}
            except Exception as exc:
                return None, {"setup_error": f"{type(exc).__name__}: {exc}"}

            diagnostics: Dict[str, Any] = {
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "duration": result.duration,
            }
            if result.raw_stderr:
                diagnostics["raw_stderr"] = result.raw_stderr
            if not result.available:
                return None, diagnostics
            return _normalize_sample(result.result, all_checkpoint_ids), diagnostics

        worker = loop.run_in_executor(None, _run_one)
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancel_event.set()
            if not worker_started.is_set():
                # Queued work owns no resources. If it races with cancellation,
                # ProcessSandbox observes the signal before creating any files.
                worker.cancel()
                raise
            # Keep the caller's resource slot until the process and sandbox are gone.
            # Shield cleanup from repeated cancellation requests as well.
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
            worker.result()
            raise

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """Return metadata about how AgenticGrader's evaluation mechanism works."""
        return {
            "aevaluate": AgenticGrader._aevaluate.__doc__,
            "protocol": (
                "Writes rubrics/checkpoints as a spec file into an isolated sandbox, invokes an external "
                "coding-agent CLI harness non-interactively, and reads back a result file the agent is "
                "instructed to write -- never parses the CLI's own stdout output format."
            ),
        }

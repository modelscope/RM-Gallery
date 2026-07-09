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
import time
from typing import Any, Dict, List, Optional, Union

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
        if isinstance(val, dict) and "passed" in val:
            normalized[checkpoint_id] = CheckpointResult(
                checkpoint_id=checkpoint_id,
                passed=bool(val.get("passed")),
                reason=str(val.get("reason", "")),
                execution_log=val.get("execution_log"),
            )
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
    total_weight = sum(cp.weight for cp in rubric.checkpoints)
    if total_weight <= 0:
        score = 0.0
    else:
        weighted = sum(
            cp.weight * (1.0 if checkpoint_results.get(cp.id) and checkpoint_results[cp.id].passed else 0.0)
            for cp in rubric.checkpoints
        )
        score = weighted / total_weight

    results = [checkpoint_results[cp.id] for cp in rubric.checkpoints if cp.id in checkpoint_results]
    return RubricResult(name=rubric.name, score=score, checkpoint_results=results)


def _aggregate_overall(rubrics: List[Rubric], rubric_results: List[RubricResult]) -> float:
    """Weighted mean of `rubric_results` by each rubric's `weight`."""
    total_weight = sum(r.weight for r in rubrics)
    if total_weight <= 0:
        return 0.0
    by_name = {r.name: r for r in rubric_results}
    weighted = sum(r.weight * by_name[r.name].score for r in rubrics if r.name in by_name)
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
            name: Grader name.
            mode: Only POINTWISE is supported.
            description: Grader description.
            model: Optional model name passed through to `harness.run(..., model=model)`.
            **kwargs: Additional keyword arguments forwarded to `BaseGrader.__init__`.

        Raises:
            ValueError: If `harness` is None or `rubrics` is empty.
        """
        super().__init__(name=name, mode=mode, description=description, **kwargs)
        if harness is None:
            raise ValueError(
                "harness is required for AgenticGrader. Construct one first, e.g. harness = ClaudeCodeHarness()."
            )
        if not rubrics:
            raise ValueError("rubrics is required for AgenticGrader and must contain at least one Rubric.")
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
            GraderScore on success. GraderError if no evidence was given, or the
            harness sample failed/was unavailable.
        """
        rubrics: List[Rubric] = kwargs.pop("rubrics", self.rubrics)
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
        checkpoint_results = await self._run_sample(prompt, schema, workspace_path, transcript, all_checkpoint_ids)
        if checkpoint_results is None:
            return GraderError(
                name=self.name,
                error="unavailable",
                reason="The harness sample failed or the harness CLI is unavailable.",
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
            },
        )

    async def _run_sample(
        self,
        prompt: str,
        schema: Dict[str, Any],
        workspace_path: Optional[str],
        transcript: Optional[Any],
        all_checkpoint_ids: List[str],
    ) -> Optional[Dict[str, CheckpointResult]]:
        """Run a single sandboxed harness invocation.

        `ProcessSandbox`/`BaseHarness.run` are both blocking, so the call runs on
        the default thread pool executor to avoid blocking the event loop.

        Returns:
            `{checkpoint_id: CheckpointResult}` if the harness came back `available`,
            else `None`.
        """
        loop = asyncio.get_running_loop()

        def _run_one() -> Optional[Dict[str, CheckpointResult]]:
            try:
                with ProcessSandbox(workspace_path=workspace_path, transcript=transcript) as sandbox_dir:
                    result = self.harness.run(sandbox_dir, prompt, schema, model=self.model)
            except Exception:
                return None
            if not result.available:
                return None
            return _normalize_sample(result.result, all_checkpoint_ids)

        return await loop.run_in_executor(None, _run_one)

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

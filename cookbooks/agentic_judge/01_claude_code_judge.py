# -*- coding: utf-8 -*-
"""
Example 1: Judge a coding candidate's workspace with Claude Code as the agent.

Dependencies: Claude Code CLI installed (`npm install -g @anthropic-ai/claude-code`).
Environment: ANTHROPIC_API_KEY (inherited by the harness's subprocess from this
    process's environment), or a prior interactive `claude` login.
"""
import asyncio
import tempfile
from pathlib import Path

from openjudge.graders.agentic_grader import AgenticGrader
from openjudge.graders.schema import Checkpoint, Rubric
from openjudge.harness import ClaudeCodeHarness


def _build_candidate_workspace() -> str:
    """Create a throwaway directory containing the candidate's submitted solution."""
    workspace = Path(tempfile.mkdtemp(prefix="agentic_judge_example_"))
    (workspace / "fibonacci.py").write_text(
        '''def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number (0-indexed: fibonacci(0) == 0)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
''',
        encoding="utf-8",
    )
    return str(workspace)


rubrics = [
    Rubric(
        name="correctness",
        description="The submitted fibonacci() implementation must be correct.",
        weight=2.0,
        checkpoints=[
            Checkpoint(
                id="matches_reference_values",
                description="fibonacci(0..10) matches the well-known Fibonacci sequence",
                content=(
                    "from fibonacci import fibonacci\n"
                    "expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]\n"
                    "for i, exp in enumerate(expected):\n"
                    "    assert fibonacci(i) == exp, f'fibonacci({i}) should be {exp}'\n"
                ),
            ),
            Checkpoint(
                id="rejects_negative_input",
                description="fibonacci(-1) raises ValueError instead of returning a wrong value silently",
                content=(
                    "from fibonacci import fibonacci\n"
                    "try:\n"
                    "    fibonacci(-1)\n"
                    "    raise AssertionError('expected ValueError for negative input')\n"
                    "except ValueError:\n"
                    "    pass\n"
                ),
            ),
        ],
    ),
    Rubric(
        name="code_quality",
        description="The implementation should be reasonably idiomatic and documented.",
        weight=1.0,
        checkpoints=[
            Checkpoint(id="has_docstring", description="fibonacci() has a docstring explaining its behavior"),
        ],
    ),
]


async def main() -> None:
    harness = ClaudeCodeHarness(timeout_s=120)
    grader = AgenticGrader(harness=harness, rubrics=rubrics)

    workspace_path = _build_candidate_workspace()
    result = await grader.aevaluate(
        query="Implement a fibonacci(n) function in fibonacci.py.",
        response="See fibonacci.py in the workspace.",
        workspace_path=workspace_path,
    )

    print(f"score: {getattr(result, 'score', None)}")
    print(f"reason: {result.reason}")
    print(f"metadata: {result.metadata}")


if __name__ == "__main__":
    asyncio.run(main())

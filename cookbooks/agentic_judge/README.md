# Agentic Judge Cookbook

Runnable examples for `AgenticGrader`'s harness-based "agent as a judge" implementation:
instead of a single LLM call, each example shells out to a real coding-agent CLI
(Claude Code / Codex / Cursor CLI) inside an isolated sandbox, lets it read the
candidate's workspace and run code to verify checkpoints, and reads back its verdict
from a fixed result file the agent writes inside the sandbox.

## Prerequisites

Exactly one CLI is required per example, installed and authenticated **on the machine
running the example** (the harness only shells out to whatever's on `PATH`):

| Example | CLI required | Install | Auth |
| --- | --- | --- | --- |
| `01_claude_code_judge.py` | `claude` | `npm install -g @anthropic-ai/claude-code` | `ANTHROPIC_API_KEY` env var, or run `claude` once interactively |
| `02_codex_judge.py` | `codex` | `npm install -g @openai/codex` | `OPENAI_API_KEY` env var, or `codex login` |
| `03_cursor_agent_judge.py` | `cursor-agent` | see https://cursor.com/cli | `CURSOR_API_KEY` env var, or run `cursor-agent` once interactively |

## Running

```bash
python cookbooks/agentic_judge/01_claude_code_judge.py
python cookbooks/agentic_judge/02_codex_judge.py
python cookbooks/agentic_judge/03_cursor_agent_judge.py
```

Each script builds a throwaway candidate workspace containing a `fibonacci.py`
solution, defines two `Rubric`s (`correctness` with checkpoints whose `content` is
an executable assertion, and `code_quality` as a documentation checkpoint), runs
the agentic judge against it, and prints the resulting score/reason/metadata.

## Notes

- These examples make real, billed calls to the configured CLI/model and are not run in
  CI (see `pytest.ini`'s `norecursedirs = cookbooks`).
- This is a v1 `AgenticGrader`: each call runs exactly one sandboxed harness sample and
  aggregates checkpoints with a plain weighted mean. There is no `must_have` AND-gate
  and no k-sample majority voting/`agreement` reliability gate yet — those are deferred
  to a follow-up iteration (see the design doc). If a single sample's judgment turns out
  to be noisy in practice, that is the signal to prioritize building that follow-up.

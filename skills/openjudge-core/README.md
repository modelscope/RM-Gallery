# OpenJudge Core — build evaluation pipelines & RL reward signals

Skills that teach the core OpenJudge library API directly: selecting and
running graders over a dataset, and turning grader output into reward signals
for RL training.

Each skill is a self-contained workflow in `<NN-name>/SKILL.md`.

## The workflows

| # | Skill | Use it when |
|---|---|---|
| 01 | `01-graders-and-pipeline` | You want to evaluate LLM outputs: pick/configure graders, run batch evaluation with `GradingRunner`, aggregate scores, auto-generate graders, analyze results. |
| 02 | `02-rl-reward` | You want to build RL reward signals: pointwise multi-dimensional rewards, pairwise tournament rewards for GRPO, preference pairs for DPO/RLAIF. |

## No router — and why

Unlike `academic-eval`/`arena-eval`, this suite has **no `00-xxx-router`**.
The two skills' scopes don't overlap: "build an evaluation pipeline" and
"build an RL reward signal" are different keywords, different user intents,
and different entry points into the OpenJudge API. Each skill's `description`
frontmatter is specific enough for client-side semantic routing (Cursor,
Claude Code, Codex, etc.) to pick the right one without an extra triage step.
If you're building an RL reward on top of an evaluation pipeline you already
built with `01`, just read `02` next — no diagnostic question needed.

## Sub-documents

Both skills ship topic-specific sub-documents in their own directory (read
inline from the `SKILL.md`, not through this README):

| Skill | Sub-doc | Topic |
|---|---|---|
| `01-graders-and-pipeline` | `graders.md` | Grader selection & configuration |
| `01-graders-and-pipeline` | `pipeline.md` | Batch evaluation pipeline |
| `01-graders-and-pipeline` | `generator.md` | Auto-generate graders from labeled data |
| `01-graders-and-pipeline` | `analyzer.md` | Analyze & compare results (win rates, stats) |
| `02-rl-reward` | `pointwise.md` | Pointwise multi-dimensional reward |
| `02-rl-reward` | `pairwise.md` | Pairwise reward (tournament / DPO preference pairs) |

## Dependency

```bash
pip install py-openjudge
```

Both skills teach the `openjudge` core library directly (`openjudge.graders`,
`openjudge.runner`, `openjudge.evaluation_strategy`) — no cookbook dependency.

## Self-contained skills

Each `<NN-name>/SKILL.md` (with its sub-documents) is self-contained per the
Anthropic Agent Skill protocol — it can be installed and used on its own
without this README or the rest of the suite.

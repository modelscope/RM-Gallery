# Arena Eval Functional Testing Guide

This guide explains how to prove the `arena-eval` skills work on realistic
scenarios. It follows the same actor+judge functional-test pattern as
[`skills/eval_pipeline/tests/`](../../eval_pipeline/tests/eval_pipeline_testing_guide.md) —
this suite reuses that suite's runner rather than shipping a duplicate one
(see [Why a shared runner](#why-a-shared-runner) below, same rationale as
`academic-eval`'s guide).

## What To Test

1. Skill package quality test — `cookbooks/skills_evaluation/evaluate_skills.py`.
2. Functional scenario test (this guide) — `skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py`
   pointed at `skills/arena-eval`. Checks whether a skill guides an agent to the
   right recommendation (routing) or the right concrete answer (config/flags/
   interpretation) for a realistic user request.

## API Key

```bash
# .env at repo root — Aliyun DashScope (Bailian)
DASHSCOPE_API_KEY=sk-...

# or, any OpenAI-compatible provider
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://your-provider.example/v1"   # omit for OpenAI itself
```

Actor and judge default to different models (`qwen3.6-plus` / `qwen3-max`) so
one model doesn't grade its own output.

## Quick Smoke Test

```bash
python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py \
  --skill-root skills/arena-eval \
  --cases skills/arena-eval/tests/arena_eval_test_cases.jsonl \
  --out-dir skills/arena-eval/tests/results \
  --report-prefix arena_eval \
  --case-id arena_router_002_citation_hallucination_benchmark
```

Representative smoke set:

```bash
python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py \
  --skill-root skills/arena-eval \
  --cases skills/arena-eval/tests/arena_eval_test_cases.jsonl \
  --out-dir skills/arena-eval/tests/results \
  --report-prefix arena_eval \
  --case-id arena_router_003_better_at_recommending_papers \
  --case-id arena_router_004_single_document_not_arena \
  --case-id auto_arena_002_rerun_judge_only \
  --case-id ref_arena_arena_002_ranking_tiebreak_order \
  --case-id arena_eval_001_end_to_end
```

Full set:

```bash
python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py \
  --skill-root skills/arena-eval \
  --cases skills/arena-eval/tests/arena_eval_test_cases.jsonl \
  --out-dir skills/arena-eval/tests/results \
  --report-prefix arena_eval
```

Outputs:

```text
skills/arena-eval/tests/results/arena_eval_functional_report.md
skills/arena-eval/tests/results/arena_eval_functional_results.json
```

## Recommended Test Samples

| Purpose | Case ID | Why |
|---|---|---|
| Router: citation-specific | `arena_router_002_citation_hallucination_benchmark` | Ensures citation-fabrication requests go to `02-ref-hallucination-arena`, not `01-auto-arena`. |
| Router: "better" trap | `arena_router_003_better_at_recommending_papers` | Ensures phrasing as "better" (not "hallucinates less") still routes to the verifiable workflow. |
| Router: out-of-scope redirect | `arena_router_004_single_document_not_arena` | Ensures a single-document (not multi-model) request is redirected to `academic-eval`. |
| Auto arena: flag choice | `auto_arena_002_rerun_judge_only` | Ensures `--rerun-judge` is recommended over `--fresh` when only the judge changes. |
| Ref arena: tiebreak | `ref_arena_arena_002_ranking_tiebreak_order` | Ensures the documented tiebreak order is used, not an arbitrary one. |
| End-to-end | `arena_eval_001_end_to_end` | Ensures a request spanning both workflows isn't collapsed into one run. |

## Handling Variance

Use `--repeat 3` before claiming a skill change worked — actor and judge are
both LLMs, so single-run verdicts near a boundary are noisy.

## How To Interpret Results

- `pass`: all critical acceptance criteria met.
- `partial`: mostly correct but missed a flag, field, or caveat.
- `fail`: routed incorrectly or gave a misleading answer.

Useful pass threshold for a first audit: smoke set 5/5 pass or partial with at
least 4 pass; full set at least 80% pass.

## What To Do After A Failure

1. Read the actor output in the Markdown report.
2. Compare it to the missed acceptance criteria.
3. Patch the relevant `SKILL.md` (or the router's triage table / key-distinction
   section) with a clearer rule.
4. Re-run only that case, then the smoke set.

## Why a shared runner

Same rationale as `academic-eval`: the runner at
`skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py` is parametrized
via `--skill-root` / `--cases` / `--out-dir` / `--report-prefix` and isn't
subject to the Agent Skill protocol's independent-installability constraint
(that constraint is about `SKILL.md` content, not internal test tooling), so
copying it per suite would only create three scripts to keep in sync instead
of one.

# Academic Eval Functional Testing Guide

This guide explains how to prove the `academic-eval` skills work on realistic
scenarios. It follows the same actor+judge functional-test pattern as
[`skills/eval_pipeline/tests/`](../../eval_pipeline/tests/eval_pipeline_testing_guide.md) —
this suite reuses that suite's runner rather than shipping a duplicate one
(see [Why a shared runner](#why-a-shared-runner) below).

## What To Test

1. Skill package quality test
   - Use `cookbooks/skills_evaluation/evaluate_skills.py`.
   - Grades whether the SKILL.md files are complete, relevant, safe, and well-designed.
2. Functional scenario test (this guide)
   - Use `skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py` pointed at
     `skills/academic-eval`.
   - Checks whether a skill actually guides an agent to the right recommendation
     (routing) or the right concrete answer (usage/troubleshooting/interpretation)
     for a realistic user request.

## API Key

Same as `eval_pipeline`: two model roles, actor (follows the skill) and judge
(grades against acceptance criteria), defaulting to different models so one
model doesn't grade its own output.

```bash
# .env at repo root — Aliyun DashScope (Bailian)
DASHSCOPE_API_KEY=sk-...

# or, any OpenAI-compatible provider
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://your-provider.example/v1"   # omit for OpenAI itself
```

## Quick Smoke Test

Run one router case first:

```bash
python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py \
  --skill-root skills/academic-eval \
  --cases skills/academic-eval/tests/academic_eval_test_cases.jsonl \
  --out-dir skills/academic-eval/tests/results \
  --report-prefix academic_eval \
  --case-id academic_router_001_route_paper_review
```

Run a representative smoke set (one case per skill + the end-to-end case):

```bash
python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py \
  --skill-root skills/academic-eval \
  --cases skills/academic-eval/tests/academic_eval_test_cases.jsonl \
  --out-dir skills/academic-eval/tests/results \
  --report-prefix academic_eval \
  --case-id academic_router_002_bib_only_no_paper \
  --case-id paper_review_002_bad_request_base_url_diagnosis \
  --case-id bib_verify_002_interpret_suspect_entry \
  --case-id ref_arena_002_interpret_low_accuracy \
  --case-id academic_eval_001_end_to_end_route_and_execute
```

Run all cases:

```bash
python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py \
  --skill-root skills/academic-eval \
  --cases skills/academic-eval/tests/academic_eval_test_cases.jsonl \
  --out-dir skills/academic-eval/tests/results \
  --report-prefix academic_eval
```

Outputs:

```text
skills/academic-eval/tests/results/academic_eval_functional_report.md
skills/academic-eval/tests/results/academic_eval_functional_results.json
```

(`results/` is git-ignored — it regenerates on every run; the Markdown report
is for local/PR-description review, not something committed.)

## Recommended Test Samples

| Purpose | Case ID | Why |
|---|---|---|
| Router: paper vs bib-only | `academic_router_002_bib_only_no_paper` | Ensures a bib-only request doesn't get routed to the full review pipeline. |
| Router: combined request | `academic_router_003_paper_plus_bib` | Ensures "review + verify refs" collapses into one `01-paper-review` run, not two workflows. |
| Router: out-of-scope redirect | `academic_router_005_general_quality_not_citations` | Ensures the router doesn't force an unrelated request into this suite. |
| Paper review: troubleshooting | `paper_review_002_bad_request_base_url_diagnosis` | Ensures the skill diagnoses the `/v1/chat/completions` vs `/v1` bug instead of bypassing the pipeline. |
| Bib verify: interpretation | `bib_verify_002_interpret_suspect_entry` | Ensures a `suspect` entry isn't waved through as verified. |
| Ref arena: interpretation | `ref_arena_002_interpret_low_accuracy` | Ensures 45% accuracy is graded "Fair", not "Good". |
| End-to-end | `academic_eval_001_end_to_end_route_and_execute` | Ensures a request spanning two workflows isn't collapsed into one. |

## Handling Variance

Both actor and judge are LLMs — a single run is noisy. Use `--repeat 3` before
claiming a skill change worked:

```bash
python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py \
  --skill-root skills/academic-eval \
  --cases skills/academic-eval/tests/academic_eval_test_cases.jsonl \
  --out-dir skills/academic-eval/tests/results \
  --report-prefix academic_eval \
  --case-id academic_router_003_paper_plus_bib --repeat 3
```

## How To Interpret Results

- `pass`: the skill gave enough guidance to satisfy all critical acceptance criteria.
- `partial`: the skill mostly worked but missed an artifact, caveat, or field.
- `fail`: the skill routed incorrectly or gave a misleading answer.

Useful pass threshold for a first audit: smoke set 5/5 pass or partial with at
least 4 pass; full set at least 80% pass.

## What To Do After A Failure

1. Read the actor output in the Markdown report.
2. Compare it to the missed acceptance criteria.
3. Patch the relevant `SKILL.md` (or the router's triage table) with a clearer
   rule or example.
4. Re-run only that case, then the smoke set.

## Why a shared runner

`skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py` is suite-agnostic
via `--skill-root` / `--cases` / `--out-dir` / `--report-prefix` — it derives
its own file paths from `__file__`, not from being inside a specific suite.
Copying the ~400-line runner into every suite's `tests/` folder would triple
the actor/judge prompt logic to keep in sync for no benefit: the runner is
internal test tooling, not something bound by the Agent Skill protocol's
independent-installability rule (that rule applies to `SKILL.md` content, not
to how the maintainers verify it). Only the JSONL fixture and this guide are
suite-specific.

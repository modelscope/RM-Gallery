# Arena Eval Skill Audit

Initial functional-test audit of the `arena-eval` suite, run right after the
suite was created (folder-migrated from `auto-arena` + a fresh copy of
`ref-hallucination-arena` + new `00-arena-router`).

## Run 1 — full set, `--repeat 1`

```bash
python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py \
  --skill-root skills/arena-eval \
  --cases skills/arena-eval/tests/arena_eval_test_cases.jsonl \
  --out-dir skills/arena-eval/tests/results \
  --report-prefix arena_eval
```

Actor: `qwen3.6-plus` · Judge: `qwen3-max`

**Result: 9/9 pass, 0 partial, 0 fail.**

| Case | Skill | Verdict | Score |
|---|---|---:|---:|
| `arena_router_001_custom_task_comparison` | `00-arena-router` | pass | 1.00 |
| `arena_router_002_citation_hallucination_benchmark` | `00-arena-router` | pass | 1.00 |
| `arena_router_003_better_at_recommending_papers` | `00-arena-router` | pass | 1.00 |
| `arena_router_004_single_document_not_arena` | `00-arena-router` | pass | 1.00 |
| `auto_arena_001_minimal_config_two_endpoints` | `01-auto-arena` | pass | 1.00 |
| `auto_arena_002_rerun_judge_only` | `01-auto-arena` | pass | 1.00 |
| `ref_arena_arena_001_dataset_format_check` | `02-ref-hallucination-arena` | pass | 1.00 |
| `ref_arena_arena_002_ranking_tiebreak_order` | `02-ref-hallucination-arena` | pass | 1.00 |
| `arena_eval_001_end_to_end` | `arena_eval_collection` | pass | 1.00 |

## Run 2 — variance check, `--repeat 3` on the three most ambiguity-prone cases

```bash
python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py \
  --skill-root skills/arena-eval \
  --cases skills/arena-eval/tests/arena_eval_test_cases.jsonl \
  --out-dir /tmp/arena_eval_repeat_check --report-prefix arena_eval \
  --case-id arena_router_003_better_at_recommending_papers \
  --case-id arena_router_004_single_document_not_arena \
  --case-id ref_arena_arena_002_ranking_tiebreak_order \
  --repeat 3
```

- `arena_router_003_better_at_recommending_papers`: pass / pass / pass — score 1.00 (n=3)
- `arena_router_004_single_document_not_arena`: pass / pass / pass — score 1.00 (n=3)
- `ref_arena_arena_002_ranking_tiebreak_order`: pass / pass / pass — score 1.00 (n=3)

No variance observed across 3 runs on any of the three boundary cases,
including the deliberately adversarial "better" phrasing case designed to
tempt a router into judge-preference framing over the verifiable workflow.

## Reading

Same reading as `academic-eval` (see
[`../../academic-eval/tests/academic_eval_skill_audit.md`](../../academic-eval/tests/academic_eval_skill_audit.md#reading)):
a clean 15/15 pass across both full runs and repeat checks is a good initial
signal but was authored and graded in the same sitting as the skill content
being tested, so it should be treated as a smoke signal, not a substitute for
independently authored adversarial cases or a different actor/judge pairing
down the line.

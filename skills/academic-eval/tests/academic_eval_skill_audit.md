# Academic Eval Skill Audit

Initial functional-test audit of the `academic-eval` suite, run right after the
suite was created (folder-migrated from `paper-review`/`bib-verify`/
`ref-hallucination-arena` + new `00-academic-router`).

## Run 1 — full set, `--repeat 1`

```bash
python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py \
  --skill-root skills/academic-eval \
  --cases skills/academic-eval/tests/academic_eval_test_cases.jsonl \
  --out-dir skills/academic-eval/tests/results \
  --report-prefix academic_eval
```

Actor: `qwen3.6-plus` · Judge: `qwen3-max`

**Result: 12/12 pass, 0 partial, 0 fail.**

| Case | Skill | Verdict | Score |
|---|---|---:|---:|
| `academic_router_001_route_paper_review` | `00-academic-router` | pass | 1.00 |
| `academic_router_002_bib_only_no_paper` | `00-academic-router` | pass | 1.00 |
| `academic_router_003_paper_plus_bib` | `00-academic-router` | pass | 1.00 |
| `academic_router_004_arena_benchmark_request` | `00-academic-router` | pass | 1.00 |
| `academic_router_005_general_quality_not_citations` | `00-academic-router` | pass | 1.00 |
| `paper_review_001_model_fallback_no_explicit_model` | `01-paper-review` | pass | 1.00 |
| `paper_review_002_bad_request_base_url_diagnosis` | `01-paper-review` | pass | 1.00 |
| `bib_verify_001_standalone_bib_zh_report` | `02-bib-verify` | pass | 1.00 |
| `bib_verify_002_interpret_suspect_entry` | `02-bib-verify` | pass | 1.00 |
| `ref_arena_001_dataset_format_check` | `03-ref-hallucination-arena` | pass | 1.00 |
| `ref_arena_002_interpret_low_accuracy` | `03-ref-hallucination-arena` | pass | 1.00 |
| `academic_eval_001_end_to_end_route_and_execute` | `academic_eval_collection` | pass | 1.00 |

## Run 2 — variance check, `--repeat 3` on the two most ambiguity-prone router cases

```bash
python skills/eval_pipeline/tests/run_eval_pipeline_skill_tests.py \
  --skill-root skills/academic-eval \
  --cases skills/academic-eval/tests/academic_eval_test_cases.jsonl \
  --out-dir /tmp/academic_eval_repeat_check --report-prefix academic_eval \
  --case-id academic_router_003_paper_plus_bib \
  --case-id academic_router_005_general_quality_not_citations \
  --repeat 3
```

- `academic_router_003_paper_plus_bib`: pass / pass / pass — score 1.00 (n=3)
- `academic_router_005_general_quality_not_citations`: pass / pass / pass — score 1.00 (n=3)

No variance observed across 3 runs on either boundary case.

## Reading

The suite's 4 skills (router + 3 migrated originals) pass their functional
tests immediately after the folder migration and router rewrite — no
follow-up SKILL.md patch was needed. The router's triage-table wording for
the two trickiest rows (combined paper+bib request; out-of-scope redirect to
`arena-eval`) held up under repeat sampling.

**Caveat**: 21/21 clean passes across two suites on a first run is a stronger
result than typical for a brand-new skill collection (`eval_pipeline`'s own
first audit — see `../../eval_pipeline/tests/eval_pipeline_skill_audit.md` —
needed several fix rounds). Two contributing factors, not just "this suite
is more polished": (1) most of the SKILL.md content is unchanged domain
documentation carried over from the pre-migration skills, which had already
been used in practice; (2) all 12 test cases here were authored by the same
person who wrote the router content in the same sitting, so criteria and
skill text may share blind spots that an independently-written test set or a
different actor/judge model pairing would catch. Re-running with `--repeat
5`, a different judge model, or independently authored adversarial cases is
the natural next step before treating this as a strong reliability claim
rather than an initial smoke signal.

# Academic Eval — paper review & citation accuracy

A set of skills for academic-paper workflows built on OpenJudge: reviewing a
paper end-to-end, spot-checking a bibliography for fabricated references, or
benchmarking how often LLMs hallucinate citations at scale.

Each skill is a self-contained workflow in `<NN-name>/SKILL.md`. Start at
`00-academic-router` if you're not sure which one you need.

## The workflows

| # | Skill | Use it when |
|---|---|---|
| 00 | `00-academic-router` | You're not sure whether you want a paper review, a BibTeX check, or an arena-style benchmark. |
| 01 | `01-paper-review` | You have a paper (PDF or LaTeX source) and want a multi-stage review — safety, correctness, quality/novelty, criticality, optionally + BibTeX. |
| 02 | `02-bib-verify` | You have a standalone `.bib` file (no paper) and want to check it for fabricated/mismatched references. |
| 03 | `03-ref-hallucination-arena` | You want to benchmark/compare multiple LLMs on how often they invent fake citations, across many queries. |

## Relationship between the workflows

`01-paper-review` and `02-bib-verify` both run on top of the same
`cookbooks.paper_review` pipeline — `01` is the full document review (with an
optional `--bib` flag to also verify references), `02` is the BibTeX-only mode
for when there's no paper to review, just a bibliography to sanity-check.

`03-ref-hallucination-arena` is a different axis entirely: instead of
evaluating one document's existing references, it evaluates *model behavior* —
how often a model fabricates references when asked to recommend citations,
scored across a benchmark of queries and ranked across models. It shares
`ref-hallucination-arena`'s content with the `arena-eval` suite's
`02-ref-hallucination-arena` — the two copies are independently maintained
(see the root [`skills/README.md`](../README.md) for why).

## Dependencies

| Skill | Cookbook |
|---|---|
| `01-paper-review` | `cookbooks/paper_review/` |
| `02-bib-verify` | `cookbooks/paper_review/` (BibTeX-only mode) |
| `03-ref-hallucination-arena` | `cookbooks/ref_hallucination_arena/` |

```bash
pip install py-openjudge litellm
pip install matplotlib   # only needed by 03-ref-hallucination-arena (charts)
```

## Self-contained skills

Each `<NN-name>/SKILL.md` is self-contained per the Anthropic Agent Skill
protocol — it can be installed and used on its own without this README or the
rest of the suite. Cross-references between skills in this suite use relative
links scoped to this directory (e.g. `02-bib-verify` links to
`../01-paper-review/`); there are no links out to other suites.

## Validating the skills themselves

This suite has an actor+judge functional test harness, same pattern as
`eval_pipeline`: an *actor* model follows a skill to answer a realistic user
request, and a separate *judge* model grades the answer against the case's
acceptance criteria. See [`tests/academic_eval_testing_guide.md`](tests/academic_eval_testing_guide.md)
for how to run it — it reuses `eval_pipeline`'s runner (parametrized, not
duplicated) against this suite's own 12 test cases in
[`tests/academic_eval_test_cases.jsonl`](tests/academic_eval_test_cases.jsonl).

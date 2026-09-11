# Arena Eval — model/agent arena-style comparison

Skills for comparing multiple LLMs or agents head-to-head using OpenJudge: a
general-purpose win-rate arena for any custom task, or a citation-accuracy
arena that verifies references against real bibliographic databases.

Each skill is a self-contained workflow in `<NN-name>/SKILL.md`. Start at
`00-arena-router` if you're not sure which one you need.

## The workflows

| # | Skill | Use it when |
|---|---|---|
| 00 | `00-arena-router` | You're not sure whether you want a generic arena or a citation-hallucination benchmark. |
| 01 | `01-auto-arena` | Zero-data, custom-task comparison: generates queries, collects responses, judges pairwise, ranks by win rate. |
| 02 | `02-ref-hallucination-arena` | Benchmarks how often models fabricate citations, verified against CrossRef/PubMed/arXiv/DBLP. |

## Relationship between the workflows

Both produce model rankings from pairwise/verified comparison, but differ in
what counts as "correct": `01-auto-arena` relies on LLM-judge preference for
an arbitrary task (no ground truth needed); `02-ref-hallucination-arena`
checks references against real databases, so the ranking is grounded in
verifiable fact rather than judge opinion, at the cost of being scoped to
citation recommendation only.

`02-ref-hallucination-arena` shares its content with the `academic-eval`
suite's `03-ref-hallucination-arena` — the two copies are independently
maintained (see the root [`skills/README.md`](../README.md) for why), so this
one can lean more on "how does this model compare to others" framing while
the academic-eval copy leans on "how trustworthy are this document's
citations."

## Dependencies

| Skill | Cookbook |
|---|---|
| `01-auto-arena` | `cookbooks/auto_arena/` |
| `02-ref-hallucination-arena` | `cookbooks/ref_hallucination_arena/` |

```bash
pip install py-openjudge
pip install matplotlib   # chart generation, needed by both skills
```

## Self-contained skills

Each `<NN-name>/SKILL.md` is self-contained per the Anthropic Agent Skill
protocol — it can be installed and used on its own without this README or the
rest of the suite.

## Validating the skills themselves

This suite has an actor+judge functional test harness, same pattern as
`eval_pipeline`: an *actor* model follows a skill to answer a realistic user
request, and a separate *judge* model grades the answer against the case's
acceptance criteria. See [`tests/arena_eval_testing_guide.md`](tests/arena_eval_testing_guide.md)
for how to run it — it reuses `eval_pipeline`'s runner (parametrized, not
duplicated) against this suite's own 10 test cases in
[`tests/arena_eval_test_cases.jsonl`](tests/arena_eval_test_cases.jsonl).

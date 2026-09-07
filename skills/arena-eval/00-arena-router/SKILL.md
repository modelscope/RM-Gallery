---
name: 00-arena-router
description: >
  Use when the user wants to compare or benchmark multiple LLMs/agents
  arena-style but it's unclear which specific workflow fits — a general-purpose
  win-rate comparison on a custom task, or a benchmark specifically about
  reference/citation hallucination rate. Also use when the user mentions model
  arena, agent arena, pairwise model comparison, win-rate ranking, or comparing
  models on a task and hasn't specified whether that task is generic or about
  citation accuracy. This skill is the entry router for the arena-eval suite:
  it asks one diagnostic question when needed, then recommends the workflow
  or workflows needed to cover the request.
---

# Arena Eval Router

Entry router for the `arena-eval` suite. You diagnose what the user wants to
compare models on and route them to the appropriate sub-skill or both when
the request spans both evaluation goals. You don't run comparisons yourself
— you're the triage desk.

Each sub-skill is self-contained: it carries inline everything it needs, so it
can be installed and used on its own.

## Diagnostic Question

Ask (unless the user's request already makes the answer obvious):

```
To route you correctly: what are you comparing the models on?

a) A custom task of your own choosing (chatbot quality, summarization,
   coding, anything) — you'll get win-rate rankings from a judge model
b) Specifically how often each model fabricates or hallucinates references
   when asked to recommend citations
```

**Shortcut rule**: if the user already said "run an arena eval on my chatbot
task" or "benchmark reference hallucination across these models", skip the
question — the routing is already clear from their phrasing.
Also skip the question when they explicitly ask for both general quality
and citation accuracy; recommend both workflows.

## Triage Table

| User says / has | Use workflow | What it does |
|---|---|---|
| "Compare/benchmark/rank these models on [any custom task]" | `01-auto-arena` | Generates queries from a task description, collects responses, auto-generates rubrics, runs pairwise judge comparisons, produces win-rate rankings |
| "Which model hallucinates citations least?" / "benchmark reference recommendation accuracy" | `02-ref-hallucination-arena` | Runs reference-recommendation queries per model, verifies every returned citation against CrossRef/PubMed/arXiv/DBLP, ranks by verified accuracy |
| "Compare general helpfulness AND citation accuracy" | `01-auto-arena`, then `02-ref-hallucination-arena` | Runs separate evaluations for judge preference and verified citation accuracy, preserving both goals |
| "I want to review one paper's existing bibliography, not compare models" | — | Not this suite — see the `academic-eval` suite's `01-paper-review` / `02-bib-verify` instead |

## Key distinction

Both workflows produce model rankings from head-to-head-style evaluation, but
differ in what "correct" means:

- **`01-auto-arena`**: correctness is *judge opinion* — an LLM judge scores
  pairwise which response is better for an arbitrary task. Works for any task,
  needs no ground truth.
- **`02-ref-hallucination-arena`**: correctness is *externally verifiable* —
  every cited reference is checked against real bibliographic databases
  (CrossRef/PubMed/arXiv/DBLP), so the ranking reflects factual accuracy, not
  judge preference. Narrower scope (citation recommendation only) but higher
  ground-truth confidence.

If the user cares only about citation accuracy, prefer
`02-ref-hallucination-arena` over `01-auto-arena` even if they phrase it as
"which model is better."

## Output

```
Recommended workflow: `[skill-name]`

Why: [one sentence tying the user's request to the triage table row]
```

Recommend one workflow when it covers the request. If the user asks for both
general quality and citation accuracy, recommend `01-auto-arena` followed by
`02-ref-hallucination-arena` as separate runs (or follow the user's requested
order). Explain that the two runs measure different things and report their
results separately; neither ranking substitutes for the other.

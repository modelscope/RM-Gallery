---
name: arena-router
description: >
  Use when the user wants to compare or benchmark multiple LLMs/agents
  arena-style but it's unclear which specific workflow fits — a general-purpose
  win-rate comparison on a custom task, or a benchmark specifically about
  reference/citation hallucination rate. Also use when the user mentions model
  arena, agent arena, pairwise model comparison, win-rate ranking, or comparing
  models on a task and hasn't specified whether that task is generic or about
  citation accuracy. This skill is the entry router for the arena-eval suite:
  it asks one diagnostic question then routes to the right sub-skill.
---

# Arena Eval Router

Entry router for the `arena-eval` suite. You diagnose what the user wants to
compare models on and route them to one of two sub-skills. You don't run
comparisons yourself — you're the triage desk.

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

## Triage Table

| User says / has | Use workflow | What it does |
|---|---|---|
| "Compare/benchmark/rank these models on [any custom task]" | `01-auto-arena` | Generates queries from a task description, collects responses, auto-generates rubrics, runs pairwise judge comparisons, produces win-rate rankings |
| "Which model hallucinates citations least?" / "benchmark reference recommendation accuracy" | `02-ref-hallucination-arena` | Runs reference-recommendation queries per model, verifies every returned citation against CrossRef/PubMed/arXiv/DBLP, ranks by verified accuracy |
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

If the user cares about factual/citation accuracy specifically, prefer
`02-ref-hallucination-arena` over `01-auto-arena` even if they phrase it as
"which model is better."

## Output

```
Recommended workflow: `[skill-name]`

Why: [one sentence tying the user's request to the triage table row]
```

Recommend exactly one workflow per request.

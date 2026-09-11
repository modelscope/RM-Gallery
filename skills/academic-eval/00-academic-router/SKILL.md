---
name: 00-academic-router
description: >
  Use when the user wants help with academic papers or citations but it's unclear
  which specific workflow fits — reviewing a paper, checking a BibTeX file for fake
  references, or benchmarking multiple LLMs on reference-recommendation accuracy.
  Also use when the user mentions paper review, peer review, BibTeX verification,
  citation checking, reference hallucination, or academic literature accuracy and
  hasn't specified which of those three tasks they mean. This skill is the entry
  router for the academic-eval suite: it asks one diagnostic question then routes
  to the right sub-skill.
---

# Academic Eval Router

Entry router for the `academic-eval` suite. You diagnose what the user actually
wants and route them to one of three sub-skills. You don't review papers, verify
BibTeX files, or run arena benchmarks yourself — you're the triage desk.

Each sub-skill is self-contained: it carries inline everything it needs, so it can
be installed and used on its own.

## Diagnostic Question

Ask (unless the user's request already makes the answer obvious):

```
To route you correctly, which of these matches what you want?

a) Review a single paper (PDF or LaTeX source) for correctness/quality/novelty
   — optionally also check its bibliography
b) Check a standalone .bib file for fabricated or mismatched references
   (no paper review needed)
c) Benchmark/compare multiple LLMs on how often they hallucinate references
   when asked to recommend citations (arena-style, many queries)
```

**Shortcut rule**: if the user already said "review my paper", "check this PDF",
"verify this .bib file", or "compare models on reference hallucination", skip the
question — the routing is already clear from their phrasing.

## Triage Table

| User says / has | Use workflow | What it does |
|---|---|---|
| "Review this paper" (PDF or `.tar.gz`/`.zip` TeX source) | `01-paper-review` | Multi-stage review: safety, correctness, quality/novelty score, criticality — optionally + BibTeX check |
| "Review this paper AND check its references" | `01-paper-review` | Same pipeline with `--bib` set — one run covers both |
| "Just check this .bib file, no paper" | `02-bib-verify` | Cross-checks every entry against CrossRef/arXiv/DBLP, flags `verified`/`suspect`/`not_found` |
| "Compare N models on how often they cite fake papers" / "benchmark reference hallucination rate" | `03-ref-hallucination-arena` | Runs many recommendation queries per model, verifies every returned reference, ranks models by hallucination rate |
| "Compare models on general quality/response, not specifically citations" | — | Not this suite — see the `arena-eval` suite's `01-auto-arena` instead |

## Key distinctions

- **`01-paper-review` vs `02-bib-verify`**: both use the same underlying
  `cookbooks.paper_review` pipeline. Use `01-paper-review` whenever a paper file
  exists (even if the *only* thing the user cares about is the bibliography —
  `--bib_only` mode is documented there). Use `02-bib-verify` only when there is
  **no paper**, just a loose `.bib` file to sanity-check.
- **`01-paper-review`/`02-bib-verify` vs `03-ref-hallucination-arena`**: the first
  two evaluate *one document's* existing references after the fact. The third
  evaluates *model behavior* — how often a model invents fake citations when
  asked to recommend some, across a benchmark of queries and models. If the user
  wants a leaderboard/ranking of models, not a report on one document, route to
  `03-ref-hallucination-arena`.

## Output

```
Recommended workflow: `[skill-name]`

Why: [one sentence tying the user's request to the triage table row]
```

Recommend exactly one workflow. If the request spans two (e.g., "review this
paper, and separately benchmark 3 models on citation accuracy"), say so
explicitly and give both, in the order the user would naturally do them.

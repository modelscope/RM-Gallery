# Skills

This directory holds Agent Skills (Anthropic Agent Skill protocol — YAML
frontmatter + Markdown body) for OpenJudge. Every `SKILL.md`, wherever it
lives, is **independently installable**: it carries everything it needs
inline, so a single `<name>/SKILL.md` folder can be pulled out and installed
on its own in Claude Code, Cursor, Codex, Hermes, OpenClaw, or any other
client that speaks the same protocol — you never need this README or a
suite's own README to make one skill work.

Grouping below **is folder structure only** — there is no code-level registry
(no `SUITE_REGISTRY`, no `DomainSuite` class). A "domain suite" is just a
directory of related `<NN-name>/SKILL.md` workflows plus a `README.md` that
explains how they relate.

## Domain suites (multi-skill, folder-grouped)

| Suite | Skills | Dimension | What it covers |
|---|---|---|---|
| [`eval_pipeline/`](eval_pipeline/) | 9 (`00-meta-eval` … `08-bootstrap`) | Horizontal methodology | How to build an evaluation system from scratch — dataset design, metric/grader selection, human-alignment calibration, reporting, plus RAG/prompt-regression/redteam/bootstrap scenarios. Router at `00-meta-eval`. |
| [`academic-eval/`](academic-eval/) | 4 (`00-academic-router`, `01-paper-review`, `02-bib-verify`, `03-ref-hallucination-arena`) | Vertical domain | Academic paper review, BibTeX verification, and citation-hallucination benchmarking. |
| [`arena-eval/`](arena-eval/) | 3 (`00-arena-router`, `01-auto-arena`, `02-ref-hallucination-arena`) | Vertical domain | Arena-style model/agent comparison — generic win-rate ranking, and citation-hallucination-specific ranking. |
| [`openjudge-core/`](openjudge-core/) | 2 (`01-graders-and-pipeline`, `02-rl-reward`) | Horizontal methodology | Direct OpenJudge core-library API: graders/`GradingRunner`/aggregators, and RL reward-signal construction. No router — see the suite's own README for why. |

`eval_pipeline` is the range-of-motion paradigm all the other suites above
were reshaped to match (three-layer structure, directory-as-grouping, router
only where sub-skills are genuinely ambiguous, every `<NN-name>/SKILL.md`
independently installable). See
[`docs/superpowers/specs/2026-07-09-skills-domain-suite-proposal.md`](../docs/superpowers/specs/2026-07-09-skills-domain-suite-proposal.md)
for the full design rationale.

### Functional test coverage

`eval_pipeline`, `academic-eval`, and `arena-eval` each have an actor+judge
functional test harness under their own `tests/` folder — a *runner*
(shared, parametrized: `eval_pipeline/tests/run_eval_pipeline_skill_tests.py
--skill-root <suite> --cases <suite>/tests/<suite>_test_cases.jsonl`), a
suite-specific JSONL fixture, and a testing guide. `openjudge-core` and the
isolated skills don't have this yet (see each suite's `README.md` under
"Validating the skills themselves" for the ones that do).

### `ref-hallucination-arena` — intentional duplication, not a bug

`academic-eval/03-ref-hallucination-arena/SKILL.md` and
`arena-eval/02-ref-hallucination-arena/SKILL.md` start as full copies of the
same content and are **independently maintained** — no shared `references/`
directory, no symlink, no sync script. This is a direct consequence of the
independent-installability constraint: a skill can't rely on content that
lives outside its own folder. The two copies are free to diverge (the
academic copy can lean into "how trustworthy are this paper's citations",
the arena copy into "how does this model compare to others"); duplicated
content is an accepted cost, not a problem to solve.

## Isolated skills (single-skill, no suite directory)

These have no other skill in this repo referencing them, so they stay at
`skills/<name>/SKILL.md` with no suite folder — the simplest possible
grouping (1 skill = 1 domain).

| Skill | What it covers |
|---|---|
| [`claude-authenticity/`](claude-authenticity/) | Detect whether an API endpoint is genuinely backed by Claude (vs. a wrapper/proxy/impersonator); extract injected system prompts. Zero OpenJudge dependency. |
| [`mmx-cli/`](mmx-cli/) | Generate text/image/video/speech/music via the MiniMax AI platform. Third-party CLI wrapper, zero OpenJudge dependency. |
| [`find-skills-combo/`](find-skills-combo/) | Discover and recommend combinations of skills from the **external** open agent-skills ecosystem to cover a multi-part task. Unrelated to this repo's own suite grouping. |

**Future rule**: if an isolated skill above ever becomes something a domain
suite depends on or wraps, move the whole folder into that suite's directory
(same copy-don't-link approach as `ref-hallucination-arena` above, or a
rename+move if a single owner is clear) rather than keeping it independent
and cross-linking from two places.

## Breaking change: path migration (this reorg)

This reorg moved 6 previously-published skills to new install paths. There
are no redirect stubs — installing the old path will 404; use the new path.

| Old path | New path |
|---|---|
| `skills/paper-review/` | `skills/academic-eval/01-paper-review/` |
| `skills/bib-verify/` | `skills/academic-eval/02-bib-verify/` |
| `skills/ref-hallucination-arena/` | `skills/academic-eval/03-ref-hallucination-arena/` **and** `skills/arena-eval/02-ref-hallucination-arena/` (two independent copies — see above) |
| `skills/auto-arena/` | `skills/arena-eval/01-auto-arena/` |
| `skills/openjudge/` | `skills/openjudge-core/01-graders-and-pipeline/` |
| `skills/rl-reward/` | `skills/openjudge-core/02-rl-reward/` |

The migrated skills' frontmatter `name` values also change to match their
new directory names. Update explicit skill invocations to use the final
component of each new path (for example, `paper-review` becomes
`01-paper-review`, and `openjudge` becomes `01-graders-and-pipeline`). The
two reference-arena copies use distinct names: `03-ref-hallucination-arena`
in `academic-eval` and `02-ref-hallucination-arena` in `arena-eval`.

`SkillLoader.load_from_directory("skills")` discovers skills recursively
through suite directories. A directory containing `SKILL.md` is treated as
one package, including its bundled references and examples.

`skills/eval_pipeline/`, `skills/claude-authenticity/`, `skills/mmx-cli/`,
and `skills/find-skills-combo/` are unaffected.

## Adding a new skill

- **Fits an existing suite's methodology or domain?** Add it as
  `<suite>/<NN-name>/SKILL.md` with `name: NN-name` matching its directory,
  choose a name unique across the repository, update that suite's `README.md`,
  and add a router entry only if it creates real selection ambiguity with a
  sibling (see each suite's README for its router-inclusion rationale).
- **Genuinely stands alone?** Add it as `skills/<name>/SKILL.md` — no suite
  folder needed.
- **Either way**, the skill's `SKILL.md` must be installable on its own: only
  `name` + `description` in frontmatter, no links outside its own folder (or
  its own suite folder), and any non-`openjudge`/`rl-reward` pip dependency
  listed explicitly under Prerequisites.

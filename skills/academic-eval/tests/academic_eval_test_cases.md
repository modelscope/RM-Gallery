# Academic Eval Functional Test Cases (Summary)

Source fixture: [`academic_eval_test_cases.jsonl`](academic_eval_test_cases.jsonl)
(the runner reads only the JSONL; this table is for human review). 12 cases
across the suite's 4 skills.

| ID | Skill | Category | Request (abridged) | Key acceptance criterion |
|---|---|---|---|---|
| `academic_router_001_route_paper_review` | `00-academic-router` | routing | Full PDF review request | Routes to `01-paper-review` only |
| `academic_router_002_bib_only_no_paper` | `00-academic-router` | routing | Only a `.bib`, no paper | Routes to `02-bib-verify`, not `01-paper-review` |
| `academic_router_003_paper_plus_bib` | `00-academic-router` | routing | Review paper + verify its refs | One `01-paper-review` run with `--bib`, not two workflows |
| `academic_router_004_arena_benchmark_request` | `00-academic-router` | routing | Benchmark 4 LLMs on citation fabrication | Routes to `03-ref-hallucination-arena` |
| `academic_router_005_general_quality_not_citations` | `00-academic-router` | routing | Generic chatbot quality comparison | Redirects to `arena-eval`'s `01-auto-arena` |
| `paper_review_001_model_fallback_no_explicit_model` | `01-paper-review` | model_selection | No model given, only DashScope key | Picks `dashscope/qwen-vl-plus` (vision-capable) |
| `paper_review_002_bad_request_base_url_diagnosis` | `01-paper-review` | troubleshooting | 400 error, bad `--base_url` suffix | Diagnoses `/v1/chat/completions` vs `/v1`, re-runs pipeline |
| `bib_verify_001_standalone_bib_zh_report` | `02-bib-verify` | usage | Bib-only check, Chinese report + email | Correct `--bib_only`/`--email`/`--language zh` flags |
| `bib_verify_002_interpret_suspect_entry` | `02-bib-verify` | interpretation | `suspect` entry, title mismatch | Flags for manual check, not verified |
| `ref_arena_001_dataset_format_check` | `03-ref-hallucination-arena` | dataset_design | Plain `.txt` queries | Requires JSON/JSONL with `query` field |
| `ref_arena_002_interpret_low_accuracy` | `03-ref-hallucination-arena` | interpretation | 45% verification rate | Classifies as "Fair", not "Good" |
| `academic_eval_001_end_to_end_route_and_execute` | `academic_eval_collection` | end_to_end | Review + multi-model citation comparison | Names both `01-paper-review` and `03-ref-hallucination-arena` |

See [`academic_eval_testing_guide.md`](academic_eval_testing_guide.md) for how
to run these and interpret results.

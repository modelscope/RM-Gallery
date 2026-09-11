# Arena Eval Functional Test Cases (Summary)

Source fixture: [`arena_eval_test_cases.jsonl`](arena_eval_test_cases.jsonl)
(the runner reads only the JSONL; this table is for human review). 10 cases
across the suite's 3 skills.

| ID | Skill | Category | Request (abridged) | Key acceptance criterion |
|---|---|---|---|---|
| `arena_router_001_custom_task_comparison` | `00-arena-router` | routing | Compare 3 chatbot variants, generic quality | Routes to `01-auto-arena` |
| `arena_router_002_citation_hallucination_benchmark` | `00-arena-router` | routing | Which model fabricates fewest citations | Routes to `02-ref-hallucination-arena` |
| `arena_router_003_better_at_recommending_papers` | `00-arena-router` | routing | "Better" at recommending papers (ambiguous phrasing) | Still routes to `02-ref-hallucination-arena` (verifiable > judge opinion) |
| `arena_router_004_single_document_not_arena` | `00-arena-router` | routing | Check one paper's own bibliography | Redirects to `academic-eval` |
| `arena_router_005_combined_quality_and_citations` | `00-arena-router` | routing | Compare helpfulness AND citation fabrication | Router alone recommends both workflows in order, without dropping either goal |
| `auto_arena_001_minimal_config_two_endpoints` | `01-auto-arena` | usage | Compare gpt-4 vs qwen-max, have both keys | Config has `task`, both `target_endpoints`, `judge_endpoint` |
| `auto_arena_002_rerun_judge_only` | `01-auto-arena` | usage | Swap judge model, keep queries/responses | Recommends `--rerun-judge`, not `--fresh` |
| `ref_arena_arena_001_dataset_format_check` | `02-ref-hallucination-arena` | dataset_design | Plain `.txt` queries | Requires JSON/JSONL with `query` field |
| `ref_arena_arena_002_ranking_tiebreak_order` | `02-ref-hallucination-arena` | interpretation | Two models tied on accuracy | Applies documented tiebreak order (year compliance first) |
| `arena_eval_001_end_to_end` | `arena_eval_collection` | end_to_end | Compare on helpfulness AND citation hallucination | Names both `01-auto-arena` and `02-ref-hallucination-arena` |

See [`arena_eval_testing_guide.md`](arena_eval_testing_guide.md) for how to
run these and interpret results.

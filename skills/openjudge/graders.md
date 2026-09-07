# Graders Reference

Graders are the core evaluation units in OpenJudge.
Every grader inherits from `BaseGrader` and implements `async _aevaluate(**kwargs)`.

## Grader Types

| Type | Class | Best for |
|------|-------|----------|
| LLM-based | `LLMGrader` | Subjective quality, semantic understanding |
| Function-based | `FunctionGrader` | Exact rules, fast deterministic checks |
| Agentic | `AgenticGrader` | Evaluation requiring an autonomous coding agent (read a workspace/transcript, run code, verify rubric checkpoints) |

---

## Built-in Graders — Quick Reference

### `common/` — General-purpose (all LLM-based, POINTWISE, score 1–5)

| Class | Import | Key inputs | What it measures |
|-------|--------|------------|-----------------|
| `CorrectnessGrader` | `openjudge.graders.common.correctness` | `query`, `response`, `reference_response`, `context` | Factual match against reference |
| `HallucinationGrader` | `openjudge.graders.common.hallucination` | `query`, `response`, `context` | Fabricated/unsupported claims |
| `RelevanceGrader` | `openjudge.graders.common.relevance` | `query`, `response` | How relevant the response is |
| `HarmfulnessGrader` | `openjudge.graders.common.harmfulness` | `query`, `response` | Toxic or harmful content |
| `InstructionFollowingGrader` | `openjudge.graders.common.instruction_following` | `query`, `response` | Instruction compliance |

All `common/` graders accept `model` (required) and optional `threshold`, `language`, `strategy`.

```python
from openjudge.graders.common.hallucination import HallucinationGrader

grader = HallucinationGrader(model=model)
result = await grader.aevaluate(
    query="Who invented the telephone?",
    response="Thomas Edison invented the telephone in 1876.",
    context="Alexander Graham Bell is credited with the telephone (1876).",
)
# result.score: 1–5  (5 = no hallucination, 1 = severe hallucination)
```

---

### `text/` — String & Text Matching (no LLM needed)

| Class | Import | Key inputs | What it measures |
|-------|--------|------------|-----------------|
| `StringMatchGrader` | `openjudge.graders.text.string_match` | `response`, `reference_response` | Exact/regex/overlap matching |
| `SimilarityGrader` | `openjudge.graders.text.similarity` | `response`, `reference` | ROUGE / BM25 / embedding similarity |
| `NumberAccuracyGrader` | `openjudge.graders.text.number_accuracy` | `response`, `reference` | Numerical value accuracy |

**StringMatchGrader algorithms:** `exact_match`, `prefix_match`, `suffix_match`, `regex_match`,
`substring_match`, `contains_all`, `contains_any`, `word_overlap`, `char_overlap`

> **Important:** The algorithm must be set at **init time** via the `algorithm=` constructor
> argument. Passing `algorithm` in `aevaluate()` has **no effect** — the init value is always used.

```python
from openjudge.graders.text.string_match import StringMatchGrader

# Set algorithm at init time
grader = StringMatchGrader(algorithm="substring_match")
result = await grader.aevaluate(
    response="The capital is Paris.",
    reference_response="Paris",
)
# result.score: 1.0 (match) or 0.0 (no match)

# Different algorithm — create a new grader instance
grader_overlap = StringMatchGrader(algorithm="word_overlap")
result2 = await grader_overlap.aevaluate(
    response="The quick brown fox",
    reference_response="quick fox",
)
# result2.score: overlap ratio (0.0–1.0)
```

---

### `code/` — Code Evaluation

| Class | Import | Key inputs | What it measures |
|-------|--------|------------|-----------------|
| `CodeExecutionGrader` | `openjudge.graders.code.code_execution` | `response` | Test case pass rate (test cases from harness/metadata) |
| `SyntaxCheckGrader` | `openjudge.graders.code.syntax_checker` | `response` | Syntax validity |
| `CodeStyleGrader` | `openjudge.graders.code.code_style` | `response` | Style/lint quality |
| `PatchSimilarityGrader` | `openjudge.graders.code.patch_similarity` | `response`, `reference` | Patch/diff similarity |

```python
from openjudge.graders.code.code_execution import CodeExecutionGrader

grader = CodeExecutionGrader(timeout=10)
result = await grader.aevaluate(response="def add(a, b): return a + b")
# result.score: fraction of passed test cases (0.0–1.0).
# Test cases must be provided via sample metadata or external harness; see grader docs.
```

---

### `format/` — Output Format Validation

| Class | Import | Key inputs | What it measures |
|-------|--------|------------|-----------------|
| `JsonValidatorGrader` | `openjudge.graders.format.json.json_validator` | `response` | Is response valid JSON? |
| `JsonMatchGrader` | `openjudge.graders.format.json.json_match` | `response`, `reference` | JSON structure/content match |
| `LengthPenaltyGrader` | `openjudge.graders.format.length_penalty` | `response` | Penalizes over/under-length |
| `NgramRepetitionPenaltyGrader` | `openjudge.graders.format.ngram_repetition_penalty` | `response` | Penalizes repeated n-grams |
| `ReasoningFormatGrader` | `openjudge.graders.format.reasoning_format` | `response` | `<think>...</think>` format check |

```python
from openjudge.graders.format.json.json_validator import JsonValidatorGrader

grader = JsonValidatorGrader()
result = await grader.aevaluate(response='{"key": "value"}')
# result.score: 1.0 (valid JSON) or 0.0 (invalid)
```

---

### `math/` — Mathematical Expressions

| Class | Import | Key inputs | What it measures |
|-------|--------|------------|-----------------|
| `MathExpressionVerifyGrader` | `openjudge.graders.math.math_expression_verify` | `response`, `reference` | Mathematical equivalence |

---

### `agent/` — Agent Behavior Evaluation (all LLM-based)

| Category | Class | What it measures |
|----------|-------|-----------------|
| **Tool** | `ToolCallAccuracyGrader` | Whether tool calls are correct |
| **Tool** | `ToolCallSuccessGrader` | Whether tool calls succeeded |
| **Tool** | `ToolSelectionGrader` | Whether the right tool was chosen |
| **Tool** | `ToolParameterCheckGrader` | Correctness of tool parameters |
| **Tool** | `ToolCallStepSequenceMatchGrader` | Tool call order vs expected |
| **Tool** | `ToolCallPrecisionRecallMatchGrader` | Precision/recall of tool call set |
| **Memory** | `MemoryAccuracyGrader` | Accuracy of stored memory |
| **Memory** | `MemoryDetailPreservationGrader` | Detail retention in memory |
| **Memory** | `MemoryRetrievalEffectivenessGrader` | Quality of memory retrieval |
| **Plan** | `PlanFeasibilityGrader` | Whether the plan is feasible |
| **Reflection** | `ReflectionAccuracyGrader` | Accuracy of self-reflection |
| **Action** | `ActionAlignmentGrader` | Action alignment with intent |
| **Trajectory** | `TrajectoryAccuracyGrader` | Trajectory vs reference |
| **Trajectory** | `TrajectoryComprehensiveGrader` | End-to-end trajectory quality |

```python
from openjudge.graders.agent import ToolCallAccuracyGrader

grader = ToolCallAccuracyGrader(model=model)
result = await grader.aevaluate(
    query="Search for today's weather",
    tool_definitions=[{"name": "web_search", "description": "Search the web", "parameters": {}}],
    tool_calls=[{"name": "web_search", "arguments": {"query": "today weather"}}],
)
# result.score: 1–5 (tool call accuracy)
```

---

### `multi_turn/` — Multi-turn Conversation (all LLM-based)

| Class | What it measures |
|-------|-----------------|
| `ContextMemoryGrader` | Recalls details from early turns |
| `AnaphoraResolutionGrader` | Pronoun/reference resolution |
| `TopicSwitchGrader` | Handles sudden topic changes |
| `SelfCorrectionGrader` | Corrects errors when given feedback |
| `InstructionClarificationGrader` | Asks for clarification when needed |
| `ProactiveInteractionGrader` | Proactively engages in conversation |
| `ResponseRepetitionGrader` | Avoids repeating prior content |

```python
from openjudge.graders.multi_turn import ContextMemoryGrader

grader = ContextMemoryGrader(model=model)
result = await grader.aevaluate(
    history=[
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What's my name?"},
    ],
    response="Your name is Alice.",
)
```

---

### `multimodal/` — Vision & Image (requires VL model)

| Class | Import | What it measures |
|-------|--------|-----------------|
| `TextToImageGrader` | `openjudge.graders.multimodal.text_to_image` | Text-image alignment |
| `ImageCoherenceGrader` | `openjudge.graders.multimodal.image_coherence` | Image sequence coherence |
| `ImageHelpfulnessGrader` | `openjudge.graders.multimodal.image_helpfulness` | Image usefulness for context |

```python
from openjudge.models.qwen_vl_model import QwenVLModel
from openjudge.models.schema.qwen.mllmImage import MLLMImage
from openjudge.graders.multimodal.text_to_image import TextToImageGrader

vl_model = QwenVLModel(model="qwen-vl-plus", api_key="sk-xxx")
grader = TextToImageGrader(model=vl_model)
result = await grader.aevaluate(
    query="A red apple on a wooden table",
    response=MLLMImage(url="https://example.com/image.jpg"),
)
```

---

## LLMGrader — Custom Prompt Grader

Use `LLMGrader` directly when no built-in grader fits. Provide a template string with
`{placeholder}` variables that match your `aevaluate()` kwargs.

```python
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.schema import GraderMode

grader = LLMGrader(
    model=model,
    name="helpfulness",
    mode=GraderMode.POINTWISE,
    template="""
You are an evaluation assistant.

Query: {query}
Response: {response}

Rate the helpfulness of the response on a scale of 0.0 to 1.0.
Respond in JSON: {{"score": <float>, "reason": "<explanation>"}}
""",
)

result = await grader.aevaluate(
    query="How do I reverse a list in Python?",
    response="Use list.reverse() or reversed().",
)
# result.score, result.reason
```

### Listwise (ranking) mode

```python
ranking_grader = LLMGrader(
    model=model,
    name="quality_rank",
    mode=GraderMode.LISTWISE,
    template="""
Rank the following responses to the query from best (1) to worst.

Query: {query}
Response 1: {response_1}
Response 2: {response_2}

Respond in JSON: {{"rank": [<int>, <int>], "reason": "<explanation>"}}
""",
)

result = await ranking_grader.aevaluate(
    query="Explain gravity",
    response_1="Gravity is a fundamental force...",
    response_2="Things fall down.",
)
# result.rank e.g. [1, 2]  → response_1 is better
```

---

## FunctionGrader — Pure Python Evaluation

Use when the scoring logic is deterministic and requires no LLM.

```python
from functools import partial
from openjudge.graders.function_grader import FunctionGrader
from openjudge.graders.schema import GraderScore, GraderMode

def length_check(response: str, min_words: int = 10) -> GraderScore:
    word_count = len(response.split())
    score = 1.0 if word_count >= min_words else word_count / min_words
    return GraderScore(
        name="length_check",
        score=score,
        reason=f"Response has {word_count} words (min: {min_words})",
    )

# Option A: use functools.partial to bake in extra params
grader = FunctionGrader(
    func=partial(length_check, min_words=20),
    name="length_check",
    mode=GraderMode.POINTWISE,
)
result = await grader.aevaluate(response="Short answer.")

# Option B: pass extra params directly in aevaluate()
grader2 = FunctionGrader(func=length_check, name="length_check", mode=GraderMode.POINTWISE)
result2 = await grader2.aevaluate(response="Short answer.", min_words=20)
```

> **Note:** Extra `**kwargs` passed to `FunctionGrader(...)` at construction time are stored in `grader.kwargs` but are **not** automatically forwarded to `func`. Use `functools.partial` (Option A) or pass them directly to `aevaluate()` (Option B).

### Decorator syntax

```python
@FunctionGrader.wrap
def exact_match(response: str, reference: str) -> GraderScore:
    score = 1.0 if response.strip() == reference.strip() else 0.0
    return GraderScore(name="exact_match", score=score, reason="")

grader = exact_match(name="exact_match", mode=GraderMode.POINTWISE)
```

---

## AgenticGrader — Agent-as-judge Evaluation

Use when evaluation requires an autonomous agent that can inspect a candidate's produced
files and/or execution transcript and run code to verify claims, not just read text.
`AgenticGrader` shells out to a real external coding-agent CLI (Claude Code / Codex /
Cursor CLI) inside an isolated, throwaway sandbox — it never runs untrusted code
in-process. See `openjudge/harness/` for the harness implementations and
`cookbooks/agentic_judge/` for full runnable examples against each CLI.

Rubric names and checkpoint IDs must be unique across an evaluation. The constructor
rejects duplicates; invalid per-call rubric overrides return `GraderError`. Result
verdicts must use JSON booleans (`true`/`false`). Malformed checkpoint results are
dropped and count as failed, while invalid workspace paths return `GraderError`.

```python
from openjudge.graders.agentic_grader import AgenticGrader
from openjudge.graders.schema import Checkpoint, Rubric
from openjudge.harness import ClaudeCodeHarness  # or CodexHarness / CursorAgentHarness

# Step 1: define what to check as Rubric/Checkpoint (a checkpoint's `content` can be
# executable code the agent is instructed to actually run, or a natural-language criterion)
rubrics = [
    Rubric(
        name="correctness",
        weight=2.0,
        checkpoints=[
            Checkpoint(
                id="passes_tests",
                description="The submitted solution passes the provided test",
                content="assert fibonacci(10) == 55",
            ),
        ],
    ),
]

# Step 2: pick a harness (requires the corresponding CLI installed + authenticated
# on this machine -- e.g. `claude`, authenticated via ANTHROPIC_API_KEY)
harness = ClaudeCodeHarness(timeout_s=120)

# Step 3: create grader
grader = AgenticGrader(harness=harness, rubrics=rubrics)

result = await grader.aevaluate(
    query="Implement fibonacci(n) in fibonacci.py.",
    response="See fibonacci.py in the workspace.",
    workspace_path="/path/to/candidate/workspace",  # and/or transcript=[...]/"path/to/transcript.jsonl"
)
# result is a GraderScore on success, or a GraderError if no evidence was given
# or the harness sample was not trustworthy (e.g. CLI unavailable) --
# it is never silently coerced into a 0 score.
#
# v1 note: this runs a single sample and has no must_have AND-gate or k-sample
# majority voting yet; see the design doc's "Deferred to a follow-up iteration"
# discussion if you need those for a higher-stakes grading.
```

---

## Custom Grader — Extend BaseGrader

```python
from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderMode, GraderScore

class KeywordGrader(BaseGrader):
    def __init__(self, keywords: list[str], **kwargs):
        super().__init__(name="keyword_grader", mode=GraderMode.POINTWISE, **kwargs)
        self.keywords = keywords

    async def _aevaluate(self, response: str, **kwargs) -> GraderScore:
        hits = sum(1 for kw in self.keywords if kw.lower() in response.lower())
        score = hits / len(self.keywords)
        return GraderScore(
            name=self.name,
            score=score,
            reason=f"{hits}/{len(self.keywords)} keywords found",
        )

    @staticmethod
    def get_metadata():
        return {"description": "Checks keyword presence in response"}

grader = KeywordGrader(keywords=["Python", "list", "reverse"])
result = await grader.aevaluate(response="Use list.reverse() in Python.")
```

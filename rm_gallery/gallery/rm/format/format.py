import json
import re
from collections import Counter
from typing import Dict, List, Literal

from rm_gallery.core.grader import Grader, GraderMode, GraderScore
from rm_gallery.core.utils.tokenizer import get_tokenizer


class ReasoningFormatGrader(Grader):
    """
    Check format reward for thinking format and answer format with proper tags.

    This reward verifies if the generated content follows the required format
    with proper <think> and <answer> tags.
    """

    def __init__(
        self,
        name: str = "format_reward",
        grader_mode: GraderMode = GraderMode.POINTWISE,
        think_token: str = "think",
        answer_token: str = "answer",
        description: str = "",
    ):
        """
        Initialize the ReasoningFormatGrader.
        Args:
            name: The name of the grader.
            grader_mode: The evaluation mode.
            description: The description of the grader.

        """
        super().__init__(name, grader_mode, description)
        self.think_token = think_token
        self.answer_token = answer_token

    async def evaluate(self, answer: str, *args, **kwargs) -> GraderScore:
        """
        Check format and calculate reward.
        """

        # Check thinking format tags
        think_pattern = f"<{self.think_token}>.*?</{self.think_token}>"
        has_think_tag = bool(re.search(think_pattern, answer, re.DOTALL))

        # Check answer format tags
        answer_pattern = f"<{self.answer_token}>.*?</{self.answer_token}>"
        has_answer_tag = bool(re.search(answer_pattern, answer, re.DOTALL))

        # Calculate reward
        reward = 1.0 if has_think_tag and has_answer_tag else 0.0
        reasons = []

        if not has_think_tag:
            reasons.append(f"Missing <{self.think_token}></{self.think_token}> tags")

        if not has_answer_tag:
            reasons.append(f"Missing <{self.answer_token}></{self.answer_token}> tags")

        if reward == 1.0:
            reasons.append("All format requirements met")

        return GraderScore(
            score=reward,
            reason="; ".join(reasons),
            metadata={
                "has_think_tag": has_think_tag,
                "has_answer_tag": has_answer_tag,
                "total_reward": reward,
                "think_token": self.think_token,
                "answer_token": self.answer_token,
            },
        )


class ReasoningToolCallFormatGrader(Grader):
    """
    Check tool call format including think, answer and tool_call tags with JSON validation.

    This reward verifies if the generated content follows the required format
    with proper <think>, <answer> and <tool_call> tags, including JSON validation
    for tool calls.
    """

    def __init__(
        self,
        name: str = "tool_call_format",
        grader_mode: GraderMode = GraderMode.POINTWISE,
        description: str = "",
    ):
        """
        Initialize the ReasoningToolCallFormatGrader.
        Args:
            name: The name of the grader.
            grader_mode: The evaluation mode.
            description: The description of the grader.
        """
        super().__init__(name, grader_mode, description)

    async def evaluate(self, answer: str, **kwargs) -> GraderScore:
        """
        Check tool call format and calculate reward.

        """

        # Extract tag contents
        think_pattern = r"<think>(.*?)</think>"
        answer_pattern = r"<answer>(.*?)</answer>"
        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"

        think_matches = re.search(think_pattern, answer, re.DOTALL)
        answer_matches = re.search(answer_pattern, answer, re.DOTALL)
        tool_call_matches = re.findall(tool_call_pattern, answer, re.DOTALL)

        has_think_tag = think_matches is not None
        has_answer_tag = answer_matches is not None
        has_tool_call_tag = len(tool_call_matches) > 0

        valid_format = False
        valid_tool_call_json = False
        reasons = []

        if has_think_tag:
            # Case 1: <think></think> + <answer></answer>
            if has_answer_tag and not has_tool_call_tag:
                # Check overall format
                format_pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
                valid_format = bool(re.match(format_pattern, answer, re.DOTALL))

                # Check tag occurrence count
                if valid_format:
                    valid_format = (
                        answer.count("<think>") == 1
                        and answer.count("</think>") == 1
                        and answer.count("<answer>") == 1
                        and answer.count("</answer>") == 1
                    )

                if valid_format:
                    reasons.append("Valid <think></think> + <answer></answer> format")
                else:
                    reasons.append("Invalid <think></think> + <answer></answer> format")

            # Case 2: <think></think> + <tool_call></tool_call>
            elif has_tool_call_tag and not has_answer_tag:
                # Check overall format
                format_pattern = (
                    r"^\s*<think>.*?</think>\s*(?:<tool_call>.*?</tool_call>\s*)+$"
                )
                valid_format = bool(re.match(format_pattern, answer, re.DOTALL))

                # Check <think> tag occurrence count
                if valid_format:
                    valid_format = (
                        answer.count("<think>") == 1 and answer.count("</think>") == 1
                    )

                # Check if <tool_call> and </tool_call> tags appear in pairs
                if valid_format:
                    if answer.count("<tool_call>") != answer.count("</tool_call>"):
                        valid_format = False

                # Check for consecutive duplicate tags
                if valid_format:
                    if re.search(r"</tool_call>\s*</tool_call>", answer) or re.search(
                        r"<tool_call>\s*<tool_call>", answer
                    ):
                        valid_format = False

                # Check tool_call JSON format
                valid_tool_call_json = True
                tool_calls = []
                if valid_format:
                    for tool_call_content in tool_call_matches:
                        try:
                            tool_call_json = json.loads(tool_call_content.strip())
                            # Check if JSON contains required fields
                            if not (
                                "name" in tool_call_json
                                and "arguments" in tool_call_json
                            ):
                                valid_tool_call_json = False
                                break
                            tool_calls.append(
                                {
                                    "function": {
                                        "name": tool_call_json["name"],
                                        "arguments": json.dumps(
                                            tool_call_json["arguments"],
                                            ensure_ascii=False,
                                        ),
                                    }
                                }
                            )
                        except json.JSONDecodeError:
                            valid_tool_call_json = False
                            break

                valid_format = valid_format and valid_tool_call_json

                if valid_format:
                    reasons.append(
                        "Valid <think></think> + <tool_call></tool_call> format with valid JSON"
                    )
                else:
                    if not valid_tool_call_json:
                        reasons.append("Invalid JSON format in <tool_call> tags")
                    else:
                        reasons.append(
                            "Invalid <think></think> + <tool_call></tool_call> format"
                        )
            else:
                # Has both answer and tool_call, or neither
                reasons.append(
                    "Invalid combination: should have either <answer> or <tool_call> tags, not both or neither"
                )
        else:
            reasons.append("Missing <think></think> tags")

        # Calculate reward score
        reward = 1.0 if valid_format else 0.0
        return GraderScore(
            score=reward,
            reason="; ".join(reasons),
            extra_data={
                "has_think_tag": has_think_tag,
                "has_answer_tag": has_answer_tag,
                "has_tool_call_tag": has_tool_call_tag,
                "valid_format": valid_format,
                "valid_tool_call_json": valid_tool_call_json,
                "tool_call_count": len(tool_call_matches),
                "reward": reward,
            },
        )


class LengthPenaltyGrader(Grader):
    """
    Text length based penalty for content that is too short or too long.
    """

    def __init__(
        self,
        name: str = "",
        grader_mode: GraderMode = GraderMode.POINTWISE,
        min_length: int = 10,
        max_length: int = 1000,
        penalty_rate: float = 0.01,
        description: str = "",
    ):
        """
        Initialize the LengthPenaltyGrader.
        Args:
            name: Name of the grader
            grader_mode: Mode of the grader (POINTWISE or LISTWISE)
            min_length: Minimum length of the content
            max_length: Maximum length of the content
            penalty_rate: Penalty rate for each character beyond the maximum length
        """
        super().__init__(name, grader_mode, description)
        self.min_length = min_length
        self.max_length = max_length
        self.penalty_rate = penalty_rate

    async def evaluate(self, answer) -> GraderScore:
        """
        Check code syntax.
        """

        length = len(answer)

        penalty = 0.0
        reason_parts = []

        if length < self.min_length:
            penalty = -(self.min_length - length) * self.penalty_rate
            reason_parts.append(f"Too short: {length} < {self.min_length}")
        elif length > self.max_length:
            penalty = -(length - self.max_length) * self.penalty_rate
            reason_parts.append(f"Too long: {length} > {self.max_length}")
        else:
            reason_parts.append(
                f"Length acceptable: {self.min_length} <= {length} <= {self.max_length}"
            )

        return GraderScore(
            score=penalty,
            reason="; ".join(reason_parts),
            metadata={
                "length": length,
                "min_length": self.min_length,
                "max_length": self.max_length,
                "penalty": penalty,
            },
        )


class NgramRepetitionPenaltyGrader(Grader):
    """
    Calculate N-gram repetition penalty supporting Chinese processing and multiple penalty strategies.
    """

    def __init__(
        self,
        name: str = "",
        grader_mode: GraderMode = GraderMode.POINTWISE,
        n: int = 3,
        penalty_threshold: float = 0.3,
        penalty_rate: float = 1.0,
        use_soft_penalty: bool = False,
        max_penalty: float = -1.0,
        min_scaling: float = 0.0,
        tokenizer_type: Literal["tiktoken", "jieba", "simple"] = "tiktoken",
        encoding_name: str = "cl100k_base",
        chinese_only: bool = False,
        analyze_scope: Literal["thought", "full"] = "full",
        description: str = "",
    ):
        """
        Initialize the NgramRepetitionPenaltyGrader.
        Args:

            name: Name of the grader
            grader_mode: Mode of the grader (POINTWISE or LISTWISE)
            n: N value for N-gram
            penalty_threshold: Threshold for hard threshold penalty
            penalty_rate: Penalty rate for each repetition
            use_soft_penalty: Use soft threshold penalty
            max_penalty: Maximum penalty value
            min_scaling: Minimum scaling factor for soft threshold penalty
            tokenizer_type: Tokenizer type (tiktoken, jieba, simple)
            encoding_name: Encoding name for tiktoken
            chinese_only: Whether to keep only Chinese characters (for jieba tokenizer)
            analyze_scope: Analyze scope (thought or full)
            description: Description of the grader
        """
        super().__init__(name, grader_mode, description=description)
        self.n = n
        self.penalty_threshold = penalty_threshold
        self.penalty_rate = penalty_rate
        self.use_soft_penalty = use_soft_penalty
        self.analyze_scope = analyze_scope
        self.chinese_only = chinese_only
        self.encoding_name = encoding_name
        self.tokenizer_type = tokenizer_type
        self.chinese_only = chinese_only
        self.max_penalty = max_penalty
        self.min_scaling = min_scaling
        self.tokenizer = get_tokenizer(
            tokenizer_type=tokenizer_type,
            encoding_name=encoding_name,
            chinese_only=chinese_only,
        )

    def _extract_thought_process(self, content: str) -> str:
        """Extract thought process"""
        think_pattern = r"<think>(.*?)</think>"
        matches = re.findall(think_pattern, content, re.DOTALL)
        return " ".join(matches) if matches else ""

    def _generate_ngrams(self, tokens: List[str]) -> List[tuple]:
        """Generate N-grams"""
        if len(tokens) < self.n:
            return []

        # Use unified approach for all tokenizers
        ngrams = []
        for i in range(len(tokens) - self.n + 1):
            ngrams.append(tuple(tokens[i : i + self.n]))
        return ngrams

    def _calculate_penalty(self, repetition_rate: float) -> float:
        """Calculate penalty value"""
        if self.use_soft_penalty:
            # Soft penalty mode
            if self.max_penalty > 0:
                raise ValueError(
                    f"max_penalty {self.max_penalty} should not be positive"
                )

            scaling = repetition_rate
            if scaling < self.min_scaling:
                scaling = 0.0
            elif scaling > self.min_scaling:
                scaling = (scaling - self.min_scaling) / (1 - self.min_scaling)

            return scaling * self.max_penalty
        else:
            # Hard threshold mode (original logic)
            if repetition_rate > self.penalty_threshold:
                return -(repetition_rate - self.penalty_threshold) * self.penalty_rate
            return 0.0

    async def evaluate(self, answer: str, **kwargs) -> GraderScore:
        """
        Calculate N-gram repetition penalty
        """

        # Select text based on analysis scope
        if self.analyze_scope == "thought":
            text_to_analyze = self._extract_thought_process(answer)
            if not text_to_analyze:
                return GraderScore(
                    score=0.0,
                    reason="No thought process found to analyze",
                    metadata={
                        "analyze_scope": self.analyze_scope,
                        "text_to_analyze": text_to_analyze,
                    },
                )

        else:
            text_to_analyze = answer

        # Tokenization using unified tokenizer
        preprocessed_text = self.tokenizer.preprocess_text(
            text_to_analyze,
            to_lower=(
                self.tokenizer_type != "jieba"
            ),  # Keep case for Chinese tokenization
        )
        tokens = self.tokenizer.tokenize(preprocessed_text)

        if len(tokens) < self.n:
            return GraderScore(
                score=0.0,
                reason=f"Text too short for {self.n}-gram analysis",
                metadata={
                    "token_count": len(tokens),
                    "n": self.n,
                    "analyze_scope": self.analyze_scope,
                    "tokenizer_type": self.tokenizer_type,
                },
            )

        # Generate N-grams
        ngrams = self._generate_ngrams(tokens)

        if not ngrams:
            return GraderScore(
                score=0.0,
                reason="No ngrams generated",
                metadata={
                    "token_count": len(tokens),
                    "n": self.n,
                    "analyze_scope": self.analyze_scope,
                    "tokenizer_type": self.tokenizer_type,
                },
            )

        # Calculate repetition rate
        ngram_counts = Counter(ngrams)
        total_ngrams = len(ngrams)
        unique_ngrams = len(ngram_counts)
        repetition_rate = (
            1 - (unique_ngrams / total_ngrams) if total_ngrams > 0 else 0.0
        )

        # Calculate penalty
        penalty = self._calculate_penalty(repetition_rate)

        # Build reason description
        penalty_mode = "soft" if self.use_soft_penalty else "hard"
        return GraderScore(
            score=penalty,
            reason=f"{self.n}-gram repetition rate: {repetition_rate:.3f}, penalty: {penalty:.3f} ({penalty_mode} penalty, {self.tokenizer_type} tokenizer, scope: {self.analyze_scope})",
            metadata={
                "repetition_rate": repetition_rate,
                "unique_ngrams": unique_ngrams,
                "total_ngrams": total_ngrams,
                "penalty": penalty,
                "most_common_ngrams": ngram_counts.most_common(5),
                "analyze_scope": self.analyze_scope,
                "tokenizer_type": self.tokenizer_type,
                "use_soft_penalty": self.use_soft_penalty,
                "penalty_mode": penalty_mode,
            },
        )


class PrivacyLeakageGrader(Grader):
    """
    Privacy information leakage detection for emails, phone numbers, ID cards, credit cards, and IP addresses.

    This reward checks for potential privacy leaks in the generated content,
    including email addresses, phone numbers, ID numbers, credit card numbers,
    and IP addresses. Applies penalties for each detected leak.
    """

    def __init__(
        self,
        name: str = "privacy_leakage",
        penalty_per_leak: float = -0.5,
        grader_mode: GraderMode = GraderMode.POINTWISE,
        description: str = "Privacy leakage detection reward",
    ):
        """
        Initialize the PrivacyLeakageGrader.
        Parameters:
        name: Name of the grader.
        penalty_per_leak: Penalty per leak.
        grader_mode: Grader mode.
        description: Description of the grader.
        """
        super().__init__(name=name, grader_mode=grader_mode, description=description)
        self.penalty_per_leak = penalty_per_leak

    def _detect_privacy_leaks(self, text: str) -> List[Dict[str, str]]:
        """Detect privacy information leaks"""
        leaks = []

        # Email addresses
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails = re.findall(email_pattern, text)
        for email in emails:
            leaks.append({"type": "email", "value": email})

        # Phone numbers (simple pattern)
        phone_pattern = (
            r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
        )
        phones = re.findall(phone_pattern, text)
        for phone in phones:
            leaks.append({"type": "phone", "value": phone})

        # ID numbers (China)
        id_pattern = r"\b[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]\b"
        ids = re.findall(id_pattern, text)
        for id_num in ids:
            leaks.append({"type": "id_card", "value": id_num})

        # Credit card numbers (simple detection)
        credit_card_pattern = r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
        cards = re.findall(credit_card_pattern, text)
        for card in cards:
            leaks.append({"type": "credit_card", "value": card})

        # IP addresses
        ip_pattern = r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
        ips = re.findall(ip_pattern, text)
        for ip in ips:
            # Exclude common non-sensitive IPs (like localhost)
            if not ip.startswith(("127.", "192.168.", "10.", "172.")):
                leaks.append({"type": "ip_address", "value": ip})

        return leaks

    async def evaluate(self, answer: str) -> GraderScore:
        """
        Detect privacy leaks.

        Args:
            sample: Data sample containing text content

        Returns:
            RewardResult: Reward result containing privacy leak penalty score
        """
        leaks = self._detect_privacy_leaks(answer)
        penalty = len(leaks) * self.penalty_per_leak

        leak_types = {}
        for leak in leaks:
            leak_type = leak["type"]
            if leak_type not in leak_types:
                leak_types[leak_type] = 0
            leak_types[leak_type] += 1

        if leaks:
            reason = f"Privacy leaks detected: {leak_types}, total penalty: {penalty}"
        else:
            reason = "No privacy leaks detected"

        return GraderScore(
            score=penalty,
            reason=reason,
            metadata={
                "leaks": leaks,
                "leak_types": leak_types,
                "total_leaks": len(leaks),
                "penalty": penalty,
            },
        )

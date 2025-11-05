"""
JSON Matching and Validation Metrics

JSON format validation and deep comparison metrics based on OpenAI Evals implementation.
"""

import json
from typing import Any, Dict, Optional

from pydantic import Field

from rm_gallery.core.metrics.base import BaseStringMetric
from rm_gallery.core.metrics.registry import register_metric
from rm_gallery.core.metrics.schema import ComparisonInput, MetricResult


@register_metric("json_match")
class JsonMatchMetric(BaseStringMetric):
    """
    JSON Deep Match Metric

    Recursively compares JSON structures element by element.
    Based on OpenAI Evals' JsonMatch implementation.

    For dicts: all keys must match and all values must match recursively
    For lists: must have same length and all elements must match recursively in order
    For primitives: must be exactly equal

    Attributes:
        name: Metric name
        strict_order: Whether to strictly compare list order (default True)
        ignore_extra_keys: Whether to ignore extra keys in candidate dict (default False)

    Example:
        >>> metric = JsonMatchMetric()
        >>> input_data = ComparisonInput(
        ...     reference='{"name": "Alice", "age": 30}',
        ...     candidate='{"name": "Alice", "age": 30}'
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"Matched: {result.score}")
        Matched: 1.0

    References:
        OpenAI Evals: evals/elsuite/basic/json_match.py
    """

    name: str = "json_match"
    strict_order: bool = Field(
        default=True, description="Whether to strictly compare list order"
    )
    ignore_extra_keys: bool = Field(
        default=False, description="Whether to ignore extra keys in candidate"
    )
    normalize_text: bool = Field(
        default=False, description="JSON matching typically doesn't normalize"
    )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute JSON match

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result, score is 1.0 (matched) or 0.0 (not matched)
        """
        reference = input_data.reference
        candidate = input_data.candidate

        # Parse candidate JSON
        try:
            candidate_json = json.loads(candidate)
        except (json.JSONDecodeError, TypeError) as e:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={
                    "matched": False,
                    "error": "candidate_parse_error",
                    "error_message": str(e),
                },
            )

        # Handle multiple reference texts
        if isinstance(reference, list):
            matches = []
            parse_errors = []

            for i, ref in enumerate(reference):
                try:
                    ref_json = json.loads(ref)
                    matched = self._json_match(candidate_json, ref_json)
                    matches.append(matched)
                except (json.JSONDecodeError, TypeError) as e:
                    matches.append(False)
                    parse_errors.append(f"Reference {i}: {str(e)}")

            matched = any(matches)
            details = {
                "matched": matched,
                "num_references": len(reference),
                "matches_per_reference": matches,
                "matched_reference_indices": [i for i, m in enumerate(matches) if m],
            }

            if parse_errors:
                details["parse_errors"] = parse_errors
        else:
            # Single reference
            try:
                reference_json = json.loads(reference)
                matched = self._json_match(candidate_json, reference_json)
                details = {"matched": matched}
            except (json.JSONDecodeError, TypeError) as e:
                return MetricResult(
                    name=self.name,
                    score=0.0,
                    details={
                        "matched": False,
                        "error": "reference_parse_error",
                        "error_message": str(e),
                    },
                )

        return MetricResult(
            name=self.name,
            score=1.0 if matched else 0.0,
            details=details,
            metadata={
                "strict_order": self.strict_order,
                "ignore_extra_keys": self.ignore_extra_keys,
            },
        )

    def _json_match(self, sampled: Any, correct: Any) -> bool:
        """
        Recursively compare JSON structures

        Based on OpenAI Evals implementation.

        Args:
            sampled: Candidate JSON object
            correct: Reference JSON object

        Returns:
            bool: Whether the structures match
        """
        # Handle None values
        if sampled is None or correct is None:
            return sampled == correct

        # Handle dict
        if isinstance(sampled, dict):
            if not isinstance(correct, dict):
                return False

            if self.ignore_extra_keys:
                # Only check keys that exist in correct
                return all(
                    self._json_match(sampled.get(key), correct.get(key))
                    for key in correct.keys()
                )
            else:
                # All keys must match (from both dicts)
                all_keys = set(sampled.keys()) | set(correct.keys())
                return all(
                    self._json_match(sampled.get(key), correct.get(key))
                    for key in all_keys
                )

        # Handle list
        elif isinstance(sampled, list):
            if not isinstance(correct, list):
                return False

            # Lists must have the same length
            if len(sampled) != len(correct):
                return False

            if self.strict_order:
                # Compare elements in order
                return all(self._json_match(s, c) for s, c in zip(sampled, correct))
            else:
                # Allow different order (try to find matching permutation)
                # This is more expensive but useful for unordered lists
                if len(sampled) == 0:
                    return True

                # Try to match each sampled element to a correct element
                used = [False] * len(correct)
                for s_item in sampled:
                    found_match = False
                    for i, c_item in enumerate(correct):
                        if not used[i] and self._json_match(s_item, c_item):
                            used[i] = True
                            found_match = True
                            break
                    if not found_match:
                        return False
                return True

        # Primitive types: direct comparison
        return sampled == correct


@register_metric("json_validator")
class JsonValidatorMetric(BaseStringMetric):
    """
    JSON Format Validator

    Validates if the candidate text is valid JSON.
    Based on OpenAI Evals' JsonValidator implementation.

    Attributes:
        name: Metric name

    Example:
        >>> metric = JsonValidatorMetric()
        >>> input_data = ComparisonInput(
        ...     reference="",  # reference not needed for validation
        ...     candidate='{"valid": "json"}'
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"Valid: {result.score}")
        Valid: 1.0

    References:
        OpenAI Evals: evals/elsuite/basic/json_validator.py
    """

    name: str = "json_validator"
    normalize_text: bool = Field(
        default=False, description="JSON validation doesn't normalize"
    )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Validate JSON format

        Args:
            input_data: Comparison input data (only candidate is used)

        Returns:
            MetricResult: Evaluation result, score is 1.0 (valid) or 0.0 (invalid)
        """
        candidate = input_data.candidate

        is_valid, error_msg = self._is_valid_json(candidate)

        details = {
            "is_valid": is_valid,
            "candidate_length": len(candidate),
        }

        if not is_valid:
            details["error_message"] = error_msg

        return MetricResult(
            name=self.name,
            score=1.0 if is_valid else 0.0,
            details=details,
        )

    def _is_valid_json(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Check if text is valid JSON

        Args:
            text: Text to validate

        Returns:
            tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        try:
            json.loads(text)
            return True, None
        except json.JSONDecodeError as e:
            return False, f"JSON decode error: {str(e)}"
        except TypeError as e:
            return False, f"Type error: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"


@register_metric("json_schema_validator")
class JsonSchemaValidatorMetric(BaseStringMetric):
    """
    JSON Schema Validator

    Validates JSON against a JSON Schema.
    Extends JsonValidator with schema validation support.

    Attributes:
        name: Metric name
        schema: JSON Schema dict for validation

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"},
        ...         "age": {"type": "number"}
        ...     },
        ...     "required": ["name", "age"]
        ... }
        >>> metric = JsonSchemaValidatorMetric(schema=schema)
        >>> input_data = ComparisonInput(
        ...     reference="",
        ...     candidate='{"name": "Alice", "age": 30}'
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"Valid: {result.score}")
        Valid: 1.0

    Note:
        Requires jsonschema package: pip install jsonschema
    """

    name: str = "json_schema_validator"
    schema: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON Schema for validation"
    )
    normalize_text: bool = Field(
        default=False, description="Schema validation doesn't normalize"
    )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Validate JSON against schema

        Args:
            input_data: Comparison input data (candidate must be valid JSON)

        Returns:
            MetricResult: Evaluation result
        """
        candidate = input_data.candidate

        # First check if it's valid JSON
        try:
            candidate_json = json.loads(candidate)
        except (json.JSONDecodeError, TypeError) as e:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={
                    "is_valid": False,
                    "error_type": "json_parse_error",
                    "error_message": str(e),
                },
            )

        # If no schema provided, just validate JSON format
        if self.schema is None:
            return MetricResult(
                name=self.name,
                score=1.0,
                details={
                    "is_valid": True,
                    "message": "JSON is valid (no schema provided)",
                },
            )

        # Validate against schema
        try:
            import jsonschema

            jsonschema.validate(instance=candidate_json, schema=self.schema)
            return MetricResult(
                name=self.name,
                score=1.0,
                details={
                    "is_valid": True,
                    "schema_valid": True,
                },
            )
        except ImportError:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={
                    "is_valid": False,
                    "error_type": "import_error",
                    "error_message": "jsonschema package not installed. Install with: pip install jsonschema",
                },
            )
        except jsonschema.exceptions.ValidationError as e:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={
                    "is_valid": False,
                    "error_type": "schema_validation_error",
                    "error_message": str(e.message),
                    "failed_path": list(e.path),
                    "validator": e.validator,
                },
            )
        except jsonschema.exceptions.SchemaError as e:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={
                    "is_valid": False,
                    "error_type": "schema_error",
                    "error_message": f"Invalid schema: {str(e)}",
                },
            )


__all__ = [
    "JsonMatchMetric",
    "JsonValidatorMetric",
    "JsonSchemaValidatorMetric",
]

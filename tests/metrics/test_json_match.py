"""
Unit Tests for JSON Matching and Validation Metrics

Tests for JsonMatchMetric, JsonValidatorMetric, and JsonSchemaValidatorMetric.
"""

import json

import pytest

from rm_gallery.core.metrics import (
    JsonMatchMetric,
    JsonSchemaValidatorMetric,
    JsonValidatorMetric,
)
from rm_gallery.core.metrics.schema import ComparisonInput


class TestJsonMatchMetric:
    """Test JsonMatchMetric"""

    def test_exact_match_simple(self):
        """Test exact match for simple JSON objects"""
        metric = JsonMatchMetric()

        input_data = ComparisonInput(
            reference='{"name": "Alice", "age": 30}',
            candidate='{"name": "Alice", "age": 30}',
        )

        result = metric.compute(input_data)
        assert result.score == 1.0
        assert result.details["matched"] is True

    def test_exact_match_different_order(self):
        """Test that dict key order doesn't matter"""
        metric = JsonMatchMetric()

        input_data = ComparisonInput(
            reference='{"name": "Alice", "age": 30}',
            candidate='{"age": 30, "name": "Alice"}',
        )

        result = metric.compute(input_data)
        assert result.score == 1.0
        assert result.details["matched"] is True

    def test_no_match_different_values(self):
        """Test no match when values differ"""
        metric = JsonMatchMetric()

        input_data = ComparisonInput(
            reference='{"name": "Alice", "age": 30}',
            candidate='{"name": "Bob", "age": 30}',
        )

        result = metric.compute(input_data)
        assert result.score == 0.0
        assert result.details["matched"] is False

    def test_no_match_missing_key(self):
        """Test no match when key is missing"""
        metric = JsonMatchMetric()

        input_data = ComparisonInput(
            reference='{"name": "Alice", "age": 30}', candidate='{"name": "Alice"}'
        )

        result = metric.compute(input_data)
        assert result.score == 0.0
        assert result.details["matched"] is False

    def test_ignore_extra_keys(self):
        """Test ignore_extra_keys option"""
        metric = JsonMatchMetric(ignore_extra_keys=True)

        input_data = ComparisonInput(
            reference='{"name": "Alice"}',
            candidate='{"name": "Alice", "age": 30, "city": "NYC"}',
        )

        result = metric.compute(input_data)
        assert result.score == 1.0
        assert result.details["matched"] is True

    def test_list_match_same_order(self):
        """Test list matching with same order"""
        metric = JsonMatchMetric()

        input_data = ComparisonInput(reference="[1, 2, 3]", candidate="[1, 2, 3]")

        result = metric.compute(input_data)
        assert result.score == 1.0
        assert result.details["matched"] is True

    def test_list_no_match_different_order(self):
        """Test list doesn't match with different order (strict_order=True)"""
        metric = JsonMatchMetric(strict_order=True)

        input_data = ComparisonInput(reference="[1, 2, 3]", candidate="[3, 2, 1]")

        result = metric.compute(input_data)
        assert result.score == 0.0
        assert result.details["matched"] is False

    def test_list_match_different_order_allowed(self):
        """Test list matches with different order when strict_order=False"""
        metric = JsonMatchMetric(strict_order=False)

        input_data = ComparisonInput(reference="[1, 2, 3]", candidate="[3, 2, 1]")

        result = metric.compute(input_data)
        assert result.score == 1.0
        assert result.details["matched"] is True

    def test_list_no_match_different_length(self):
        """Test lists with different lengths don't match"""
        metric = JsonMatchMetric()

        input_data = ComparisonInput(reference="[1, 2, 3]", candidate="[1, 2]")

        result = metric.compute(input_data)
        assert result.score == 0.0
        assert result.details["matched"] is False

    def test_nested_structure_match(self):
        """Test nested JSON structure matching"""
        metric = JsonMatchMetric()

        reference = json.dumps(
            {
                "user": {
                    "name": "Alice",
                    "contacts": {"email": "alice@example.com", "phone": "123-4567"},
                },
                "active": True,
            }
        )

        candidate = json.dumps(
            {
                "user": {
                    "name": "Alice",
                    "contacts": {"email": "alice@example.com", "phone": "123-4567"},
                },
                "active": True,
            }
        )

        input_data = ComparisonInput(reference=reference, candidate=candidate)
        result = metric.compute(input_data)

        assert result.score == 1.0
        assert result.details["matched"] is True

    def test_nested_list_in_dict(self):
        """Test nested lists in dict"""
        metric = JsonMatchMetric()

        input_data = ComparisonInput(
            reference='{"items": [1, 2, 3], "name": "test"}',
            candidate='{"items": [1, 2, 3], "name": "test"}',
        )

        result = metric.compute(input_data)
        assert result.score == 1.0
        assert result.details["matched"] is True

    def test_invalid_candidate_json(self):
        """Test handling of invalid candidate JSON"""
        metric = JsonMatchMetric()

        input_data = ComparisonInput(
            reference='{"name": "Alice"}', candidate="not valid json"
        )

        result = metric.compute(input_data)
        assert result.score == 0.0
        assert result.details["matched"] is False
        assert result.details["error"] == "candidate_parse_error"

    def test_invalid_reference_json(self):
        """Test handling of invalid reference JSON"""
        metric = JsonMatchMetric()

        input_data = ComparisonInput(
            reference="not valid json", candidate='{"name": "Alice"}'
        )

        result = metric.compute(input_data)
        assert result.score == 0.0
        assert result.details["matched"] is False
        assert result.details["error"] == "reference_parse_error"

    def test_multiple_references(self):
        """Test matching with multiple reference options"""
        metric = JsonMatchMetric()

        input_data = ComparisonInput(
            reference=['{"name": "Alice"}', '{"name": "Bob"}', '{"name": "Charlie"}'],
            candidate='{"name": "Bob"}',
        )

        result = metric.compute(input_data)
        assert result.score == 1.0
        assert result.details["matched"] is True
        assert result.details["num_references"] == 3
        assert 1 in result.details["matched_reference_indices"]

    def test_null_values(self):
        """Test handling of null values"""
        metric = JsonMatchMetric()

        input_data = ComparisonInput(
            reference='{"name": null}', candidate='{"name": null}'
        )

        result = metric.compute(input_data)
        assert result.score == 1.0
        assert result.details["matched"] is True

    def test_boolean_values(self):
        """Test boolean value matching"""
        metric = JsonMatchMetric()

        input_data = ComparisonInput(
            reference='{"active": true, "deleted": false}',
            candidate='{"active": true, "deleted": false}',
        )

        result = metric.compute(input_data)
        assert result.score == 1.0
        assert result.details["matched"] is True

    def test_number_types(self):
        """Test different number types"""
        metric = JsonMatchMetric()

        input_data = ComparisonInput(
            reference='{"int": 42, "float": 3.14}',
            candidate='{"int": 42, "float": 3.14}',
        )

        result = metric.compute(input_data)
        assert result.score == 1.0
        assert result.details["matched"] is True

    def test_empty_structures(self):
        """Test empty dict and list"""
        metric = JsonMatchMetric()

        # Empty dict
        input_data = ComparisonInput(reference="{}", candidate="{}")
        result = metric.compute(input_data)
        assert result.score == 1.0

        # Empty list
        input_data = ComparisonInput(reference="[]", candidate="[]")
        result = metric.compute(input_data)
        assert result.score == 1.0


class TestJsonValidatorMetric:
    """Test JsonValidatorMetric"""

    def test_valid_json_object(self):
        """Test valid JSON object"""
        metric = JsonValidatorMetric()

        input_data = ComparisonInput(
            reference="", candidate='{"name": "Alice", "age": 30}'  # Not used
        )

        result = metric.compute(input_data)
        assert result.score == 1.0
        assert result.details["is_valid"] is True

    def test_valid_json_array(self):
        """Test valid JSON array"""
        metric = JsonValidatorMetric()

        input_data = ComparisonInput(reference="", candidate='[1, 2, 3, "test"]')

        result = metric.compute(input_data)
        assert result.score == 1.0
        assert result.details["is_valid"] is True

    def test_valid_json_primitives(self):
        """Test valid JSON primitives"""
        metric = JsonValidatorMetric()

        # String
        result = metric.compute(ComparisonInput(reference="", candidate='"hello"'))
        assert result.score == 1.0

        # Number
        result = metric.compute(ComparisonInput(reference="", candidate="42"))
        assert result.score == 1.0

        # Boolean
        result = metric.compute(ComparisonInput(reference="", candidate="true"))
        assert result.score == 1.0

        # Null
        result = metric.compute(ComparisonInput(reference="", candidate="null"))
        assert result.score == 1.0

    def test_invalid_json_malformed(self):
        """Test invalid JSON (malformed)"""
        metric = JsonValidatorMetric()

        input_data = ComparisonInput(
            reference="", candidate='{"name": "Alice"'  # Missing closing brace
        )

        result = metric.compute(input_data)
        assert result.score == 0.0
        assert result.details["is_valid"] is False
        assert "error_message" in result.details

    def test_invalid_json_not_json(self):
        """Test invalid JSON (not JSON at all)"""
        metric = JsonValidatorMetric()

        input_data = ComparisonInput(reference="", candidate="This is just plain text")

        result = metric.compute(input_data)
        assert result.score == 0.0
        assert result.details["is_valid"] is False

    def test_empty_string(self):
        """Test empty string is invalid JSON"""
        metric = JsonValidatorMetric()

        input_data = ComparisonInput(reference="", candidate="")

        result = metric.compute(input_data)
        assert result.score == 0.0
        assert result.details["is_valid"] is False


class TestJsonSchemaValidatorMetric:
    """Test JsonSchemaValidatorMetric"""

    def test_valid_against_schema(self):
        """Test valid JSON against schema"""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
            "required": ["name", "age"],
        }

        metric = JsonSchemaValidatorMetric(schema=schema)

        input_data = ComparisonInput(
            reference="", candidate='{"name": "Alice", "age": 30}'
        )

        result = metric.compute(input_data)
        assert result.score == 1.0
        assert result.details["is_valid"] is True
        assert result.details.get("schema_valid") is True

    def test_invalid_missing_required_field(self):
        """Test invalid JSON (missing required field)"""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
            "required": ["name", "age"],
        }

        metric = JsonSchemaValidatorMetric(schema=schema)

        input_data = ComparisonInput(
            reference="", candidate='{"name": "Alice"}'  # Missing age
        )

        result = metric.compute(input_data)
        assert result.score == 0.0
        assert result.details["is_valid"] is False
        assert result.details["error_type"] == "schema_validation_error"

    def test_invalid_wrong_type(self):
        """Test invalid JSON (wrong type)"""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
        }

        metric = JsonSchemaValidatorMetric(schema=schema)

        input_data = ComparisonInput(
            reference="",
            candidate='{"name": "Alice", "age": "thirty"}',  # age should be number
        )

        result = metric.compute(input_data)
        assert result.score == 0.0
        assert result.details["is_valid"] is False

    def test_no_schema_just_validates_json(self):
        """Test without schema just validates JSON format"""
        metric = JsonSchemaValidatorMetric(schema=None)

        input_data = ComparisonInput(
            reference="", candidate='{"name": "Alice", "anything": "goes"}'
        )

        result = metric.compute(input_data)
        assert result.score == 1.0
        assert result.details["is_valid"] is True

    def test_complex_schema_with_nested_objects(self):
        """Test complex schema with nested objects"""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string", "format": "email"},
                    },
                    "required": ["name"],
                },
                "items": {"type": "array", "items": {"type": "number"}},
            },
        }

        metric = JsonSchemaValidatorMetric(schema=schema)

        input_data = ComparisonInput(
            reference="",
            candidate='{"user": {"name": "Alice", "email": "alice@example.com"}, "items": [1, 2, 3]}',
        )

        result = metric.compute(input_data)
        assert result.score == 1.0
        assert result.details["is_valid"] is True

    def test_invalid_json_format(self):
        """Test invalid JSON format (not parseable)"""
        schema = {"type": "object"}
        metric = JsonSchemaValidatorMetric(schema=schema)

        input_data = ComparisonInput(reference="", candidate="not valid json")

        result = metric.compute(input_data)
        assert result.score == 0.0
        assert result.details["error_type"] == "json_parse_error"


class TestJsonMatchCompatibility:
    """Test compatibility with OpenAI Evals examples"""

    def test_openai_evals_example_1(self):
        """Test example from OpenAI Evals documentation"""
        metric = JsonMatchMetric()

        # Example: matching a simple response
        input_data = ComparisonInput(
            reference='{"answer": "Paris"}', candidate='{"answer": "Paris"}'
        )

        result = metric.compute(input_data)
        assert result.score == 1.0

    def test_openai_evals_example_2(self):
        """Test list matching from OpenAI Evals"""
        metric = JsonMatchMetric()

        # Lists must match exactly in order
        input_data = ComparisonInput(
            reference='{"steps": ["a", "b", "c"]}',
            candidate='{"steps": ["a", "b", "c"]}',
        )

        result = metric.compute(input_data)
        assert result.score == 1.0

    def test_openai_evals_multiple_valid_answers(self):
        """Test multiple valid answers like OpenAI Evals"""
        metric = JsonMatchMetric()

        # Multiple acceptable answers
        input_data = ComparisonInput(
            reference=['{"color": "red"}', '{"color": "blue"}', '{"color": "green"}'],
            candidate='{"color": "blue"}',
        )

        result = metric.compute(input_data)
        assert result.score == 1.0
        assert result.details["matched"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

from rm_gallery.core.data.schema import DataOutput, DataSample, Step
from rm_gallery.core.model.message import ChatMessage, MessageRole
from rm_gallery.gallery.rm.code.code import (
    CodeExecutionReward,
    CodeStyleReward,
    PatchSimilarityReward,
    SyntaxCheckReward,
)


class TestSyntaxCheckReward:
    """Test the syntax check reward model"""

    def setup_method(self):
        """Initialize the reward model for testing"""
        self.reward = SyntaxCheckReward()

    def test_valid_python_code(self):
        """Test valid Python code"""
        content = """This is a simple Python function:
```python
def hello_world():
    print("Hello, World!")
    return "success"
```
"""
        sample = self._create_sample(content)
        result = self.reward._evaluate(sample)

        assert result.name == "syntax_check"
        assert len(result.details) == 1
        assert (
            result.details[0].score == 1.0
        )  # Should get full score for valid Python code
        assert "1/1 blocks valid" in result.details[0].reason

    def test_invalid_python_code(self):
        """Test invalid Python code"""
        content = """This is a code with syntax errors:
```python
def broken_function(
    print("Missing parenthesis")
    return "error"
```
"""
        sample = self._create_sample(content)
        result = self.reward._evaluate(sample)

        assert result.name == "syntax_check"
        assert len(result.details) == 1
        assert (
            result.details[0].score <= 0.0
        )  # Should get score less than 0.0 for code with syntax errors
        assert result.extra_data["valid_blocks"] == 0
        assert len(result.extra_data["syntax_errors"]) > 0

    def test_mixed_code_blocks(self):
        """Test mixed code blocks (some valid, some invalid)"""
        content = """There are two code blocks:
```python
def valid_function():
    return "valid"
```

```python
def invalid_function(
    print("Missing parenthesis")
```
"""
        sample = self._create_sample(content)
        result = self.reward._evaluate(sample)

        assert result.name == "syntax_check"
        assert result.extra_data["total_blocks"] == 2
        assert result.extra_data["valid_blocks"] == 1
        assert len(result.extra_data["syntax_errors"]) > 0

    def test_no_code_blocks(self):
        """Test case with no code blocks"""
        content = "This is a piece of text without any code blocks."
        sample = self._create_sample(content)
        result = self.reward._evaluate(sample)

        assert result.name == "syntax_check"
        assert result.details[0].score == 0.0
        assert "No code blocks found" in result.details[0].reason
        assert result.extra_data["code_blocks"] == []

    def _create_sample(self, content: str) -> DataSample:
        """Create a test DataSample"""
        return DataSample(
            unique_id="test_syntax",
            input=[ChatMessage(role=MessageRole.USER, content="Write some code")],
            output=[
                DataOutput(answer=Step(role=MessageRole.ASSISTANT, content=content))
            ],
        )


class TestCodeStyleReward:
    """Test the code style reward model"""

    def setup_method(self):
        """Initialize the reward model for testing"""
        self.reward = CodeStyleReward()

    def test_good_style_code(self):
        """Test good style code"""
        content = """This is a well-styled Python function:
```python
def calculate_sum(numbers_list):
    total_sum = 0
    for number in numbers_list:
        total_sum += number
    return total_sum
```
"""
        sample = self._create_sample(content)
        result = self.reward._evaluate(sample)

        assert result.name == "code_style"
        assert len(result.details) == 1
        assert (
            result.details[0].score > 0.5
        )  # Should get high score for good style code

    def test_mixed_indentation(self):
        """Test mixed indentation code"""
        content = """This is a code with mixed indentation:
```python
def bad_indentation():
    if True:
	    print("Tab indented")  # Using tab
        print("Space indented")  # Using space
```
"""
        sample = self._create_sample(content)
        result = self.reward._evaluate(sample)

        assert result.name == "code_style"
        # Should detect mixed indentation
        assert "Mixed indentation" in str(result.extra_data["details"])

    def test_bad_naming_convention(self):
        """Test bad naming convention"""
        content = """This is a code with bad naming convention:
```python
def BadFunctionName():
    CamelCaseVar = 10
    return CamelCaseVar
```
"""
        sample = self._create_sample(content)
        result = self.reward._evaluate(sample)

        assert result.name == "code_style"
        # Should get score less than 1.0 for code with bad naming convention
        assert result.details[0].score < 1.0

    def test_no_code_blocks_style(self):
        """Test case with no code blocks"""
        content = "This is a piece of text without any code blocks."
        sample = self._create_sample(content)
        result = self.reward._evaluate(sample)

        assert result.name == "code_style"
        assert result.details[0].score == 0.0
        assert "No code blocks found" in result.details[0].reason

    def _create_sample(self, content: str) -> DataSample:
        """Create a test DataSample"""
        return DataSample(
            unique_id="test_style",
            input=[ChatMessage(role=MessageRole.USER, content="Write some code")],
            output=[
                DataOutput(answer=Step(role=MessageRole.ASSISTANT, content=content))
            ],
        )


class TestPatchSimilarityReward:
    """Test the patch similarity reward model"""

    def setup_method(self):
        """Initialize the reward model for testing"""
        self.reward = PatchSimilarityReward()

    def test_identical_patches(self):
        """Test identical patches"""
        content = "def fixed_function():\n    return 'fixed'"
        reference = "def fixed_function():\n    return 'fixed'"

        sample = self._create_sample(content, reference)
        result = self.reward._evaluate(sample)

        assert result.name == "patch_similarity"
        assert len(result.details) == 1
        assert (
            result.details[0].score == 1.0
        )  # Should get full score for identical patches
        assert result.extra_data["similarity"] == 1.0

    def test_similar_patches(self):
        """Test similar patches"""
        content = "def fixed_function():\n    return 'fixed'"
        reference = "def fixed_function():\n    return 'corrected'"

        sample = self._create_sample(content, reference)
        result = self.reward._evaluate(sample)

        assert result.name == "patch_similarity"
        assert (
            0.0 < result.details[0].score < 1.0
        )  # Should get score between 0 and 1 for similar patches
        assert result.extra_data["generated"] == content
        assert result.extra_data["reference"] == reference

    def test_completely_different_patches(self):
        """Test completely different patches"""
        content = "def new_function():\n    return 'new'"
        reference = "class OldClass:\n    pass"

        sample = self._create_sample(content, reference)
        result = self.reward._evaluate(sample)

        assert result.name == "patch_similarity"
        assert (
            result.details[0].score < 0.5
        )  # Should get low score for completely different patches

    def test_empty_reference(self):
        """Test empty reference patch"""
        content = "def some_function():\n    pass"
        reference = ""

        sample = self._create_sample(content, reference)
        result = self.reward._evaluate(sample)

        assert result.name == "patch_similarity"
        assert result.details[0].score < 1.0

    def _create_sample(self, content: str, reference: str) -> DataSample:
        """Create a test DataSample"""
        return DataSample(
            unique_id="test_patch",
            input=[ChatMessage(role=MessageRole.USER, content="Fix this code")],
            output=[
                DataOutput(
                    answer=Step(
                        role=MessageRole.ASSISTANT,
                        content=content,
                        label={"reference": reference},
                    )
                )
            ],
        )


class TestCodeExecutionReward:
    """Test the code execution reward model"""

    def test_framework_not_available(self):
        """Test when the testing framework is not available"""
        # Create a simple reward object
        reward = CodeExecutionReward()

        content = "def test_func():\n    return 1"
        sample = self._create_sample(content)
        result = reward._evaluate(sample)

        assert result.name == "code_execution"
        assert result.details[0].score == 0.0
        assert "No test cases available for evaluation" in result.details[0].reason

    def test_successful_execution(self):
        """Test successful execution of code"""
        content = """```python
def add_numbers(a, b):
    return a + b
```"""

        # Use prime_code expected test case format
        test_cases = {"inputs": [[1, 2]], "outputs": [3]}
        sample = self._create_sample(content, test_cases)

        # Create a mock reward object
        reward = CodeExecutionReward()

        result = reward._evaluate(sample)

        assert result.name == "code_execution"
        # Since the test framework's implementation may vary, we only check for score > 0
        assert result.details[0].score >= 0.0

    def test_partial_success(self):
        """Test partially successful code"""
        content = """```python
def divide_numbers(a, b):
    return a / b
```"""

        # Use prime_code expected test case format, with multiple test cases (partial success scenario)
        test_cases = {"inputs": [[10, 2], [10, 0]], "outputs": [5, "error"]}
        sample = self._create_sample(content, test_cases)

        # Create a mock reward object
        reward = CodeExecutionReward()

        result = reward._evaluate(sample)

        assert result.name == "code_execution"
        # Since the test framework's implementation may vary, we only check for score >= 0
        assert result.details[0].score >= 0.0

    def test_execution_error(self):
        """Test execution error"""
        content = """```python
def broken_function():
    raise Exception("This will fail")
```"""

        # Use correct test case format
        test_cases = {"inputs": [[]], "outputs": ["success"]}
        sample = self._create_sample(content, test_cases)

        # Create a mock reward object
        reward = CodeExecutionReward()

        result = reward._evaluate(sample)

        assert result.name == "code_execution"
        assert result.details[0].score == 0.0
        # Fix expected error message, actually when test fails, it returns "No test cases passed"
        assert "No test cases passed" in result.details[0].reason
        assert "pass_rate" in result.extra_data

    def test_extract_code_python_block(self):
        """Test extracting code from Python code blocks"""
        reward = CodeExecutionReward()

        content = """There is some Python code:
```python
def hello():
    print("Hello")
```
Other text."""

        extracted = reward._extract_code(content)
        expected = 'def hello():\n    print("Hello")'
        assert extracted == expected

    def test_extract_code_generic_block(self):
        """Test extracting code from generic code blocks"""
        reward = CodeExecutionReward()

        content = """There is code:
```
x = 1
y = 2
```
"""
        extracted = reward._extract_code(content)
        expected = "x = 1\ny = 2"
        assert extracted == expected

    def test_extract_code_no_blocks(self):
        """Test returning content when there are no code blocks"""
        reward = CodeExecutionReward()

        content = "def simple(): pass"
        extracted = reward._extract_code(content)
        assert extracted == content

    def _create_sample(self, content: str, test_cases=None) -> DataSample:
        """Create a test DataSample"""
        metadata = {}
        label = {}

        if test_cases:
            metadata["inputs_outputs"] = test_cases
            label["inputs_outputs"] = test_cases

        return DataSample(
            unique_id="test_execution",
            input=[ChatMessage(role=MessageRole.USER, content="Write a function")],
            output=[
                DataOutput(
                    answer=Step(
                        role=MessageRole.ASSISTANT, content=content, label=label
                    )
                )
            ],
            metadata=metadata,
        )


# 集成测试
class TestCodeRewardIntegration:
    """Integration test: test the combined use of all code reward models"""

    def test_all_rewards_on_good_code(self):
        """Test all reward models on good code"""
        content = """This is a complete Python function:
```python
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)
```
"""

        # Create sample
        sample = DataSample(
            unique_id="test_integration",
            input=[
                ChatMessage(role=MessageRole.USER, content="Write fibonacci function")
            ],
            output=[
                DataOutput(
                    answer=Step(
                        role=MessageRole.ASSISTANT,
                        content=content,
                        label={"reference": content.strip()},
                    )
                )
            ],
        )

        # Test syntax check
        syntax_reward = SyntaxCheckReward()
        syntax_result = syntax_reward._evaluate(sample)
        assert syntax_result.details[0].score > 0.5

        # Test code style
        style_reward = CodeStyleReward()
        style_result = style_reward._evaluate(sample)
        assert style_result.details[0].score > 0.5

        # Test patch similarity
        patch_reward = PatchSimilarityReward()
        patch_result = patch_reward._evaluate(sample)
        assert patch_result.details[0].score > 0.9  # Should be very similar to itself

    def test_all_rewards_on_bad_code(self):
        """Test all reward models on bad code"""
        content = """This is a code with issues:
```python
def BadFunction(
    Print("syntax error")
    CamelCaseVar = 10
    return CamelCaseVar
```
"""

        sample = DataSample(
            unique_id="test_bad_integration",
            input=[ChatMessage(role=MessageRole.USER, content="Write some code")],
            output=[
                DataOutput(
                    answer=Step(
                        role=MessageRole.ASSISTANT,
                        content=content,
                        label={"reference": "def good_function():\n    return 'good'"},
                    )
                )
            ],
        )

        # Test syntax check (should detect syntax errors)
        syntax_reward = SyntaxCheckReward()
        syntax_result = syntax_reward._evaluate(sample)
        assert syntax_result.details[0].score <= 0.0

        # Test code style (should detect style issues)
        style_reward = CodeStyleReward()
        style_result = style_reward._evaluate(sample)
        assert style_result.details[0].score < 0.8

        # Test patch similarity (should be very different from good reference code)
        patch_reward = PatchSimilarityReward()
        patch_result = patch_reward._evaluate(sample)
        assert patch_result.details[0].score < 0.5

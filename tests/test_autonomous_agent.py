"""Integration tests for the Autonomous Software Development Agent."""

import asyncio
from datetime import datetime

from src.fastcoder.errors import ErrorClassifier, ErrorRecoveryCoordinator, RecoveryManager
from src.fastcoder.generator import CodeGenerator, GenerationResult
from src.fastcoder.reviewer import CodeReviewer
from src.fastcoder.tester import TestGenerator, TestGenResult
from src.fastcoder.types.codebase import ProjectProfile
from src.fastcoder.types.errors import ErrorContext, ErrorDetail
from src.fastcoder.types.llm import CompletionResponse, Message
from src.fastcoder.types.plan import PlanTask, TaskAction
from src.fastcoder.types.story import (
    AcceptanceCriterion,
    Priority,
    StorySpec,
    StoryType,
)
from src.fastcoder.types.task import FileChange, TestFailure


# Mock LLM complete function
async def mock_llm_complete(messages: list[Message], metadata: dict) -> CompletionResponse:
    """Mock LLM for testing."""
    # Return appropriate responses based on purpose
    purpose = metadata.get("purpose", "")

    if purpose == "code_generation":
        return CompletionResponse(
            id="mock-1",
            content="""Here's the implementation:

```python
def calculate_total(items: list[dict]) -> float:
    '''Calculate total from items list.

    Args:
        items: List of dicts with 'price' and 'quantity' keys

    Returns:
        Total price as float
    '''
    if not items:
        return 0.0

    total = 0.0
    for item in items:
        if "price" not in item or "quantity" not in item:
            raise ValueError("Item missing price or quantity")
        total += item["price"] * item["quantity"]

    return round(total, 2)
```""",
            model="mock-model",
        )

    elif purpose == "self_reflection":
        return CompletionResponse(
            id="mock-2",
            content="""```json
{
  "correctness": 9,
  "edge_cases": 8,
  "security": 9,
  "consistency": 9,
  "testability": 8,
  "performance": 9,
  "issues": [
    {"axis": "edge_cases", "issue": "Consider very large numbers", "severity": "minor"}
  ]
}
```""",
            model="mock-model",
        )

    elif purpose == "test_generation":
        return CompletionResponse(
            id="mock-3",
            content="""```python
import pytest
from src.order import calculate_total

@pytest.mark.criterion(criterion_id="AC-1")
def test_calculate_total_happy_path():
    '''Test normal calculation path.'''
    items = [{"price": 10.0, "quantity": 2}]
    assert calculate_total(items) == 20.0

@pytest.mark.criterion(criterion_id="AC-1")
def test_calculate_total_multiple_items():
    '''Test with multiple items.'''
    items = [
        {"price": 10.0, "quantity": 2},
        {"price": 5.0, "quantity": 3}
    ]
    assert calculate_total(items) == 35.0

def test_calculate_total_empty_list():
    '''Test edge case: empty list.'''
    assert calculate_total([]) == 0.0

def test_calculate_total_missing_price():
    '''Test error case: missing price.'''
    items = [{"quantity": 2}]
    with pytest.raises(ValueError):
        calculate_total(items)

def test_calculate_total_zero_quantity():
    '''Test boundary: zero quantity.'''
    items = [{"price": 10.0, "quantity": 0}]
    assert calculate_total(items) == 0.0

@pytest.mark.regression
def test_calculate_total_rounding():
    '''Regression: ensure proper rounding.'''
    items = [{"price": 0.1, "quantity": 3}]
    result = calculate_total(items)
    assert result == 0.3
    assert isinstance(result, float)
```""",
            model="mock-model",
        )

    elif purpose == "code_review":
        return CompletionResponse(
            id="mock-4",
            content="""```json
{
  "issues": [
    {
      "severity": "suggestion",
      "category": "maintainability",
      "file": "src/order.py",
      "line": 5,
      "description": "Consider adding type hints to the return value",
      "suggested_fix": "def calculate_total(items: list[dict]) -> float:"
    }
  ],
  "summary": "Good implementation. All acceptance criteria met. Minor suggestions for improvement.",
  "approval": true
}
```""",
            model="mock-model",
        )

    else:
        return CompletionResponse(
            id="mock-5",
            content="Mock response",
            model="mock-model",
        )


# Tests
async def test_code_generator():
    """Test CodeGenerator."""
    print("\n=== Testing CodeGenerator ===")

    generator = CodeGenerator(mock_llm_complete)

    task = PlanTask(
        id="task-1",
        action=TaskAction.CREATE_FILE,
        target="src/order.py",
        description="Create a function to calculate total price from items",
    )

    context = {
        "project_profile": ProjectProfile(
            language="python",
            test_framework="pytest",
        ),
        "relevant_files": {},
        "type_definitions": "Item = Dict[str, float]",
        "conventions": "Use snake_case for functions, PascalCase for classes",
        "error_history": [],
    }

    result: GenerationResult = await generator.generate(task, context)

    print(f"Generated code: {result.code[:100]}...")
    print(f"Confidence: {result.confidence}")
    print(f"File changes: {len(result.file_changes)}")
    print(f"Issues found: {len(result.reflection_issues)}")

    assert result.code, "Should generate code"
    assert result.confidence > 0, "Should have confidence score"
    assert len(result.file_changes) > 0, "Should have file changes"


async def test_test_generator():
    """Test TestGenerator."""
    print("\n=== Testing TestGenerator ===")

    tester = TestGenerator(mock_llm_complete)

    task = PlanTask(
        id="task-1",
        action=TaskAction.CREATE_FILE,
        target="src/order.py",
        description="Create a function to calculate total price",
    )

    spec = StorySpec(
        title="Order Total Calculator",
        description="Calculate total price from items",
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC-1",
                description="Must handle multiple items",
            ),
        ],
    )

    context = {
        "test_framework": "pytest",
        "existing_tests": "",
        "project_conventions": "Use test_<function>_<scenario> naming",
    }

    result: TestGenResult = await tester.generate_tests(
        task,
        "def calculate_total(items): pass",
        spec,
        context,
    )

    print(f"Generated tests: {result.test_code[:100]}...")
    print(f"Test file: {result.test_file}")
    print(f"Coverage estimate: {result.coverage_estimate}")
    print(f"Edge cases covered: {result.edge_cases_covered}")
    print(f"Criteria mapping: {result.criteria_mapping}")

    assert result.test_code, "Should generate test code"
    assert "test_" in result.test_file, "Test file should have test_ prefix"
    assert result.coverage_estimate >= 0, "Should estimate coverage"


async def test_code_reviewer():
    """Test CodeReviewer."""
    print("\n=== Testing CodeReviewer ===")

    reviewer = CodeReviewer(mock_llm_complete)

    changes = [
        FileChange(
            file_path="src/order.py",
            change_type="created",
            content="def calculate_total(items): return sum(item['price'] * item['quantity'] for item in items)",
        ),
    ]

    spec = StorySpec(
        title="Order Total",
        description="Calculate order total",
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC-1",
                description="Must calculate total correctly",
            ),
        ],
    )

    profile = ProjectProfile(
        naming_conventions={
            "functions": "snake_case",
            "classes": "PascalCase",
        }
    )

    report = await reviewer.review(changes, spec, profile)

    print(f"Approved: {report.approved}")
    print(f"Issues found: {len(report.issues)}")
    print(f"Summary: {report.summary[:100]}...")

    assert isinstance(report.approved, bool), "Should have approval decision"
    assert isinstance(report.issues, list), "Should have issues list"


async def test_error_classifier():
    """Test ErrorClassifier."""
    print("\n=== Testing ErrorClassifier ===")

    classifier = ErrorClassifier()

    # Test syntax error
    classification = classifier.classify(
        error_type="SyntaxError",
        message="Invalid syntax on line 5",
    )
    print(f"SyntaxError classification: {classification.category.value}")
    assert classification.category.value == "syntax_error"
    assert classification.confidence > 0.9

    # Test type error
    classification = classifier.classify(
        error_type="TypeError",
        message="Expected str, got int",
    )
    print(f"TypeError classification: {classification.category.value}")
    assert classification.category.value == "type_error"

    # Test import error
    classification = classifier.classify(
        error_type="ModuleNotFoundError",
        message="No module named 'requests'",
    )
    print(f"ImportError classification: {classification.category.value}")
    assert classification.category.value == "import_error"

    # Test fingerprint uniqueness
    fp1 = classifier.generate_fingerprint("TypeError", "Expected str, got 123")
    fp2 = classifier.generate_fingerprint("TypeError", "Expected str, got 456")
    print(f"Fingerprint 1: {fp1}")
    print(f"Fingerprint 2: {fp2}")
    assert fp1 == fp2, "Dynamic values should be normalized"

    # Test different error types have different fingerprints
    fp3 = classifier.generate_fingerprint("TypeError", "Expected str")
    fp4 = classifier.generate_fingerprint("ValueError", "Expected str")
    assert fp3 != fp4, "Different error types should have different fingerprints"


async def test_recovery_manager():
    """Test RecoveryManager."""
    print("\n=== Testing RecoveryManager ===")

    classifier = ErrorClassifier()
    recovery_mgr = RecoveryManager()

    classification = classifier.classify(
        error_type="SyntaxError",
        message="Invalid syntax",
    )

    # First attempt
    action = recovery_mgr.get_strategy(classification, attempt=1)
    print(f"Attempt 1 strategy: {action.strategy.value}")
    print(f"Switch to top tier: {action.switch_to_top_tier}")
    assert action.max_retries >= 1

    # Record and lookup fix
    fingerprint = classification.fingerprint
    recovery_mgr.record_fix(fingerprint, "fixed_code", "story-1")
    found_fix = recovery_mgr.lookup_fix(fingerprint)
    print(f"Known fix lookup: {found_fix}")
    assert found_fix == "fixed_code"

    # Test progressive context enrichment
    action2 = recovery_mgr.get_strategy(classification, attempt=2)
    print(f"Attempt 2 context level: {action2.additional_context.get('context_level', 'unknown')}")

    action3 = recovery_mgr.get_strategy(classification, attempt=3)
    print(f"Attempt 3 context level: {action3.additional_context.get('context_level', 'unknown')}")


async def test_error_recovery_coordinator():
    """Test ErrorRecoveryCoordinator."""
    print("\n=== Testing ErrorRecoveryCoordinator ===")

    classifier = ErrorClassifier()
    recovery_mgr = RecoveryManager()
    coordinator = ErrorRecoveryCoordinator(classifier, recovery_mgr)

    classification, action = coordinator.handle_error(
        error_type="TypeError",
        message="Expected string, got int",
        attempt=1,
    )

    print(f"Classification: {classification.category.value}")
    print(f"Recovery strategy: {action.strategy.value}")
    print(f"Escalate to human: {action.escalate}")

    assert classification.category.value == "type_error"
    assert action.strategy.value == "include_types"


async def test_integration():
    """Test full integration flow."""
    print("\n=== Testing Full Integration ===")

    # Create all components
    generator = CodeGenerator(mock_llm_complete)
    tester = TestGenerator(mock_llm_complete)
    reviewer = CodeReviewer(mock_llm_complete)
    classifier = ErrorClassifier()
    recovery_mgr = RecoveryManager()
    coordinator = ErrorRecoveryCoordinator(classifier, recovery_mgr)

    # Create task
    task = PlanTask(
        id="task-1",
        action=TaskAction.CREATE_FILE,
        target="src/order.py",
        description="Create order total calculator",
    )

    spec = StorySpec(
        title="Order Total",
        description="Calculate order total",
        story_type=StoryType.FEATURE,
        complexity_score=5,
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC-1",
                description="Must handle multiple items",
            ),
        ],
    )

    context = {
        "project_profile": ProjectProfile(language="python"),
        "relevant_files": {},
        "type_definitions": "",
        "conventions": "",
        "error_history": [],
        "test_framework": "pytest",
    }

    # Step 1: Generate code
    print("\n1. Generating code...")
    gen_result = await generator.generate(task, context)
    print(f"   Generated {len(gen_result.code)} chars of code")

    # Step 2: Generate tests
    print("2. Generating tests...")
    test_result = await tester.generate_tests(task, gen_result.code, spec, context)
    print(f"   Generated {len(test_result.test_code)} chars of test code")

    # Step 3: Review code
    print("3. Reviewing code...")
    review_result = await reviewer.review(gen_result.file_changes, spec, context["project_profile"])
    print(f"   Approved: {review_result.approved}, Issues: {len(review_result.issues)}")

    # Step 4: Handle error scenario
    print("4. Handling error scenario...")
    classification, action = coordinator.handle_error(
        error_type="NameError",
        message="name 'total' is not defined",
        attempt=1,
    )
    print(f"   Error category: {classification.category.value}")
    print(f"   Recovery strategy: {action.strategy.value}")

    print("\nIntegration test passed!")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("AUTONOMOUS SOFTWARE DEVELOPMENT AGENT - INTEGRATION TESTS")
    print("=" * 60)

    try:
        await test_code_generator()
        await test_test_generator()
        await test_code_reviewer()
        await test_error_classifier()
        await test_recovery_manager()
        await test_error_recovery_coordinator()
        await test_integration()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())

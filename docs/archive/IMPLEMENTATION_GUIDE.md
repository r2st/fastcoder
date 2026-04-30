# Autonomous Software Development Agent — Implementation Guide

## Overview

This document describes the implementation of three core components for the Autonomous Software Development Agent in TypeScript:

1. **Code Generator** — Generates production-ready code with self-reflection
2. **Test Generator** — Creates comprehensive test suites with criterion mapping
3. **Code Reviewer** — Performs independent code reviews with multi-axis analysis
4. **Error Classification & Recovery** — Classifies errors and routes to appropriate recovery strategies

## Architecture

### 1. Code Generator (`src/generator/index.ts`)

The `CodeGenerator` class generates code using an LLM with self-reflection and error handling capabilities.

#### Key Features

- **Prompt Construction**: Builds rich context including task description, project conventions, type definitions, and relevant files
- **Code Parsing**: Extracts code from markdown code blocks in LLM responses
- **Syntax Validation**: Validates bracket matching, import statements, and basic structure
- **Self-Reflection**: Second-pass validation checking correctness, edge cases, security, consistency, testability, and performance
- **Auto-Fixes**: Applies automatic fixes when confidence score is below 0.7
- **Error Handling**: Progressive context enrichment and model escalation on attempt 5+

#### API

```typescript
const generator = new CodeGenerator(llmRouter, contextManager);

// Generate code for a task
const result = await generator.generate(
  task,
  generationContext,
  storySpec,
  storyId
);

// Fix code based on error
const fixedResult = await generator.fix(
  task,
  errorContext,
  generationContext,
  storySpec,
  storyId
);
```

#### Self-Reflection Protocol

After code generation, the generator runs a second-pass evaluation on six axes:

1. **Correctness** (0-1): Does it implement the task correctly?
2. **Edge Cases** (0-1): Are edge cases handled?
3. **Security** (0-1): Any vulnerabilities or unsafe patterns?
4. **Consistency** (0-1): Does it follow conventions?
5. **Testability** (0-1): Is it easy to test?
6. **Performance** (0-1): Any efficiency issues?

If overall confidence < 0.7, the generator automatically applies fixes and re-evaluates.

### 2. Test Generator (`src/tester/index.ts`)

The `TestGenerator` class creates comprehensive test suites with acceptance criteria mapping.

#### Key Features

- **Dual Modes**:
  - **Coverage Mode**: Aims for 80%+ code coverage, tests all branches
  - **Criteria Mode**: Generates tests directly from acceptance criteria
- **Criterion Mapping**: Links each test to specific acceptance criteria with @criterion annotations
- **Edge Case Testing**: Includes null inputs, empty collections, boundaries, error states
- **Framework Support**: Jest, Vitest, pytest, Mocha
- **Regression Tests**: Creates lock tests to prevent reoccurrence of fixed bugs

#### Test Annotation Format

```typescript
// Each test includes @criterion annotation
it('should validate standard email format', () => {
  // @criterion AC1
  expect(validateEmail('test@example.com')).toBe(true);
});
```

### 3. Code Reviewer (`src/reviewer/index.ts`)

The `CodeReviewer` class performs independent code reviews using a different LLM provider for objectivity.

#### Key Features

- **Multi-Axis Review**:
  - Security vulnerabilities
  - Performance anti-patterns
  - Logical correctness
  - Convention adherence
  - Maintainability concerns
- **Three-Tier Issue Classification**:
  - **Blocking**: Must fix before merge
  - **Suggestion**: Should consider
  - **Nit**: Nice to have
- **Structured Feedback**: Each issue includes severity, category, file, line, description, and suggested fix

### 4. Error Classification & Recovery

#### ErrorClassifier (`src/errors/classifier.ts`)

Classifies errors into categories and generates fingerprints for pattern matching.

#### Supported Error Categories

| Category | Pattern Matches | Recovery Strategy |
|----------|-----------------|-------------------|
| `syntax_error` | SyntaxError, Unexpected token | direct_fix |
| `type_error` | TypeError, TS2\d{3} | include_types |
| `import_error` | Cannot find module, ENOENT | consult_symbol_table |
| `logic_error` | AssertionError | include_broad_context |
| `integration_error` | NetworkError, ECONNREFUSED | load_api_specs |
| `environment_error` | ENOENT, EACCES | environment_repair |
| `flaky_error` | timeout, ETIMEDOUT | rerun |
| `architectural_error` | stack overflow, recursion | replan |

#### Fingerprinting

The classifier generates error fingerprints by:

1. Normalizing error type and message
2. Stripping dynamic values (file paths, line numbers, quoted strings)
3. Creating SHA-256 hash of normalized pattern
4. Returning first 16 characters as fingerprint

#### RecoveryManager (`src/errors/recovery.ts`)

Manages error recovery strategies and maintains a database of known fixes.

#### Known Fixes Database

Pre-initialized with common fixes:

```typescript
manager.recordError('fingerprint', {
  fingerprint: 'syntax_missing_bracket',
  category: 'syntax_error',
  fix_description: 'Missing closing bracket or brace',
  success_rate: 0.95,
  last_used: new Date(),
});
```

## Implementation Details

### Type Safety

All files use `import type` for TypeScript-only imports and include `.js` extensions for ESM:

```typescript
import type { PlanTask } from '../types/plan.js';
import { CodeGenerator } from '../generator/index.js';
```

### Error Handling

The generator includes robust error handling:

1. **Syntax Validation**: Checks bracket matching before accepting generated code
2. **Type Checking**: Validates imports and type compatibility
3. **Progressive Escalation**: Switches to higher-tier models on repeated failures (attempt 5+)
4. **Context Enrichment**: Includes more information with each retry

### Self-Reflection Implementation

The generator uses LLM analysis to evaluate code quality on six axes and computes an overall confidence score. If confidence < 0.7, it automatically applies fixes.

### Test Criterion Mapping

The test generator parses @criterion annotations and maps tests to acceptance criteria:

```typescript
// Parse from test code
// @criterion AC1
// Maps: AC1 → [testName1, testName2, ...]
```

## Testing

Comprehensive test suites provided:

- `src/generator/__tests__/index.test.ts`
- `src/errors/__tests__/classifier.test.ts`
- `src/errors/__tests__/recovery.test.ts`

Run with: `npm test`

## Integration

Components integrate via the orchestrator:

```typescript
const generator = new CodeGenerator(llmRouter, contextManager);
const tester = new TestGenerator(llmRouter);
const reviewer = new CodeReviewer(llmRouter, reviewConfig);

// Generation pipeline
const genResult = await generator.generate(task, context, spec, storyId);
const testResult = await tester.generateTests(task, genResult.code, spec, ...);
const reviewReport = await reviewer.review(genResult.fileChanges, spec, ...);
```

## Files Created

```
src/
├── generator/
│   ├── index.ts                 (643 lines)
│   └── __tests__/index.test.ts
├── tester/
│   └── index.ts                 (393 lines)
├── reviewer/
│   └── index.ts                 (453 lines)
└── errors/
    ├── classifier.ts            (258 lines)
    ├── recovery.ts              (344 lines)
    ├── index.ts                 (13 lines)
    ├── __tests__/classifier.test.ts
    └── __tests__/recovery.test.ts
```

**Total: 2,104 lines of implementation code with comprehensive test coverage**

All code follows TypeScript strict mode, uses ESM with `.js` extensions, integrates with ModelRouter and project configuration.

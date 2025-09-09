# Copilot Instructions for Python / VS Code Project

> These instructions tell GitHub Copilot **how to write and refactor Python code in VS Code** for this repository. 
---

## Project Context
This is a Python project developed in **VS Code**. Suggestions must follow **modern Python best practices** and **conservative engineering defaults** that favor correctness, readability, testability, and security.

---

## Tooling & Extensions (assume installed)
- **Python** (Microsoft) + **Pylance** for analysis
- **Ruff** (formatter + linter + import sorter)
- **pytest** (+ `pytest-asyncio` where async)
- **GitHub Copilot** + **Copilot Chat**
- Optional: **Jupyter** (if notebooks are in repo), **mypy** (if stricter typing needed)

---

## Language & Runtime Preferences
- **Python**: 3.11+ (prefer **3.12** if available)
- Use **type hints** everywhere (PEP 484/526/695).
- Prefer **stdlib** first; add third-party deps only if there’s a clear value.
- Use **f-strings** for formatting.
- Install dependencies in the **virtual environment** for this project .venv/
- Run and test code using the **Python extension** in VS Code in the **virtual environment** .venv/. 

---

## Code Style Guidelines
- **Indentation**: 4 spaces.
- **Line length**: 88 (Black default).
- **Naming** (PEP 8):
  - `snake_case` for functions/variables
  - `PascalCase` for classes
  - `UPPER_SNAKE_CASE` for constants
- **Imports**: absolute, grouped stdlib / third-party / local, and sorted (Ruff).
- **Docstrings**: use **Google style** or **NumPy style** consistently (pick one per repo).
- **Typing**:
  - Prefer `collections.abc` (e.g., `Iterable`, `Mapping`) over concrete types.
  - Use `TypedDict`/`dataclass` for structured data.
  - Leverage `typing.Self`, `ParamSpec`, `TypeVar` when helpful.

---

## Preferred Patterns & Practices
- **Functional core, imperative shell**: isolate pure logic; keep I/O at boundaries.
- **Small modules** and **single responsibility**; avoid god objects.
- **Dependency injection** for external services/clients.
- **Configuration** via environment variables (12-factor), with centralized parsing/validation.
- **Logging**: use `logging` (no `print`) with structured/contextual logs.
- **Resource safety**: use context managers for files, sessions, locks.
- **Narrow scope** for functions.
- **Modularization**: break large functions into smaller ones. High cohesion, low coupling.
- **Preferred architectural patterns**: layered, hexagonal, or clean architecture where applicable. Event-driven as appropriate.

---

## Async/Concurrency
- Prefer **`asyncio`** (native `async/await`) where concurrency is I/O-bound.
- Avoid mixing threads and coroutines casually; if needed, isolate via `anyio` or thread pools.
- Never block the event loop; offload CPU-bound work.

---

## Error Handling
- Raise **specific, documented exceptions**; avoid broad `except Exception`.
- Validate inputs early; **fail fast** with clear messages and logging.
- Wrap external boundaries (network, DB, filesystem) with retry/backoff where appropriate.
- For CLI/entrypoints, convert exceptions to non-zero exit codes with helpful messages.

---

## Testing Preferences
- Use **pytest** with **Arrange-Act-Assert** structure.
- Cover **happy path + edge cases + failure modes**.
- **Mock** external I/O; prefer **fake/test containers** for integration boundaries.
- Add **property-based** tests (Hypothesis) where it pays off.
- Keep tests **fast and deterministic**.

---

## Dependencies & Package Management
- Prefer **uv** or **pip-tools** (`pip-compile`) for reproducible lock files; otherwise `requirements.txt` with pinned versions.
- For apps, use virtualenvs; for libraries, provide **PEP 621** `pyproject.toml`.
- Avoid heavyweight frameworks unless justified; add one dep at a time with tests.

---

## Documentation Style
- Docstrings on **all** **modules/classes/functions**.
- Include **Args**, **Returns**, **Raises**, and short examples where useful.
- Keep README usage examples runnable (doctest-friendly where possible).
- Avoid inline comments. Never use pronouns -- e.g., "you" or "we" should never appear in comments.
- Docstring comments should explain both **why** and **what** a function does.
- Each function should have a docstring, even if it's just a brief description of its purpose. Longer functions should have more detailed docstrings explaining their behavior, inputs, outputs, and any exceptions they might raise.

**Docstring template (Google style):**
```python
def foo(bar: int, /, baz: str = "x") -> str:
    """Short summary.

    Args:
        bar: Meaningful description.
        baz: Optional behavior switch.

    Returns:
        Description of return value.

    Raises:
        ValueError: When input is invalid.
    """
```

---

## Performance Considerations
- Write idiomatic, clear code first; optimize with **profiles**, not hunches.
- Prefer generators/iterators for streams; avoid building large interim lists.
- Cache pure computations where measurable (`functools.lru_cache`).
- Be explicit about algorithmic complexity in comments when non-trivial.

---

## Code Organization
- Package layout:
  ```
  package_name/
    __init__.py
    module_a.py
    module_b.py
  tests/
    test_module_a.py
  scripts/        # thin wrappers calling package code
  ```
- Keep **domain logic** inside the package, not in scripts or notebooks.
- Use `__all__` sparingly to define public API.

---

## Security, Privacy & Licensing
- No secrets in code; use env vars or secret managers.
- Validate/escape external input; never eval untrusted strings.
- Note license implications when Copilot surfaces boilerplate similar to public repos; prefer **original implementations** and **stdlib** first.
- Avoid sending sensitive data to prompts (commands, stack traces) unless sanitized.

---

## Common Patterns to Avoid
- Wildcard imports (`from x import *`)
- Mutable default args
- Global state and singletons (except for module-level constants)
- Inconsistent typing; avoid optional/`Any` sprawl
- “Just print and hope” debugging

---

## File/Repo Structure Preferences
- Keep modules ≤ 300 SLOC when practical; refactor early if exceeding.
- One top-level concept per file.
- Co-locate tests near code under `tests/` with mirroring names.

---

## VS Code Settings (example)
*(Adjust keys if extensions change names in future releases.)*
```json
{
  // Editor
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "charliermarsh.ruff",
  "editor.inlineSuggest.enabled": true,

  // Python analysis / types
  "python.analysis.typeCheckingMode": "strict",

  // Ruff (formatter + linter + imports)
  "ruff.lint.enable": true,
  "ruff.organizeImports": true,
  "ruff.lint.select": ["E","F","I","B","UP","SIM","PLC"],
  "ruff.lint.ignore": ["E501"],

  // Testing
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["-q"],

  // Copilot (enable globally; disable in plaintext/markdown if noisy)
  "github.copilot.enable": {
    "*": true,
    "markdown": false,
    "plaintext": false
  }
}
```

---

## How to Prompt Copilot (in this repo)
**General tone:** be explicit, constrain scope, and specify acceptance criteria. Provide **signatures**, **types**, and **io boundaries**.

**Good prompt examples**
- “Write a `validate_user(payload: Mapping[str, Any]) -> User` that:
  - performs schema validation with clear `ValueError` messages,
  - returns a `@dataclass` `User`,
  - has 100% branch coverage under `pytest`,
  - includes Google-style docstring and type hints,
  - no I/O or globals.”
- “Refactor `client.py` into `Client` with dependency-injected `requests.Session`, timeouts, retries (3, exp backoff), and structured logging. Keep public API backward-compatible and add tests for retryable/non-retryable errors.”

**Anti-patterns (don’t do)**
- “Write me a service layer” (too vague)
- “Make it faster” (no target or metric)
- “Just fix this” (no definition of done)

**Comment-driven scaffolds**
```python
# Goal: pure function, no I/O. Inputs validated. Typed. Raise ValueError on bad data.
# Output: dataclass with normalized fields. Keep it O(n).
```

---

## Example Test & Impl Stubs Copilot Should Follow
**Test first**
```python
# tests/test_slugify.py
import pytest
from package_name.slug import slugify

@pytest.mark.parametrize("raw, expected", [
    ("Hello, World!", "hello-world"),
    ("  a  b  ", "a-b"),
    ("Ümläüt", "umlaut"),
])
def test_slugify(raw, expected):
    assert slugify(raw) == expected
```

**Implementation**
```python
# package_name/slug.py
import re
import unicodedata
from typing import Final

_SEP: Final[str] = "-"

def _ascii(s: str) -> str:
    return (
        unicodedata.normalize("NFKD", s)
        .encode("ascii", "ignore")
        .decode("ascii")
    )

def slugify(text: str) -> str:
    """Return a URL-safe slug for *text*."""
    if not isinstance(text, str):
        raise ValueError("text must be str")
    s = _ascii(text).lower()
    s = re.sub(r"[^a-z0-9]+", _SEP, s).strip(_SEP)
    s = re.sub(rf"{_SEP}+", _SEP, s)
    return s
```

---

## Copilot Chat Usage Patterns
- Provide **diff-style** refactors by default. Assume: “Show a diff that extracts `parse_config()` and adds tests.”
- Provide **explanations with citations** to PEPs or docs by default.
- Decompose long, complex prompts into **small, iterative prompts** to avoid monoliths; keep the context tight (single file or function) for each iteration. Confirm each iteration is correct before proceeding.

---

## Misc/Other Instructions and Policies
- Prefer **stdlib** and **small utilities** over frameworks; reduce transitive risk.
- For notebooks, keep them **thin**; move logic into importable modules and test there.
- Avoid outdated APIs or patterns, **prefer the modern alternative** (e.g., `pathlib` over `os.path`). Check for latest APIs and accepted patterns.
- If a compact/simplified variant improves reliability with negligible cost, **choose the simpler path**.
- If a suggestion introduces magic constants or unclear behavior, **inline comments** may be used explain rationale or refactor to named constants.
- When refactoring, ensure **backward compatibility** unless explicitly told to break it.
- Stay within the **scope of the prompt**. Ask for clarification if the prompt is ambiguous or incomplete. Ask before making assumptions. Do not attempt to optimize or refactor beyond the stated requirements without first confirming with the user.

---

### Definition of Done (for Copilot-generated code)
- Passes `ruff` (lint + format) and `pytest`.
- Fully typed, with docstrings on public interfaces.
- No new global state; no network/disk I/O in pure functions.
- Measurable behavior covered by tests (including error cases).

---

*End of instructions.*

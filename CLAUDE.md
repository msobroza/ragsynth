# ragsynth — agent rules

Synthetic query generation & validation for RAG evaluation. `SPEC.md` is v1 law;
`PLAN.md` records decisions (D1+); `specs/v2/` holds the v2 execution specs.

Validation commands (run all before considering any change done):

```
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest -q
```

## Coding Rules for AI Agents

Short, imperative rules optimized for code that will be read, edited, and extended
primarily by AI coding agents. Keep this file dense — it is re-read on every iteration
and every line costs context tokens.

### Code style

* Functions: 4–20 lines. If longer, split it.
* Files: under 500 lines (ideally 200–300). Split by responsibility. A single small file must fit in one read without truncation.
* One thing per function. One responsibility per module (SRP). A module has exactly one reason to change.
* Names: specific, unique, and greppable. Avoid generic names like data, process, handler, Manager, Service, utils. Prefer names that return fewer than 5 grep hits across the codebase (e.g., UserRegistrationValidator, InvoiceLineItemTotal).
* Types: explicit everywhere. No any, no bare Dict, no untyped function signatures. Use type hints in Python, TypeScript over plain JavaScript, RBS/Sorbet in Ruby. The signature must state what goes in, what comes out, and which states are valid.
* No duplication (DRY). Extract shared logic into a single function or module. Duplicated logic gets updated in one place and silently forgotten in the others.
* Early returns and guard clauses over nested conditionals. Maximum 2 levels of indentation. Flatten if-inside-for-inside-try structures.
* Error messages must carry context: include the offending value and the expected shape. Bad: raise ValueError("invalid input"). Good: raise ValueError(f"invalid input: received {repr(x)}, expected non-empty string of digits"). Exception messages are debugging signals — vague messages force extra investigation rounds.

### Comments

* Write WHY, not WHAT. Never annotate the obvious (// increment i above i++ is forbidden — it wastes tokens and adds nothing).
* Preserve existing comments during refactors. Do not strip comments that carry intent, provenance, or decision context — they exist because that information was judged worth keeping for future edits.
* Record provenance: why this approach was chosen over the obvious one, which production bug motivated unusual logic, which business constraint forces a specific ordering, which upstream library bug a workaround exists for. Reference issue numbers or commit SHAs when a line exists because of a specific bug or external constraint.
* Public functions get a docstring: intent plus one usage example (JSDoc, Python """docstring""", Rust ///). Update the docstring together with the code whenever behavior changes.

### Tests

* Every test must run headless with a single documented command (e.g., make test, pnpm test, pytest). Put the command in the README and in this file. No manual setup, no undocumented config files, no secret credentials, no manual database seeding.
* Every new function gets a test. Every bug fix gets a regression test.
* Follow F.I.R.S.T: Fast, Independent, Repeatable, Self-Validating, Timely.
* Mock external I/O (APIs, databases, filesystem, network) with named fake classes (e.g., FakeEmailSender), not inline stubs or ad-hoc monkey patches.
* Write the test before or together with the code. Run the full test suite before considering any change done. The write → run → read output → adjust loop is the core workflow; a change without a passing test run is not finished.
* Keep test output in a predictable, parseable format.

### Dependencies

* Inject dependencies through constructors or parameters — never hardcode them or reach for globals/direct imports inside business logic. Injected dependencies can be swapped for fakes in tests without touching the logic.
* Wrap third-party libraries behind a thin interface owned by this project.
* Centralize configuration (model names, endpoints, feature flags) in one place. A value referenced in many files must live in a single constant so changing it is a one-line edit.

### Structure

* Follow the framework's directory conventions (Rails, Django, Next.js, Laravel, etc.). Predictable paths let file locations be inferred without listing directories.
* Prefer many small focused modules over god files. Three 250-line classes beat one 800-line class doing three things.
* Keep the tree conventional: controllers/models/views, src/lib/test, and similar recognizable layouts.

### Formatting

* Use the language's default or most popular formatter and nothing else: cargo fmt (Rust), gofmt (Go), prettier (JS/TS), black or ruff (Python), rubocop -A (Ruby). Run it in pre-commit and on save.
* Never debate style (tabs vs spaces, line width, brace placement). The formatter decides. Consistency keeps diffs clean and avoids wasted attention parsing inconsistent layout.

### Error handling & defensive code

* Implement the defensive patterns this project requires — do not stop at the happy path. Cover, where applicable: input validation at boundaries, timeouts on all external calls, retry with exponential backoff, circuit breakers on flaky dependencies, rate limiting, graceful degradation and fallbacks.
* Fail loudly with context in internal code; degrade gracefully at user-facing boundaries.

### Logging

* Use structured JSON logging with named fields for debugging and observability. JSON is trivially parseable and filterable; free-text printf logs require heuristic parsing.
* Plain text is acceptable only for user-facing CLI output.

### Project meta-documentation

* Keep this rules file short, imperative, and action-oriented. No philosophical prose. Bullet points of what is needed to avoid mistakes.
* Maintain a README with a high-level architecture overview. A simple ASCII or Mermaid diagram of the system shape is worth including.
* Expose validation commands prominently (make lint, pnpm test, cargo check, python -m mypy) in the README, Makefile, or package scripts so changes can always be verified.
* Keep setup scripts idempotent: bin/setup or scripts/bootstrap.sh must take a clean machine to a working state with no human-only knowledge required.

### Rationale (why these rules exist)

* Agents read files in limited chunks (typically ~2000 lines per tool call); small files and small functions fit in a single read and get full-attention reasoning instead of a fragmented mental model.
* Attention quality degrades as the context window fills. The window also holds system prompts, conversation history, tool output, and logs — compact code and compact rules leave room for actual work.
* Agents navigate codebases primarily via grep, not by reading everything. Unique, searchable names are the primary navigation API; generic names produce dozens of irrelevant matches.
* Every tool call costs tokens and latency. Short files, small test output, and lean logs keep the loop fast and cheap.
* Duplicated code is dangerous for automated refactors: an agent can update one copy and miss the others, since nothing in its attention window naturally surfaces distant duplicates.
* Comments are first-class context for an agent. It has perfect syntax fluency and needs no "what" explanations, but it cannot know the "why" behind decisions unless it is written down where the code lives.
* Agents do almost none of this by default. Without explicit rules they produce average code: long functions, no DI, missing or wrong-mocked tests, duplicated logic, giant files. These rules exist because they must be stated to be followed.

# Testing Workflow Rules

- **Skip Tests for Questions & Small Changes**: Do not run test suites or execute notebooks for user questions, documentation/comment edits, or small isolated code changes unless the user explicitly asks to verify them.
- **Run Tests Intelligently**: Reserve test execution for substantive multi-file or core logic changes, and run only the targeted test script(s) once all planned edits in a task are complete—never re-run tests after every tiny incremental tweak.
- **Confirm Before Heavy Checks**: When unsure whether an expensive verification step (such as executing full notebooks, end-to-end runs, or full test suites) is needed after minor adjustments, ask the user rather than launching it automatically.

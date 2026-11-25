# MiniReasoner Agent Guidelines

This directory defines standard rules and guidelines for autonomous agents operating in the MiniReasoner codebase.

## Active Rules
- **[Git Safety](rules/git_safety.md)**: Prohibits committing or pushing without explicit user permission; mandates consolidated amends.
- **[Training Workflow](rules/training_workflow.md)**: Enforces `tmux` background execution, 4-digit loss formatting (`.4f`), and per-run file logging in `logs/`.
- **[Code Organization](rules/code_organization.md)**: Enforces keeping Python code in `src/` (with `src/main.py` as entrypoint), `.env`-only credential configuration, and test suites in `tests/`.
- **[Testing Workflow](rules/testing_workflow.md)**: Enforces running tests selectively after substantive changes—never for user questions, docs, or small isolated edits.

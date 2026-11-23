# Git Safety Rules

- **No Unauthorized Commits**: Never run `git commit` without explicit user permission.
- **No Unauthorized Pushes**: Never run `git push` without explicit user permission.
- **Commit Consolidation**: When amending changes into a specific commit, ensure all new files and modifications are consolidated cleanly into that single commit (`git commit --amend`).
- **Secrets Protection**: Never stage or commit credentials, API keys, tokens, or environment files (`.env`).

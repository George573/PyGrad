# Editing permissions

Codex may edit files under these directories without confirmation:

- `tests/**`
- `usage_examples/**`

Before editing, creating, deleting, renaming, or formatting any file outside those
directories, Codex must:

1. Explain the proposed change.
2. List the files it intends to modify.
3. Ask for explicit confirmation.
4. Wait for confirmation before making the change.

Codex may read any repository file and run non-destructive checks or tests without
confirmation.

A user request to "fix," "update," or "implement" something does not implicitly
authorize changes outside `tests/**` and `usage_examples/**`.

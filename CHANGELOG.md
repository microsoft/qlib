# Changelog

## Unreleased

- **BREAKING:** New source builds restrict recorder artifact loading by default.
  Reloading executable artifacts requires verified source/storage and explicit
  `trusted=True` (CLI: `--trusted=True`). Supported data-only reads and fresh
  in-memory training need no opt-in. See the
  [artifact loading migration guide](https://qlib.readthedocs.io/en/latest/start/artifact_migration.html)
  for workflow, HIST and high-frequency cache upgrades.
- Merging into `main` affects source installs before a PyPI release. These changes
  remain unreleased until included in a tagged release; its versioned upgrade notes
  should link to the same guide.

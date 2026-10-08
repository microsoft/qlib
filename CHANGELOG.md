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

### Configuration-driven execution

- Local `.py` imports require explicit boolean consent: top-level `trusted: true`
  on each component configuration, or `trusted=True` on a direct
  `get_module_by_module_path` call. Package imports and class objects are unchanged.
- Directory-root authorization from earlier PR revisions was removed. Consent is
  neither global nor inherited, and it does not authorize artifact loading.
- Feature expressions use a restricted AST interpreter; custom built-in TRA
  backbones and model-performance graph names use explicit mappings.

See the [configuration migration guide](docs/start/config_migration.rst) for the
upgrade checklist, Python/YAML examples, extension registration, and trusted
older file-model pickle recovery. These changes remain unreleased until included
in a tagged/PyPI release.

The full release history is maintained in [CHANGES.rst](CHANGES.rst).

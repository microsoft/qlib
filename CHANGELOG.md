# Changelog

## Unreleased: configuration-driven execution

- Local `.py` imports require explicit boolean consent: top-level `trusted: true`
  on each component configuration, or `trusted=True` on a direct
  `get_module_by_module_path` call. Package imports and class objects are unchanged.
- Directory-root authorization from earlier PR revisions was removed. Consent is
  neither global nor inherited, and it does not authorize artifact loading.
- Feature expressions use a restricted AST interpreter; custom built-in TRA
  backbones and model-performance graph names use explicit mappings.

See the [configuration migration guide](docs/start/config_migration.rst) for the
upgrade checklist, Python/YAML examples, extension registration, and trusted
older file-model pickle recovery. These are PR #2340 source changes, not a
statement that `main` or a tagged/PyPI release contains them. PR #2339's artifact
permissions are a separate unreleased change.

The full release history is maintained in [CHANGES.rst](CHANGES.rst).
# Configuration migration

Use examples matching your installed Qlib revision: PR source, `main`, and
tagged/PyPI releases can differ. For the unreleased file-import, expression, and
extension-registry changes, start with the
[configuration migration guide](../docs/start/config_migration.rst).
Reviewed local `.py` components declare their own top-level `trusted: true`;
package imports need no new permission. This does not make untrusted YAML or
artifacts safe.

# Requirements

Here is the minimal hardware requirements to run the `workflow_by_code` example.
- Memory: 16G
- Free Disk: 5G


# NOTE
The results will slightly vary on different OSs(the variance of annualized return will be less than 2%).
The evaluation results in the `README.md` page are from Linux OS.

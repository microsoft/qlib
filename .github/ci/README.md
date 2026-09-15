# CI compatibility baseline

The three Python workflows share `constraints.txt` via a job-level, absolute
`PIP_CONSTRAINT` path. It applies to every pip invocation, including installs
inside `make dev`, later notebook/tool installs, and the released `pyqlib`
package. A bound in this checkout's `pyproject.toml` alone does not constrain a
package installed from PyPI. Keep the compatibility bounds in both files in
sync; `tests/test_ci_configuration.py` checks this.

This is a set of known compatibility bounds, not a fully reproducible Python
lockfile. The workflows run `pip check` and always print installed versions for
future diagnosis. They retain all 25 OS/Python combinations per workflow and
disable matrix fail-fast so that one failure does not hide the others.

## Why these bounds exist

| Dependency | Constraint | Failure addressed |
| --- | --- | --- |
| MLflow | `<3.13` | Qlib uses the filesystem tracking backend, disabled by default in newer versions. |
| filelock | `>=3.16.0,<3.30` | Fork-safety checks conflict with DataQueue's producer thread and multiprocessing workers. |
| fastjsonschema | `<2.22` on Python 3.8/3.9 | Python 3.10 union annotations are evaluated on older interpreters. |
| Plotly | `<7` | Qlib still imports `figure_factory.create_distplot`. |
| lxml | `<6.1.3` | The Windows job falls back to a source distribution without libxml2 headers. |
| OSQP | `==1.0.5` on Windows/Python 3.8 | Native imports crash with 1.1.3 and also with 0.6.7.post3. |
| Black | `<26.1` | Use one compatible formatter for both Python and notebooks. |

OSQP 1.0.5 is the wheel used by the successful full Windows/Python 3.8 job in
[PR #2308](https://github.com/microsoft/qlib/actions/runs/30225522618/job/89854962742).
It is selected during dependency resolution, not installed as a downgrade after
an incompatible environment has already been built. All three workflows smoke
test native solver imports rather than excluding Windows/Python 3.8.

`MLFLOW_ALLOW_FILE_STORE=true` retains the original PR's explicit file-backend
opt-in. The upper bound also protects source installs outside CI. Migrating
Qlib's tracking backend is a separate change.

## grpcio wheels on older macOS ARM64 Python

The source jobs install TensorBoard through the RL extras, which brings in
grpcio. The published Python 3.8 grpcio 1.70.0 and Python 3.9 grpcio 1.80.0
macOS wheels have `universal2` filenames and binaries with both architectures,
but their internal `WHEEL` metadata declares only `x86_64`. Pip accepts the
filename during installation, then `pip check` rejects the internal tag on
ARM64. grpcio 1.78.0 has the same problem, so an arbitrary downgrade is not a
reliable fix.

Only the source and slow-source jobs on macOS ARM64 with Python 3.8/3.9 set
`PIP_NO_BINARY=grpcio` before `make dev`. This makes pip build the resolved
version from source with metadata for the local platform. The setting persists
for subsequent pip installs, and `GRPC_PYTHON_BUILD_EXT_COMPILER_JOBS=2` bounds
compilation parallelism. The workflows then import `grpc._cython.cygrpc` and
run the unchanged `pip check`. No package metadata is rewritten and no checks
or matrix entries are skipped. The PyPI workflow does not install the RL
extras and is unaffected.

These same source-build jobs also set `GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1` to use
the macOS SDK's zlib. In grpcio 1.70.0 (selected on Python 3.8), the bundled
`third_party/zlib/zutil.h` defines `fdopen(fd,mode)` as `NULL` when
`TARGET_OS_MAC` is defined. With the Xcode 16.4 SDK this corrupts the subsequent
`fdopen` declaration in `_stdio.h`, causing compilation to fail on macOS 15.
grpcio's supported system-zlib option removes the bundled zlib C sources from
the build and links `-lz` instead. It stays scoped to the existing macOS ARM64
Python 3.8/3.9 source-build workaround; other environments remain unchanged.

Remove this workaround only after validating both the internal wheel tags and
native imports on the affected ARM64 runners; a cross-platform resolver check
alone does not inspect the internal `WHEEL` metadata.

## Network and workflow retries

Dataset commands have at most three attempts, with a 15-minute timeout per
attempt. `--delete_old False` makes retries noninteractive in these fresh CI
directories, including when an earlier attempt already extracted one dataset.
Bash command blocks use `set -euo pipefail`, so an earlier failure cannot be
hidden by a later successful command. Exhausted retries still fail the job.

The configuration-driven workflow has at most three 30-minute attempts. This
is a bounded mitigation for the observed MLflow file-store error
`Metric 'IC' is malformed. No data found.`, not a fix for a suspected
read/write race. A deterministic error must still fail all attempts; do not
introduce `continue-on-error` or silently skip assertions. Fixing file-store
consistency should be handled separately, with a reproducer.

The MLflow client performance test initializes the tracking store before
timing repeated client construction, uses a monotonic performance clock, and
retains the original 10ms (Linux) / 20ms (other platforms) thresholds. Cold
plugin/store initialization is not a client-construction regression.

The PyPI workflow runs outside the checkout (`RUNNER_TEMP`) and passes the
example YAML by absolute path, so imports test the installed distribution.

## Title lint and maintenance

Checkout, Python setup, Node setup, and retry actions use Node 24-based releases
(`checkout@v6`, `setup-python@v6`, `setup-node@v6`, `retry@v4`). The action
runtime is separate from the Python and Node versions installed for tests.

Node 22 satisfies commitlint's Node >=22.12 requirement. The root `package.json`
and `package-lock.json` pin the title tooling and its transitive dependencies;
CI uses `npm ci --ignore-scripts` and `npx --no-install`.

When updating a bound, update the package metadata and constraints together,
run the offline policy checks, and verify the full matrix, including PyPI:

```sh
python -m unittest discover -s tests -p test_ci_configuration.py
npm ci --ignore-scripts --no-audit --no-fund
printf '%s\n' 'ci: update compatibility baseline' | npx --no-install commitlint --config .commitlintrc.js
```

For local pip installs, set `PIP_CONSTRAINT` to the absolute path of this
directory's `constraints.txt`. Do not infer Windows/macOS runtime correctness
from a Linux resolver check. Remove individual bounds only after reproducing
and verifying the upstream fix on the affected environment.

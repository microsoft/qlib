# CI compatibility baseline

The three Python workflows share `constraints.txt` via a job-level, absolute
`PIP_CONSTRAINT` path. It applies to every pip invocation, including installs
inside `make ci-install`, later notebook/tool installs, and the released `pyqlib`
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

## Install only each test job's dependencies

CI calls `make ci-install` with an explicit `CI_EXTRAS` profile. The existing
`make dev` target still installs the complete developer environment.

| Source job | Extras |
| --- | --- |
| Regular, Windows/macOS | `dev,test,analysis,lint` |
| Regular, Ubuntu 24.04 | `dev,test,analysis,lint,rl` |
| Regular, Ubuntu 22.04 | `dev,test,analysis,lint,rl,docs` |
| Slow, Windows/macOS | `dev,test,analysis` |
| Slow, Linux | `dev,test,analysis,rl` |

All source jobs additionally install `test-requirements.txt`: PyTorch is
required for the neural-network tests even without the RL extra. NumPy/SciPy
retain the compatibility bounds previously inherited from the RL/docs extras.
Ubuntu jobs select CPU PyTorch wheels. torchvision/torchaudio are not used by
the test suite and are no longer installed explicitly. No test job needs the
package-building extra; docs dependencies are installed only where `docs-gen`
runs. Lint checks stay in the existing regular source matrix.

`tests/conftest.py` already excludes RL tests on non-Linux platforms. Installing
RL extras there used to pull in `tianshou -> tensorboard -> grpcio`, requiring
expensive source builds to work around incorrect macOS wheel metadata and an
Xcode 16.4 bundled-zlib build failure. Those jobs no longer install this unused
dependency chain, so the grpcio source-build workaround has been removed.
Linux retains the RL extra and the same RL tests. The PyPI workflow is unchanged.

Explicit imports of PyTorch, Qlib's PyTorch model registry, GeneralPTNN, and
both dataset classes prevent missing dependencies from silently bypassing
neural-network tests. On macOS, prepare LightGBM and its `libomp` runtime before
these checks: importing the model package also eagerly imports LightGBM. The
preparation step verifies the LightGBM native import before checking the model
registry. A non-Linux footprint check rejects accidental
reintroduction of tianshou/TensorBoard/grpcio, followed by the unchanged
`pip check`. If a future test genuinely needs these dependencies on macOS,
update this policy and validate native wheel metadata/imports rather than
disabling the checks. This change does not reduce the matrix, alter test
selection, add caches, or reduce test data sizes.

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

Remote action references are pinned to full commit SHAs with version comments,
including the download and workflow retry steps. When updating an action,
verify its SHA against the upstream repository and update the version comment.
The offline policy checks reject mutable action references. Dependabot checks
GitHub Actions weekly and groups their version updates; its seven-day cooldown
filters newly published versions, rather than setting the interval between PRs.

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

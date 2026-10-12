.. _cross_platform_builds:

==============================
Cross-Platform Native Builds
==============================

This guide is the single reference for producing pyqlib's native artifacts on
Linux, macOS, and Windows across x86_64 and ARM64. It is written for two
audiences at once: the engineer who runs one command on a laptop, and the
release manager who needs the same artifacts to emerge untouched from CI.

Design and architecture
=======================

The fundamental constraint shaping this design is that pyqlib ships compiled
code. Two Cython extensions, ``qlib/data/_libs/rolling.pyx`` and
``expanding.pyx``, implement the rolling-window operators at the heart of
feature engineering, and they must be compiled to C++ and then to machine code
on every target. Everything else in the package is pure Python and therefore
portable, which means the entire cross-platform problem reduces to compiling
those two translation units correctly per OS and architecture, then packaging
the results into tagged wheels that pip can resolve unambiguously.

The architecture answers this with three layers that stay strictly separated.
The lowest layer is the build backend declaration in ``pyproject.toml``:
``setuptools`` with ``setuptools-scm``, ``cython``, and ``numpy>=1.24.0`` as
build requirements. Crucially, these are build-time-only inputs. The NumPy
headers are needed to compile against the NumPy C API, but the runtime NumPy
comes from the project's declared dependencies, never vendored into the wheel.
This separation is what keeps the artifacts small and avoids the classic
failure where a wheel pins a NumPy ABI its installer cannot satisfy. The
middle layer is ``setup.py``, which declares exactly two ``Extension`` objects
pointing at the two ``.pyx`` sources with ``language="c++"`` and the NumPy
include directory. There is deliberately no platform branching in ``setup.py``:
no ``if sys.platform`` blocks, no per-OS compiler flags. All platform
knowledge lives in the outer layer, so application logic and build
infrastructure never intermingle. The outer layer is ``scripts/build_wheels.py``
plus the ``build-wheels.yml`` CI workflow, both described below, which select
containers, deployment targets, and architectures while ``setup.py`` stays
oblivious.
A deliberate non-decision deserves emphasis. There is no cross-compilation in
this design: no MinGW targeting Windows from Linux, no ``zig cc`` tricks, no
single beefy build server with six toolchains. Cross-compiling CPython C
extensions is fragile precisely where it matters most, in the C++ standard
library linkage, the Apple-Silicon ABI boundary, and the MSVC runtime version.
Instead each artifact is compiled natively on a hosted runner of the target OS
family, with exactly one exception. Linux ARM64 builds run inside the standard
manylinux container under QEMU user-mode emulation on the ordinary Ubuntu
runner pool. QEMU makes an ARM64 container execute on x86_64 hardware at a
performance cost, but wheel builds are infrequent and the alternative, either
self-hosted ARM hardware or a niche hosted ARM pool, would violate the
constraint against fragile dedicated infrastructure. Every other combination,
Linux x86_64, macOS Intel and Apple Silicon, Windows x64 and ARM64, compiles
on first-party hosted runners for that platform.

Container and tooling strategy
==============================

Local development and CI share one entry point, ``scripts/build_wheels.py``,
which wraps `cibuildwheel <https://cibuildwheel.pypa.io/>`_, the community
standard for building wheels across platforms. A developer reproduces any
artifact with a single command after ``pip install cibuildwheel``::

    python scripts/build_wheels.py --platform linux --arch x86_64
    python scripts/build_wheels.py --platform macos --arch arm64
    python scripts/build_wheels.py --platform windows --arch AMD64
    python scripts/build_wheels.py --platform all   # every supported combination

The ``--platform auto`` default resolves the host OS, so bare
``make wheels`` builds exactly what the current machine can produce natively.
The script's ``MATRIX`` constant is the single source of truth for supported
combinations, six in total, and the ``CIBW_BUILD`` selector pins the CPython
range to ``cp38`` through ``cp312``, aligned with the classifiers in
``pyproject.toml``. PyPy, GraalPy, and musllinux variants are skipped
explicitly rather than by accident, and the skip list documents why.

Platform differences are handled through environment, never through code
forks. On Linux, ``CIBW_MANYLINUX_*_IMAGE=manylinux_2_28`` selects a glibc 2.28
baseline container, a modest step forward from the legacy manylinux2014 images
the old release job used, old enough for broad compatibility and new enough
for current compilers. The container provides the entire toolchain, so the
host needs nothing beyond Docker, and the ``CIBW_BEFORE_BUILD`` hook installs
only ``cython`` and ``numpy>=1.24.0`` inside the container before compiling.
On macOS there are no containers at all; the workflow runs directly on
``macos-13`` for Intel and ``macos-14`` for Apple Silicon, with
``MACOSX_DEPLOYMENT_TARGET=10.15`` ensuring Intel wheels install on supported
older systems while ARM64 wheels target the 11.0 ABI floor where Apple Silicon
began. Windows similarly builds on ``windows-2022`` and ``windows-11-arm``
with the MSVC toolchain the runners provide. File paths are normalized with
``pathlib`` throughout the scripts, and no build script ever hardcodes a
drive letter, a ``/usr`` prefix, or an OS-specific shell invocation beyond
what cibuildwheel itself abstracts.
Dependency caching is what keeps this fast enough for everyday use. The
workflow enables setup-python's built-in ``cache: pip`` keyed on
``pyproject.toml``, so the host-side downloads, cibuildwheel itself plus
twine and build, resolve once and reuse. Inside the Linux containers,
cibuildwheel's image layers are content-addressed and reused across runs, and
the only compilation units in the project, the two Cython extensions, rebuild
only when their ``.pyx`` sources or the pinned build requirements change. The
elevation workflow's own gates are untouched by this work; packaging sections
in ``pyproject.toml`` gained no new dependency bounds, so the
``test_ci_configuration.py`` constraint-equality assertions continue to pass
verbatim.

CI/CD integration
=================

The ``.github/workflows/build-wheels.yml`` workflow is the CI counterpart of
the local script, and it follows the repository's established conventions:
every remote action pinned to a full 40-character commit SHA with its version
tag in a trailing comment, concurrency cancellation per ref, and the shared
``PIP_CONSTRAINT`` plus ``MLFLOW_ALLOW_FILE_STORE`` environment the other
workflows use. It triggers on three events. A manual ``workflow_dispatch``
with an ``upload_to_pypi`` flag covers ad-hoc release builds. Tags matching
``v*`` drive real releases. And pull requests touching the native sources,
``setup.py``, ``pyproject.toml``, or the build scripts themselves get
validation builds, so a Cython regression can never merge silently.

The matrix fans out to six jobs, one per supported combination, each running
the same four steps: check out the tree, optionally start QEMU for the ARM64
Linux leg, set up the host Python with pip caching, then build, twine-check,
and statically audit. There is deliberately no ``upload-artifact`` round trip
and no separate publish job. Each matrix job publishes its own wheels directly
to TestPyPI on release events, gated by the same tag-or-flag condition, while
pull-request builds stop after validation. This removes an entire class of
failure, expired or mismatched intermediate artifacts, at the cost of one
repeated twine invocation per job, which is negligible next to compilation
time. The legacy ``release.yml`` manylinux-only path is left in place for
continuity; the new workflow supersedes it for multi-architecture releases.
Validation and testing of binaries
==================================

Validation happens in two stages with complementary strengths. Inside every
build, cibuildwheel's ``CIBW_TEST_COMMAND`` executes on the target runner
against the freshly built wheel: it imports NumPy, imports the compiled
``Rolling`` class from ``qlib.data._libs.rolling``, and prints the machine
architecture, proving the extension loads and executes on the silicon it was
built for. This is the dynamic gate, and it runs where emulation or hardware
differences would actually manifest.

The static gate is ``scripts/validate_wheels.py``, runnable anywhere with no
dependencies beyond the standard library, and it audits four properties of
every ``*.whl`` in ``dist/``. First, the filename must parse and its platform
tag must meet the supported baseline for its family, manylinux 2.28 or newer,
the macOS deployment floors, or the Windows tags. Second, the archive must
contain both compiled extensions with the correct native suffix for its OS,
``.so`` on Linux and macOS, ``.pyd`` on Windows. Third, there must be no
cross-family leakage, a Linux ``.so`` inside a ``win_`` wheel or vice versa,
which would indicate a misconfigured build container. Fourth, the
``dist-info/METADATA`` must declare the ``pyqlib`` name, guarding against
misbuilt source distributions masquerading as wheels. Any violation fails the
job with a named error, and ``twine check`` runs alongside to catch malformed
long descriptions before upload.

Reproducing the whole pipeline locally
======================================

A new team member needs exactly four commands to go from checkout to validated
wheels on their own machine::

    cp .env.example .env
    pip install cibuildwheel twine build      # or: make wheels-deps
    make wheels                                # build + twine check + static audit
    make sdist                                 # optional platform-independent sdist

No secrets are involved, no dataset download is required, and the
platform-specific knowledge lives entirely in the two scripts and one
workflow file. The application source never branches on the host OS for
build purposes, the dependency bounds never change per platform, and the
six release artifacts emerge from one trigger with no manual steps between
compilation and a clean, installable, pre-validated wheel set.

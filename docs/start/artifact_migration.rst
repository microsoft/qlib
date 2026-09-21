.. _artifact_loading_migration:

=============================================
Migration: explicit trust for saved artifacts
=============================================

.. important::

    **Unreleased backward-incompatible safety-default change.** This guide applies
    to new source builds containing the artifact-loading changes. Merging into
    ``main`` affects source installs before a new ``pyqlib`` release is published
    on PyPI; it does not change an already-installed tagged release. Until the
    first tagged release containing these changes, treat them as unreleased.
    That release's versioned upgrade notes should link to this guide.

What changes when upgrading
===========================

The built-in MLflow recorder now uses a restricted unpickler by default. Fresh
training and prediction using in-memory objects need no opt-in. Supported data-only
reads, including predictions, labels and numerical reports, also remain available
without consent.

Saved models, datasets, meta-models and tasks containing executable objects such as
reweighters require explicit ``trusted=True`` when reloaded. Backtest artifacts
containing ``Position`` instances or indicator objects require the same decision;
they are not data-only reports. A ``.pkl`` suffix, a familiar artifact name or a
successful run does not establish safety.

Before updating an existing workflow:

1. Identify which loads read data and which restore executable objects. Leave
   supported data-only reads restricted.
2. Verify both the artifact producer and storage write permissions before setting
   ``trusted=True`` at each relevant workflow entry point (CLI: ``--trusted=True``).
3. Review separately saved workflow components, local caches and custom loaders
   using the sections below. Update HIST mappings and high-frequency cache paths
   where applicable.
4. Test representative artifacts in a compatible environment before deployment.

Ordinary trusted model/dataset artifacts do not need deletion, conversion or a
full retrain merely to adopt explicit consent. Example ``main``/``first_run``
methods can reset experiments and task pools; do not use them as migration
commands. Preserve existing results and use dedicated stores when trying examples.

Single-object loads and trust boundaries
========================================

Keep data-only reads in the default mode:

.. code-block:: python

    from qlib.workflow import R

    rec = R.get_recorder()  # Select the intended run in your experiment.
    predictions = rec.load_object("pred.pkl")
    labels = R.load_object("label.pkl")
    report = rec.load_object("portfolio_analysis/report_normal_1day.pkl")

After verifying the writer and store, explicitly authorize executable objects:

.. code-block:: python

    model = rec.load_object("params.pkl", trusted=True)
    dataset = R.load_object("dataset", trusted=True)
    positions = rec.load_object(
        "portfolio_analysis/positions_normal_1day.pkl", trusted=True
    )

``Recorder.load_object`` and ``R.load_object`` expose keyword-only
``trusted=False``. All public trust options use this single name and require an
actual Python ``bool``, not strings, integers or other truthy values. The example
CLIs accept ``--trusted=True`` as boolean consent. Unrestricted artifact loads
emit an unsafe-loading warning and may execute code with the loading process's
permissions. The flag does not validate, sanitize or authenticate the contents.

``MLflowRecorder.load_object(name, unpickler=None, *, trusted=False)`` also accepts
a custom unpickler. That is trusted code, emits an unsafe-loading warning, and is
not a sandbox or a way to make unknown artifacts safe. A custom ``unpickler`` and
``trusted=True`` are mutually exclusive and raise ``ValueError`` together; choose
one deliberate loading policy, not both.

Verify **both source and store**: who produced the artifact, how it reached the
run, and who can replace it. Restrict write access to MLflow artifact directories,
shared filesystems and remote object stores. Creating a run yourself is not
enough if another user or job can overwrite its artifacts. Prefer dedicated,
access-controlled storage.

There is no automatic fallback to ordinary pickle. Do not catch a restricted-load
failure and retry with ``trusted=True``. A refusal may indicate an unsupported
representation, a missing dependency or executable content; it does not establish
trust. Inspect the reported type and provenance. Do not expand the global class
allowlist simply to suppress a model-loading error.

Workflow-level entry points
===========================

These public entry points use default-off ``trusted=False`` so callers need not
patch internal recorder loads:

* ``RMDLoader``, ``DSBasedUpdater``, ``PredUpdater`` and ``LabelUpdater``:
  model/dataset loads needed for updating.
* ``OnlineToolR`` and ``RollingStrategy``: executable task/model/dataset reads.
  A strategy forwards its setting to the online tool it creates, then through the
  updater to the loader.
* ``DelayTrainerR`` and ``DelayTrainerRM``: saved task reads when finishing delayed
  training. Constructor consent reaches ``end_train`` and the ``DelayTrainerRM``
  worker. Direct callers can use ``end_task_train(..., trusted=True)`` or pass an
  explicit ``trusted`` override to ``end_train``/``worker`` for that call.
* ``DDGDA``: recorder-backed meta-model/task reads and the local handler and
  ``InternalData`` caches needed by that workflow, described below.
* ``MetaDatasetDS`` and ``InternalData.setup``: lower-level recorder task reads.
  Their recorder consent alone does not authorize unrelated local pickle caches.

For example, after checking this workflow's artifact sources:

.. code-block:: python

    from qlib.model.trainer import DelayTrainerR
    from qlib.workflow.online.manager import OnlineManager
    from qlib.workflow.online.strategy import RollingStrategy

    strategy = RollingStrategy(
        "my_strategy",
        task_template=task_template,
        rolling_gen=rolling_gen,
        trusted=True,
    )
    trainer = DelayTrainerR(trusted=True)
    manager = OnlineManager(strategy, trainer=trainer)

Ordinary ``TrainerR``/``TrainerRM`` and ``OnlineManager`` have no ``trusted``
constructor option. Configure each strategy, including ones added later, and
configure a caller-supplied delayed trainer separately. A workflow does not
silently change that trainer's policy.

Consent covers the necessary executable objects, not every artifact in a run.
Prediction, label and numerical-report reads in these workflows remain restricted
even when ``trusted=True``. See :ref:`online_serving` and the
`example commands <https://github.com/microsoft/qlib/blob/main/examples/README.md#recorder-artifact-trust>`_.

Restored managers and components
---------------------------------

Only restore a local serialized ``OnlineManager`` from an independently trusted
source. Its strategies, tools and delayed trainer retain their own saved settings;
previously released objects without a trust field default to ``False``.

After reviewing each source, explicitly reconfigure or recreate every affected
strategy, its ``strategy.tool``, and any delayed trainer. Changing only
``strategy.trusted`` does not update an already-created tool. A new example
constructor or CLI flag does not overwrite a manager subsequently restored from
disk. ``add_strategy`` applies the current flag to new strategies only. There is
no manager-wide permission or revocation.

Example: re-authorizing a restored manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Initialize Qlib with your existing market-data and tracking configuration first.
Replace the input filename and ``my_strategy`` below with the saved manager and
strategy you intend to resume. Review that strategy's experiment storage and,
separately, all task stores used by its delayed trainer before granting consent.
A tool or trainer can be shared: its policy changes for every reference to that
same instance, not only for the selected strategy.

.. warning::

    ``OnlineManager.load`` itself uses unrestricted pickle/dill and can execute
    code before any component settings are changed. Only restore an independently
    trusted manager file. Setting component flags later does not make this initial
    load safe.

.. code-block:: python
    :name: restored-manager-migration

    from pathlib import Path

    from qlib.model.trainer import DelayTrainerR, DelayTrainerRM
    from qlib.workflow.online.manager import OnlineManager
    from qlib.workflow.online.strategy import RollingStrategy
    from qlib.workflow.online.utils import OnlineToolR

    manager = OnlineManager.load("manager.pkl")
    matches = [s for s in manager.strategies if s.name_id == "my_strategy"]
    if len(matches) != 1:
        raise ValueError("Expected exactly one saved strategy named my_strategy")
    strategy = matches[0]
    if not isinstance(strategy, RollingStrategy) or not isinstance(strategy.tool, OnlineToolR):
        raise TypeError("Adapt this example to your custom strategy/tool's loading policy")

    strategy.trusted = True
    strategy.tool.trusted = True
    if isinstance(manager.trainer, (DelayTrainerR, DelayTrainerRM)):
        manager.trainer.trusted = True

    with Path("manager.migrated.pkl").open("xb") as stream:
        manager.get_backend().dump(manager, stream, protocol=4)

This changes the selected strategy and its existing tool, plus the delayed
trainer when present. Other strategy objects are not automatically opted in.
Ordinary ``TrainerR``/``TrainerRM`` need no trust setting, and custom trainers
require their own loading-policy review. Repeat
the strategy configuration only for other components whose sources you have
reviewed, before resuming routines that use them.

The new file retains the manager's saved history and selected settings; exclusive
creation (``"xb"``) refuses to overwrite an existing output. After reviewing it,
point your resume command at ``manager.migrated.pkl`` rather than loading the old
file again. This migration does not run ``first_train``, reset experiments, or
retrain models. Prediction and label reads remain restricted.

DDG-DA caches and exported tasks
================================

``DDGDA(..., trusted=True)`` (CLI: ``--trusted=True``) authorizes the necessary
recorder objects and DDG-DA's local handler/``InternalData`` pickle cache reads.
Check ``working_dir``, the configuration directory used for handler caches, any
explicit ``h_path``, and the MLflow store, including all write permissions.
Restricted cache loads refuse executable handlers and ``InternalData`` objects.
Explicit consent uses ordinary pickle with a warning; it neither authenticates
files nor relaxes the global restricted loader.

Generated tasks keep a lightweight handler-cache **configuration reference**,
including its path and selected ``trusted`` setting, instead of embedding all
market data. Code consuming ``task["dataset"]["kwargs"]["handler"]`` must not
assume it is a ``file://`` string. Treat exported tasks as executable
configurations: their selected cache consent persists when saved and reused,
independently of a newly created workflow's default setting.

An old exported task may need regeneration using its matching meta-model and
workflow configuration to obtain the current cache-loader reference and policy.
Reloading a recorder task containing a reweighter still requires recorder consent;
that flag alone does not globally permit local caches referenced by the task.
Do not delete experiments or retrain everything just to migrate normal trusted
artifacts. See the
`DDG-DA example <https://github.com/microsoft/qlib/blob/main/examples/benchmarks_dynamic/DDG-DA/README.md#recorder-artifacts-and-local-working-files>`_
for workflow commands and existing full-flow regression coverage.

HIST stock-index mapping
========================

The bundled mapping is now
``examples/benchmarks/HIST/qlib_csi300_stock_index.json``, preserving all **735
entries** and their concept-matrix row assignments. Update
``task.model.kwargs.stock_index`` in custom YAML from the old object ``.npy`` path
to this JSON file. The bundled workflow already uses JSON.

An old saved HIST model also retains its own ``stock_index`` path. After
independently trusting and restoring that model, update ``model.stock_index`` to
the corresponding JSON path before prediction or further fitting; changing YAML
alone does not update a restored instance. Preserve the matching ``stock2concept``
matrix. It remains a numeric, two-dimensional ``.npy`` file loaded without pickle,
not JSON, and must contain the unknown-stock row **733** (at least **734 rows**).
Every mapped index must also be within the matrix's row bounds.

For a known-trusted custom mapping, re-export from the original trusted metadata
or producer into a JSON object with instrument strings as keys and non-negative
integer row indices as values (not booleans, floats or strings). Preserve each
instrument's row assignment. Merely renaming a file does not convert it.
Object-pickled ``.npy`` mappings remain refused even with recorder/workflow
``trusted=True``. Do not load an unknown object file to convert it; recover or
regenerate trusted source metadata instead. See the
`HIST example <https://github.com/microsoft/qlib/blob/main/examples/benchmarks/HIST/README.md#stock-index-mapping-migration>`_.

High-frequency provider artifact paths
======================================

``HighFreqProvider`` confines artifact paths to ``artifact_root``, which defaults
to the current working directory at construction. Choose a dedicated,
access-controlled root and update feature, label and backtest configuration paths
accordingly. Relative paths resolve against this root, not an arbitrary later
working directory.

All configured and derived paths, including split, per-day and per-stock files,
must remain inside the root after canonical resolution (including symlinks and
``..``). Returned artifact paths are canonical absolute paths; callers should use
them rather than assuming the original relative spelling is preserved. A trust
flag does not bypass path containment. Do not use ``artifact_root="/"`` as a
workaround; move/reconfigure artifacts within the intended dedicated root.

Containment does not make cache contents safe: these dataset caches still contain
Python pickles and must be independently trusted. Re-create an old serialized
provider that lacks ``artifact_root`` using reviewed configuration and an explicit
root rather than relying on a missing-field fallback.

Supported data and version compatibility
========================================

The restricted loader supports common built-in containers, NumPy arrays/scalars
and pandas ``Series``/``DataFrame`` objects, including typical prediction/label
``MultiIndex`` layouts. Supported reconstruction cases include pickle protocols
4 and 5, NumPy masked arrays, pandas nullable integer/float/boolean and
Python-backed string arrays, categorical data, datetime/timedelta data, supported
timezone metadata (such as UTC and ``pytz``), period/interval data and sparse arrays.

Not every dtype or object is supported. Object-dtype cells, custom subclasses,
extension arrays and metadata can introduce executable classes. Arrow-backed
pandas data and ``zoneinfo.ZoneInfo``-backed timezones are not supported by default.
The representation depends on Python, NumPy and pandas versions and dtype
settings. Regenerate supported data in a trusted producer environment rather than
enabling unrestricted loading simply to read predictions or numerical reports.

Pickle's cross-version limitations still apply. Protocol support does not
guarantee compatibility across Python, NumPy, pandas or model-library versions,
and ``trusted=True`` does not fix missing or renamed classes. Preserve the
producing environment for legacy executable artifacts and test representative
loads before upgrading.

Example: converting Arrow-backed columns
-----------------------------------------

Prefer converting the trusted producer's in-memory data before saving it.
The sample below requires pandas 2.x and PyArrow; replace ``original`` with your
actual DataFrame and choose target types from its schema. Do not cast an entire
table to ``float64``: that can lose large integer values or change other columns.
These explicit targets preserve the sample's float width, integer values,
missing-value masks and Python-backed strings.

.. code-block:: python
    :name: arrow-artifact-migration

    import pandas as pd

    original = pd.DataFrame(
        {
            "score": pd.Series([0.1, None, -0.3], dtype="float32[pyarrow]"),
            "count": pd.Series([2**60 + 1, None, 2**60 + 3], dtype="int64[pyarrow]"),
            "enabled": pd.Series([True, None, False], dtype="bool[pyarrow]"),
            "instrument": pd.Series(["SH600000", None, "SZ000001"], dtype="string[pyarrow]"),
        }
    )
    compatible = original.astype(
        {
            "score": "Float32",
            "count": "Int64",
            "enabled": "boolean",
            "instrument": pd.StringDtype(storage="python"),
        }
    )
    pd.testing.assert_frame_equal(original.isna(), compatible.isna())
    for column in original.columns:
        assert original[column].dropna().tolist() == compatible[column].dropna().tolist()
    assert compatible.loc[0, "count"] == 2**60 + 1
    assert compatible.loc[2, "count"] == 2**60 + 3

This is not a generic converter for Arrow decimals, nested arrays, timestamps or
custom extension types. Review the index, ``attrs`` and object-valued cells too;
converting the listed columns does not necessarily remove every unsupported
object from a real artifact.

Example: preserving a ZoneInfo index's time semantics
------------------------------------------------------

This example requires Python 3.9+ and uses a Qlib-style ``datetime`` /
``instrument`` MultiIndex. It changes the timezone implementation to ``pytz``
while keeping the same named timezone. The sample crosses a daylight-saving
transition: two different instants have the same local ``01:30`` clock label.

.. code-block:: python
    :name: zoneinfo-artifact-migration

    from zoneinfo import ZoneInfo

    import pandas as pd
    import pytz

    times = pd.date_range("2023-11-05 04:30", periods=4, freq="h", tz="UTC")
    times = times.tz_convert(ZoneInfo("America/New_York"))
    original = pd.DataFrame(
        {"score": [0.1, None, 0.3, 0.4]},
        index=pd.MultiIndex.from_arrays(
            [times, ["DEMO"] * len(times)], names=["datetime", "instrument"]
        ),
    )
    compatible = original.tz_convert(pytz.timezone("America/New_York"), level="datetime")

    before = original.index.get_level_values("datetime")
    after = compatible.index.get_level_values("datetime")
    pd.testing.assert_index_equal(before.tz_convert("UTC"), after.tz_convert("UTC"))
    pd.testing.assert_index_equal(before.tz_localize(None), after.tz_localize(None))
    pd.testing.assert_index_equal(
        original.index.get_level_values("instrument"),
        compatible.index.get_level_values("instrument"),
    )
    pd.testing.assert_frame_equal(
        original.reset_index(drop=True), compatible.reset_index(drop=True)
    )

The comparisons check both absolute instants and local clock labels; timezone
removal is used only for the comparison, not on the saved result. Do not simply
strip timezones to bypass a loading error. Converting to UTC preserves instants
but can change local dates, so use it only if your trading-calendar conventions
allow that change. For timezone-aware columns, use ``Series.dt.tz_convert`` with
the intended timezone as well; changing an index does not convert its columns.

Example: checking and saving the converted artifact
----------------------------------------------------

After either conversion above, check the entire result with the restricted
loader before publishing it. The recorder example uses your initialized Qlib
configuration with the built-in MLflow backend and an access-controlled store.
It creates a new run and does not overwrite the original artifact.

.. code-block:: python
    :name: converted-artifact-roundtrip

    import pickle

    import pandas as pd

    from qlib.utils.pickle_utils import restricted_pickle_loads
    from qlib.workflow import R

    restored = restricted_pickle_loads(pickle.dumps(compatible, protocol=4))
    pd.testing.assert_frame_equal(compatible, restored)

    with R.start(experiment_name="artifact-migration"):
        destination = R.get_recorder()
        destination.save_objects(**{"data.migrated.pkl": compatible})
        restored = destination.load_object("data.migrated.pkl")
        pd.testing.assert_frame_equal(compatible, restored)
        print("Migrated data recorder:", destination.id)

For real predictions or labels, preserve their original columns, index layout
and business meaning rather than adding the sample columns above. This staging
run contains converted data only, not a complete model/Dataset/task bundle.
Keep the original run backed up and explicitly plan how to publish the verified
data under the ``pred.pkl`` / ``label.pkl`` names expected by your workflow; do
not replace its model recorder with this data-only run.

If only a legacy pickle remains, recover it only in an independently trusted
producer environment. A deliberately selected MLflow artifact can be read with
``source_rec.load_object("pred.pkl", trusted=True)`` after verifying its writer
and storage, but that unrestricted load can execute code. Never open an unknown
pickle merely to convert it, or automatically retry a restricted-load failure
with consent. Online workflow consent still does not authorize unrestricted
prediction/label reads.

Custom recorders, loaders and completion callbacks
==================================================

Custom recorders should implement ``load_object(self, name, *, trusted=False)``,
validate actual boolean consent and enforce restricted loading by default.
Unrestricted loading requires explicit ``trusted=True``. Never ignore the flag or
add an unsafe retry.

For compatibility, ``R.load_object(name)`` and ``R.load_object(name, trusted=False)``
call a legacy recorder's ``load_object(name)`` without a new keyword; explicit
``trusted=True`` is forwarded. This preserves default call signatures, **not a
custom backend's security**. A legacy backend using ordinary pickle must implement
the restricted default itself. Adapt its signature before using explicit consent
through ``R``; do not rely on this facade accommodation for direct backend calls.

An updater's default construction of a custom ``loader_cls`` likewise omits the
new keyword when consent is ``False``. Opted-in construction forwards
``trusted=True``; adapt the loader's constructor, for example
``__init__(self, rec, *, trusted=False)``, and enforce its policy on executable
artifact loads. Data-only reads must remain restricted.

Custom delayed-training completion callbacks should accept
``end_train_func(rec, experiment_name, *, trusted=False)`` (plus any existing
workflow arguments), validate the boolean, and pass consent only to necessary
executable task reads. Default delayed completion preserves legacy callback calls
without adding the keyword when no consent/override is requested. Opted-in trainers
and explicit per-call overrides forward ``trusted`` to the callback, including in
``DelayTrainerRM`` workers; accepting it without enforcing the policy is not enough.

Other executable inputs are independent
=======================================

.. warning::

    This is a scoped artifact policy, not an all-Qlib sandbox or global
    authorization. Apart from DDG-DA's explicitly covered caches, existing local
    pickle/model loaders, serialized manager files, handler caches and MongoDB
    task stores retain their own trust requirements. YAML/task configurations can
    select executable Python components and must also be trusted.

    ``trusted=False`` does not make those inputs safe, and ``trusted=True`` does
    not authenticate or globally authorize them. Only open executable inputs
    from independently verified sources and access-controlled storage. Never
    deserialize unknown files merely to convert or migrate them.

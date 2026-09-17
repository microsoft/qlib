.. _recorder:

====================================
Qlib Recorder: Experiment Management
====================================
.. currentmodule:: qlib

Introduction
============
``Qlib`` contains an experiment management system named ``QlibRecorder``, which is designed to help users handle experiment and analyse results in an efficient way.

There are three components of the system:

- `ExperimentManager`
    a class that manages experiments.

- `Experiment`
    a class of experiment, and each instance of it is responsible for a single experiment.

- `Recorder`
    a class of recorder, and each instance of it is responsible for a single run.

Here is a general view of the structure of the system:

.. code-block::

    ExperimentManager
        - Experiment 1
            - Recorder 1
            - Recorder 2
            - ...
        - Experiment 2
            - Recorder 1
            - Recorder 2
            - ...
        - ...

This experiment management system defines a set of interface and provided a concrete implementation ``MLflowExpManager``, which is based on the machine learning platform: ``MLFlow`` (`link <https://mlflow.org/>`_).

If users set the implementation of ``ExpManager`` to be ``MLflowExpManager``, they can use the command `mlflow ui` to visualize and check the experiment results. For more information, please refer to the related documents `here <https://www.mlflow.org/docs/latest/cli.html#mlflow-ui>`_.

Qlib Recorder
=============
``QlibRecorder`` provides a high level API for users to use the experiment management system. The interfaces are wrapped in the variable ``R`` in ``Qlib``, and users can directly use ``R`` to interact with the system. The following command shows how to import ``R`` in Python:

.. code-block:: Python

        from qlib.workflow import R

``QlibRecorder`` includes several common API for managing `experiments` and `recorders` within a workflow. For more available APIs, please refer to the following section about `Experiment Manager`, `Experiment` and `Recorder`.

Here are the available interfaces of ``QlibRecorder``:

.. autoclass:: qlib.workflow.__init__.QlibRecorder
    :members:

Experiment Manager
==================

The ``ExpManager`` module in ``Qlib`` is responsible for managing different experiments. Most of the APIs of ``ExpManager`` are similar to ``QlibRecorder``, and the most important API will be the ``get_exp`` method. User can directly refer to the documents above for some detailed information about how to use the ``get_exp`` method.

.. autoclass:: qlib.workflow.expm.ExpManager
    :members: get_exp, list_experiments
    :noindex:

For other interfaces such as `create_exp`, `delete_exp`, please refer to `Experiment Manager API <../reference/api.html#experiment-manager>`_.

Experiment
==========

The ``Experiment`` class is solely responsible for a single experiment, and it will handle any operations that are related to an experiment. Basic methods such as `start`, `end` an experiment are included. Besides, methods related to `recorders` are also available: such methods include `get_recorder` and `list_recorders`.

.. autoclass:: qlib.workflow.exp.Experiment
    :members: get_recorder, list_recorders
    :noindex:

For other interfaces such as `search_records`, `delete_recorder`, please refer to `Experiment API <../reference/api.html#experiment>`_.

``Qlib`` also provides a default ``Experiment``, which will be created and used under certain situations when users use the APIs such as `log_metrics` or `get_exp`. If the default ``Experiment`` is used, there will be related logged information when running ``Qlib``. Users are able to change the name of the default ``Experiment`` in the config file of ``Qlib`` or during ``Qlib``'s `initialization <../start/initialization.html#parameters>`_, which is set to be '`Experiment`'.

Recorder
========

The ``Recorder`` class is responsible for a single recorder. It will handle some detailed operations such as ``log_metrics``, ``log_params`` of a single run. It is designed to help user to easily track results and things being generated during a run.

Here are some important APIs that are not included in the ``QlibRecorder``:

.. autoclass:: qlib.workflow.recorder.Recorder
    :members: list_artifacts, list_metrics, list_params, list_tags
    :noindex:

For other interfaces such as `save_objects`, `load_object`, please refer to `Recorder API <../reference/api.html#recorder>`_.

.. _artifact_trust_migration:

Migration: explicit artifact trust
==================================

Recorder artifacts can contain either data or executable Python objects. Predictions,
labels and numerical reports normally need only data reconstruction. A fitted model,
dataset, meta-model or task containing Python classes can require arbitrary Python
code during unpickling. A ``.pkl`` suffix, an artifact name such as ``pred.pkl``, or a
run being marked successful does not establish that its contents are safe.

The built-in MLflow recorder loads artifacts with a restricted unpickler by default:

.. code-block:: python

    from qlib.workflow import R

    rec = R.get_recorder()  # select the intended run in your configured experiment
    predictions = rec.load_object("pred.pkl")
    labels = R.load_object("label.pkl")

The restricted loader accepts only explicitly supported reconstruction classes.
Unsupported objects are refused; there is no automatic fallback to unrestricted
pickle loading. When upgrading, keep data-only reads in this default mode.

Loading executable artifacts
----------------------------

For an executable artifact, make the trust decision at the entry point of the
workflow that owns the run:

.. code-block:: python

    # Only for a model produced by a trusted writer in a trusted artifact store.
    model = rec.load_object("params.pkl", trusted=True)
    dataset = R.load_object("dataset", trusted=True)

Both APIs expose a keyword-only ``trusted=False`` argument. Pass an actual boolean;
``trusted=True`` enables ordinary pickle loading and can execute code with the
permissions of the loading process. It does not validate, sanitize or authenticate
the artifact.

Before opting in, verify **both the writer and the store**: know who produced the
artifact, how it reached this run, and who can replace it. Restrict write access to
the MLflow artifact directory or object store, including shared filesystem and
remote storage permissions. A run you created is not sufficient evidence if other
users or jobs can overwrite its artifacts. Prefer a dedicated, access-controlled
store for your own workflows.

Do not catch a restricted-load failure and retry with ``trusted=True``. A refusal
can mean an unsupported data representation, a missing dependency, or an executable
object; it is not evidence that the artifact is trustworthy. Inspect the reported
type and provenance, then either regenerate supported data or deliberately opt in
at the workflow boundary. Do not expand the global class allowlist just to suppress
a model-loading error.

Workflow-level consent
----------------------

The following entry points provide a default-off ``trusted_artifacts=False``
option so callers do not need to patch internal ``load_object`` calls:

* ``RMDLoader``, ``DSBasedUpdater``, ``PredUpdater`` and ``LabelUpdater`` for
  loading the model or dataset needed for an update.
* ``OnlineToolR`` for online updates, and ``RollingStrategy`` for its task reads
  and the online tool it creates. The setting flows from strategy to tool to
  updater to loader.
* ``DelayTrainerR`` and ``DelayTrainerRM`` for resuming recorder-backed training.
  Constructor consent is forwarded to ``end_train`` and, for ``DelayTrainerRM``,
  the worker completing delayed tasks. ``end_task_train`` also accepts the option
  directly. A direct ``end_train(..., trusted_artifacts=True)`` call can override
  the constructor setting for that call.
* ``DDGDA`` for recorder-backed meta-model loading and its ``InternalData.setup``
  calls. For lower-level use, set ``trusted_artifacts`` on ``MetaDatasetDS`` or
  pass it to ``InternalData.setup`` explicitly.

For example, after verifying the artifacts and store used by this workflow:

.. code-block:: python

    from qlib.model.trainer import DelayTrainerR
    from qlib.workflow.online.manager import OnlineManager
    from qlib.workflow.online.strategy import RollingStrategy

    strategy = RollingStrategy(
        "my_strategy",
        task_template=task_template,
        rolling_gen=rolling_gen,
        trusted_artifacts=True,
    )
    trainer = DelayTrainerR(trusted_artifacts=True)
    manager = OnlineManager(strategy, trainer=trainer)

``OnlineManager`` has no global trust grant: configure each strategy independently,
including strategies added later, and configure a delayed trainer separately.
Ordinary ``TrainerR`` and ``TrainerRM`` constructors do not accept this option.
If a caller supplies a trainer instance to a workflow, the caller must configure
that trainer's consent; the workflow must not silently grant it.

Restored components saved before this option existed default to restricted loading;
components with saved flags retain their own settings. For a legacy saved
``OnlineManager``, explicitly reconfigure or recreate each strategy, its
``strategy.tool``, and any delayed trainer after reviewing their artifact sources.
Changing only the strategy's flag does not update an already-created tool. An
example constructor's ``trusted_artifacts`` flag does not override a manager
subsequently loaded from disk; there is no global grant.

These options authorize the necessary executable model, dataset and task artifact
reads, not all artifacts in a run. Prediction and label reads in these workflows
remain restricted even when consent is enabled. See :ref:`online_serving` and the
`example commands <https://github.com/microsoft/qlib/blob/main/examples/README.md#recorder-artifact-trust>`_.

.. warning::

    This is a recorder-artifact policy, not an all-Qlib sandbox. Existing local
    pickle APIs, serialized ``OnlineManager`` files, handler caches, task stores
    and DDG-DA ``working_dir`` files have their own trust requirements. Setting
    ``trusted_artifacts=False`` does not make those inputs safe, and setting it to
    ``True`` does not authenticate them. Only open such executable inputs from
    independently trusted sources. Task/YAML configurations can select executable
    Python components and must also be trusted; this flag does not sandbox them.

    In particular, DDG-DA's existing local ``restricted_pickle_load`` calls remain
    restricted. The recorder flag does not enable unsupported local cache objects,
    so such loads can still be refused even with ``trusted_artifacts=True``.

Supported data and compatibility
--------------------------------

The restricted path supports common built-in data containers, NumPy arrays and
scalars, and pandas ``Series``/``DataFrame`` objects, including typical prediction
and label ``MultiIndex`` layouts. Supported reconstruction cases include pickle
protocols 4 and 5, NumPy masked arrays, pandas nullable integer/float/boolean and
Python-backed string arrays, categorical data, datetime/timedelta data, supported
timezone metadata (such as UTC and ``pytz``), period and interval data, and sparse
arrays.

This is not a guarantee for every NumPy or pandas object. Object-dtype cells,
custom subclasses, extension arrays and metadata can introduce additional classes.
Arrow-backed pandas data and ``zoneinfo.ZoneInfo``-backed timezone representations
are not supported by default. Whether a particular representation is used depends
on Python, NumPy and pandas versions and dtype settings. Regenerate such data using
supported representations in a trusted producer environment rather than enabling
unrestricted loading just to read predictions.

Pickle's existing cross-version limitations still apply. Protocol support does not
guarantee compatibility between Python, NumPy, pandas or model-library versions,
nor does ``trusted=True`` fix missing or renamed classes. Preserve the producing
environment for legacy executable artifacts and test representative artifacts
before upgrading a workflow.

HIST stock-index mapping
------------------------

HIST's bundled stock-index mapping is now
``examples/benchmarks/HIST/qlib_csi300_stock_index.json``, containing the same
735 entries. Update custom YAML ``task.model.kwargs.stock_index`` paths from
``qlib_csi300_stock_index.npy`` to the JSON file. The bundled workflow already uses
JSON; the separate numeric ``stock2concept`` matrix remains a ``.npy`` file.

For a known-trusted custom mapping, re-export from the original trusted metadata or
producer into a JSON object with instrument strings as keys and non-negative integer
row indices as values. Preserve the correspondence with the ``stock2concept``
matrix and keep indices within its row bounds. Merely renaming an object ``.npy``
file does not convert it. Object-pickled ``.npy`` mappings are deliberately not
supported, and recorder consent does not re-enable them. See the
`HIST migration instructions <https://github.com/microsoft/qlib/blob/main/examples/benchmarks/HIST/README.md#stock-index-mapping-migration>`_.

Custom recorders and loaders
----------------------------

Custom ``Recorder`` implementations should adopt
``load_object(self, name, *, trusted=False)``, validate boolean consent, enforce
restricted loading by default and allow unrestricted deserialization only with
explicit ``trusted=True``. Never ignore the flag or add an unsafe retry path.

For compatibility, ``R.load_object(name)`` (and ``trusted=False``) delegates to a
legacy recorder's ``load_object(name)`` without adding a keyword. Explicit
``trusted=True`` is forwarded. This keeps legacy default calls usable, but **does
not certify a custom backend's security**: a legacy backend that uses unrestricted
pickle still needs to implement the restricted default. A backend without the
``trusted`` keyword must be adapted before callers can explicitly opt in through
``R``.

If an updater uses a custom ``loader_cls``, its default construction remains
legacy-compatible: the updater passes the new ``trusted_artifacts`` keyword only
when consent is ``True``. To support explicit consent, adapt that loader's
constructor to accept and enforce ``trusted_artifacts`` as well; accepting the
keyword without applying its policy is not sufficient.

Record Template
===============

The ``RecordTemp`` class is a class that enables generate experiment results such as IC and backtest in a certain format. We have provided three different `Record Template` class:

- ``SignalRecord``: This class generates the `prediction` results of the model.
- ``SigAnaRecord``: This class generates the `IC`, `ICIR`, `Rank IC` and `Rank ICIR` of the model.

Here is a simple example of what is done in ``SigAnaRecord``, which users can refer to if they want to calculate IC, Rank IC, Long-Short Return with their own prediction and label.

.. code-block:: Python

    from qlib.contrib.eva.alpha import calc_ic, calc_long_short_return

    ic, ric = calc_ic(pred.iloc[:, 0], label.iloc[:, 0])
    long_short_r, long_avg_r = calc_long_short_return(pred.iloc[:, 0], label.iloc[:, 0])

- ``PortAnaRecord``: This class generates the results of `backtest`. The detailed information about `backtest` as well as the available `strategy`, users can refer to `Strategy <../component/strategy.html>`_ and `Backtest <../component/backtest.html>`_.

Here is a simple example of what is done in ``PortAnaRecord``, which users can refer to if they want to do backtest based on their own prediction and label.

.. code-block:: Python

    from qlib.contrib.strategy.strategy import TopkDropoutStrategy
    from qlib.contrib.evaluate import (
        backtest as normal_backtest,
        risk_analysis,
    )

    # backtest
    STRATEGY_CONFIG = {
        "topk": 50,
        "n_drop": 5,
    }
    BACKTEST_CONFIG = {
        "limit_threshold": 0.095,
        "account": 100000000,
        "benchmark": BENCHMARK,
        "deal_price": "close",
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    }

    strategy = TopkDropoutStrategy(**STRATEGY_CONFIG)
    report_normal, positions_normal = normal_backtest(pred_score, strategy=strategy, **BACKTEST_CONFIG)

    # analysis
    analysis = dict()
    analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
    analysis["excess_return_with_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"] - report_normal["cost"])
    analysis_df = pd.concat(analysis)  # type: pd.DataFrame
    print(analysis_df)

For more information about the APIs, please refer to `Record Template API <../reference/api.html#module-qlib.workflow.record_temp>`_.



Known Limitations
=================
- The Python objects are saved based on pickle, which may results in issues when the environment dumping objects and loading objects are different.
- Restricted loading is intentionally not compatible with arbitrary Python objects.
  See :ref:`artifact_trust_migration` before changing trust settings.

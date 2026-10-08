.. _config_migration:

==================================
Configuration Execution: Migration
==================================

Scope and upgrade checklist
===========================

This guide describes the **unreleased source changes in PR #2340**, not a
guarantee about an installed PyPI version. A PR checkout can differ from
``main``; ``main`` can differ from the latest tagged release. Use documentation
and examples from the same revision as your installed code, and consult that
release's changelog before applying these changes to a tagged installation.

.. list-table:: Does my configuration need updating?
   :header-rows: 1
   :widths: 45 55

   * - Existing usage
     - Action
   * - Importable package paths, class objects, standard Alpha158/Alpha360
       expressions, built-in TRA and report names
     - No new permission required. Continue using these interfaces.
   * - A model, dataset, handler, or custom operator loaded from a ``.py`` path
     - Review the source and add top-level ``trusted: true`` to that
       component's configuration.
   * - Direct ``get_module_by_module_path("path.py")`` calls
     - Review the source and pass ``trusted=True`` on each importing call.
   * - Configurations copied from an earlier PR #2340 revision using
       ``trusted_module_roots`` or ``allowed_module_roots``
     - Remove the directory settings; declare trust separately for each file
       component/import as shown below.
   * - Expressions containing arbitrary Python or unregistered operators
     - Use the supported :ref:`expression_syntax` or register a custom operator.
   * - Custom built-in TRA backbones or model-performance graph names previously
       injected into module globals
     - Register them in ``MODEL_TYPES`` or ``GRAPH_FUNCTIONS`` before use.
   * - Older pickles containing classes loaded from local Python files
     - Independently verify the artifact and original source; pre-import the
       original module before using an appropriate artifact loader.

.. warning::

   Configuration files, imported code, dependencies, and the storage supplying
   them must be trusted. A ``trusted: true`` embedded in an untrusted YAML file
   is not evidence that it is safe. File import executes Python; this is neither
   a sandbox nor a general resource-exhaustion defense. Package imports remain
   available and can also execute code.

Per-import and per-component consent
====================================

For an existing, reviewed ``custom_modules/model.py`` defining ``MyModel``:

.. code-block:: python

   from qlib.utils import get_module_by_module_path, init_instance_by_config
   from qlib.utils.mod import get_callable_kwargs

   module = get_module_by_module_path("custom_modules/model.py", trusted=True)
   config = {
       "class": "MyModel",
       "module_path": "custom_modules/model.py",
       "trusted": True,
       "kwargs": {},
   }
   model = init_instance_by_config(config)
   model_class, constructor_kwargs = get_callable_kwargs(config)

The direct importer has the signature
``get_module_by_module_path(path, *, trusted=False)``: its consent argument is
keyword-only. Generic factories retain their existing signatures; only the
configuration's top-level ``trusted`` field authorizes their file import.

The equivalent workflow configuration fragment is:

.. code-block:: yaml

   qlib_init:
     provider_uri: "~/.qlib/qlib_data/cn_data"
     region: cn
   task:
     model:
       class: MyModel
       module_path: custom_modules/model.py
       trusted: true
       kwargs: {}
     # Retain your dataset and record configurations.

``trusted`` defaults to ``False`` when omitted. Only actual booleans are
accepted: Python ``True``/``False`` or YAML ``true``/``false``. Strings such as
``"true"``, integers such as ``1``, and ``null``/``None`` raise ``TypeError``;
they are not coerced. A file import without ``True`` raises ``PermissionError``
before executing that file.

``get_callable_kwargs``, ``init_instance_by_config``, and ``Operators.register``
consume the component configuration's **top-level** ``trusted`` field. It is
not passed to the constructor. Trust is not inherited from a parent component,
an earlier import, or ``qlib.init``. Every nested component with its own
``.py`` module path needs its own field. Preserve these fields when copying or
saving configurations and when sending them to workers; make the reviewed code
available there too. A saved configuration carries a request to execute code,
not proof of its safety.

Relative paths are still relative to the process's **current working directory**,
not the YAML file. Paths expand ``~`` and resolve symlinks. With ``trusted=True``,
any caller-approved existing ``.py`` file may be imported; there is no directory
containment check. Review the resolved target and protect its storage against
replacement.

The earlier directory-root design was removed to keep consent local to the
operation and reduce migration cost: copying a component configuration should
not require reconstructing process-wide directory authorization. There is no
``trusted_module_roots``, ``allowed_module_roots``, global file-trust switch,
``qlib.init(trusted=...)``, or ``qrun --trusted`` permission mechanism.

Scripts loading workflow YAML should still forward all initialization options:

.. code-block:: python

   import qlib

   # config is the parsed workflow YAML.
   qlib.init(**config["qlib_init"])

This forwards initialization settings, not file-import permission.

Other import entry points
-------------------------

Advanced callers of ``qlib.utils.register_wrapper`` can pass keyword-only
``trusted=True`` when resolving a class name from a reviewed file
``module_path``. Its default is ``False``; registering an existing class/object
or importing a package needs no file-import consent.

The legacy tuner declares consent alongside its specific module path:

.. code-block:: yaml

   experiment:
     tuner_class: MyTuner
     tuner_module_path: custom_modules/tuner.py
     trusted: true

``experiment.trusted`` authorizes only that ``tuner_module_path`` import,
not other experiment components or the workflow. Both entry points use the
same strict boolean and current-working-directory path rules; neither creates
a global permission.

Import trust is not constructor or artifact trust
-------------------------------------------------

``config["trusted"]`` controls importing the component's file.
``config["kwargs"]["trusted"]`` is the component's own constructor option, if
it has one; it does **not** authorize that import. For a component that also
requires artifact consent, these are independent decisions:

.. code-block:: yaml

   class: MyArtifactConsumer
   module_path: custom_modules/artifact_consumer.py
   trusted: true          # Execute this reviewed Python source.
   kwargs:
     trusted: true        # The component's separate constructor permission.

The separate, unreleased `PR #2339 <https://github.com/microsoft/qlib/pull/2339>`_
introduces ``trusted`` permissions on artifact-loading calls/components.
This guide does not imply that it is merged or released, and file-import
consent never authorizes artifact deserialization.

For compatibility, the generic factory has **no new import-authorization
keyword argument**: ``init_instance_by_config(config, trusted=True)`` remains
a constructor keyword argument, not permission to import a file. Existing
constructors accepting ``trusted`` continue to receive that option. Rare
callers forwarding that keyword through the factory should keep doing so;
the same distinction applies to ``trusted`` supplied through ``try_kwargs``.
These are constructor options, never file-import consent.

File-based custom operators
===========================

Save this reviewed operator as ``custom_modules/ops.py``:

.. code-block:: python

   from qlib.data.ops import Ref

   class MyRef(Ref):
       pass

Register it before parsing expressions:

.. code-block:: python

   import qlib
   from qlib.data import D

   qlib.init(
       provider_uri="~/.qlib/qlib_data/cn_data",
       custom_ops=[{
           "class": "MyRef",
           "module_path": "custom_modules/ops.py",
           "trusted": True,
       }],
   )
   features = D.features(["SH600000"], ["MyRef($close, 1)"],
                         start_time="2020-01-01", end_time="2020-01-10")

The same dictionary can be passed to ``qlib.data.ops.Operators.register`` in a
list after initialization. For workers, keep it in ``custom_ops`` in the
initialization configuration rather than relying on a parent process's
in-memory registration. Package-based and class-object registrations do not
need file-import consent.

``trusted=True`` never enables arbitrary Python in expression strings.
See :ref:`expression_syntax` for the supported grammar, limits, and migration
examples; operator implementations themselves remain trusted Python code.

Explicit extension mappings
===========================

TRA backbones
-------------

The built-in ``qlib.contrib.model.pytorch_tra`` supports ``RNN`` and
``Transformer``. Extend its mapping before constructing ``TRAModel``:

.. code-block:: python

   from qlib.contrib.model.pytorch_tra import MODEL_TYPES, RNN, TRAModel

   class MyRNN(RNN):
       pass

   MODEL_TYPES["MyRNN"] = MyRNN
   model = TRAModel(
       model_type="MyRNN",
       model_config={"input_size": 6},
       tra_config={"num_states": 1},
   )

A custom backbone must satisfy the existing backbone interface, including its
``output_size`` and tensor output. The legacy paper example in
``examples/benchmarks/TRA/src/model.py`` is a separate implementation:
its ``model_type: LSTM`` should not be renamed to ``RNN``.

Model-performance graphs
------------------------

Register a callable accepting ``pred_label`` and graph options, returning an
iterable of Plotly figures:

.. code-block:: python

   import pandas as pd
   import plotly.graph_objects as go
   from qlib.contrib.report.analysis_model.analysis_model_performance import (
       GRAPH_FUNCTIONS,
       model_performance_graph,
   )

   def score_histogram(pred_label, **kwargs):
       return (go.Figure(go.Histogram(x=pred_label["score"])),)

   GRAPH_FUNCTIONS["score_histogram"] = score_histogram
   index = pd.MultiIndex.from_product(
       [["SH600000"], pd.to_datetime(["2020-01-02", "2020-01-03"])],
       names=["instrument", "datetime"],
   )
   pred_label = pd.DataFrame(
       {"score": [0.1, 0.2], "label": [0.0, 0.1]}, index=index,
   )
   figures = model_performance_graph(
       pred_label,
       graph_names=["score_histogram"],
       show_notebook=False,
   )

Both mappings are process-local. Put custom classes/functions and registration
in reviewed importable code, and run registration before use in every worker
or process restoring a saved configuration. Saving a mapping key does not save
its registration. Unknown names raise ``ValueError``; setting ``trusted`` does
not bypass either mapping.

Recovering a trusted older file-model pickle
============================================

Only use this procedure for a model pickle you created and kept under your
control, after independently verifying the artifact **and** its original source.
Keep both on trusted storage; loading a pickle can execute arbitrary code.
This example uses Python's unrestricted pickle loader and requires no PR #2339
artifact API. ``trusted`` is the artifact-loading guard in this example's
helper, not a Qlib API; the module import has its own explicit consent:

.. code-block:: python

   import pickle
   from qlib.utils import get_module_by_module_path

   def recover_verified_model(module_path, artifact_path, *, trusted=False):
       if type(trusted) is not bool:
           raise TypeError("trusted must be a boolean")
       if not trusted:
           raise PermissionError("Verify the artifact before opting into pickle loading")
       get_module_by_module_path(module_path, trusted=True)
       with open(artifact_path, "rb") as stream:
           return pickle.load(stream)

   # Run only after reviewing BOTH the original source and the artifact.
   model = recover_verified_model(
       "custom_modules/model.py", "reviewed_artifacts/model.pkl",
       trusted=True,
   )

Use the **original module-path spelling**, and the original working directory
for relative paths, where needed to match the old artifact. The explicit import
registers legacy module-name aliases for supplied and resolved paths without
overwriting an unrelated existing module. Do not replace the original source
with an arbitrary same-named file.

Pre-importing restores module lookup; it does not make a pickle safe or satisfy
an artifact loader's separate trust checks. If using a Qlib loader instead,
follow that loader's requirements for your installed revision, including any
independent artifact consent it requires.

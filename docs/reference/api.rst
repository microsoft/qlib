.. _api:
================================
API Reference
================================



Here you can find all ``Qlib`` interfaces.


Data
====================

Provider
--------------------

.. automodule:: qlib.data.data
    :members:
		
Filter
--------------------

.. automodule:: qlib.data.filter
    :members:

Class
--------------------
.. automodule:: qlib.data.base
    :members:

Operator
--------------------
.. automodule:: qlib.data.ops
    :members:
	       
Cache
----------------
.. autoclass:: qlib.data.cache.MemCacheUnit
    :members:

.. autoclass:: qlib.data.cache.MemCache
    :members:

.. autoclass:: qlib.data.cache.ExpressionCache
    :members:

.. autoclass:: qlib.data.cache.DatasetCache
    :members:

.. autoclass:: qlib.data.cache.DiskExpressionCache
    :members:

.. autoclass:: qlib.data.cache.DiskDatasetCache
    :members:


Storage
-------------
.. autoclass:: qlib.data.storage.storage.BaseStorage
    :members:

.. autoclass:: qlib.data.storage.storage.CalendarStorage
    :members:

.. autoclass:: qlib.data.storage.storage.InstrumentStorage
    :members:

.. autoclass:: qlib.data.storage.storage.FeatureStorage
    :members:

.. autoclass:: qlib.data.storage.file_storage.FileStorageMixin
    :members:

.. autoclass:: qlib.data.storage.file_storage.FileCalendarStorage
    :members:

.. autoclass:: qlib.data.storage.file_storage.FileInstrumentStorage
    :members:

.. autoclass:: qlib.data.storage.file_storage.FileFeatureStorage
    :members:


Dataset
---------------

Dataset Class
~~~~~~~~~~~~~~~~~~~~
.. automodule:: qlib.data.dataset.__init__
    :members:

Data Loader
~~~~~~~~~~~~~~~~~~~~
.. automodule:: qlib.data.dataset.loader
    :members:

Data Handler
~~~~~~~~~~~~~~~~~~~~
.. automodule:: qlib.data.dataset.handler
    :members:

Processor
~~~~~~~~~~~~~~~~~~~~
.. automodule:: qlib.data.dataset.processor
    :members:


Contrib
====================

Model
--------------------
.. automodule:: qlib.model.base
    :members:

Strategy
-------------------

.. automodule:: qlib.contrib.strategy.strategy
    :members:

Evaluate
-----------------

.. automodule:: qlib.contrib.evaluate
    :members:
    

Report
-----------------

.. automodule:: qlib.contrib.report.analysis_position.report
    :members:



.. automodule:: qlib.contrib.report.analysis_position.score_ic
    :members:



.. automodule:: qlib.contrib.report.analysis_position.cumulative_return
    :members:



.. automodule:: qlib.contrib.report.analysis_position.risk_analysis
    :members:



.. automodule:: qlib.contrib.report.analysis_position.rank_label
    :members:



.. automodule:: qlib.contrib.report.analysis_model.analysis_model_performance
    :members:


Workflow
====================


Experiment Manager
--------------------
.. autoclass:: qlib.workflow.expm.ExpManager
    :members:

Experiment
--------------------
.. autoclass:: qlib.workflow.exp.Experiment
    :members:

Recorder
--------------------
.. autoclass:: qlib.workflow.recorder.Recorder
    :members:

Record Template
--------------------
.. automodule:: qlib.workflow.record_temp
    :members:

Task Management
====================


TaskGen
--------------------
.. automodule:: qlib.workflow.task.gen
    :members:

TaskManager
--------------------
.. automodule:: qlib.workflow.task.manage
    :members:

Trainer
--------------------
.. automodule:: qlib.model.trainer
    :members:

Collector
--------------------
.. automodule:: qlib.workflow.task.collect
    :members:

Group
--------------------
.. automodule:: qlib.model.ens.group
    :members:

Ensemble
--------------------
.. automodule:: qlib.model.ens.ensemble
    :members:

Utils
--------------------
.. automodule:: qlib.workflow.task.utils
    :members:


Online Serving
====================


Online Manager
--------------------
.. automodule:: qlib.workflow.online.manager
    :members:

Online Strategy
--------------------
.. automodule:: qlib.workflow.online.strategy
    :members:

Online Tool
--------------------
.. automodule:: qlib.workflow.online.utils
    :members:

RecordUpdater
--------------------
.. automodule:: qlib.workflow.online.update
    :members:


Utils
====================

Serializable
--------------------

.. automodule:: qlib.utils.serial.Serializable
    :members:


    
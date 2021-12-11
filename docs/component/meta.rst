.. _meta:

=================================
Meta Controller: Meta-Task & Meta-Dataset & Meta-Model
=================================
.. currentmodule:: qlib


Introduction
=============
TODO: Add introduction.

Meta Task
=============

A `Meta Task` instance is the basic element in the meta-learning framework. It saves the data that can be used for the `Meta Model`. Multiple `Meta Task` instances may share the same `Data Handler`, controlled by `Meta Dataset`. Users should use `prepare_task_data()` to obtain the data that can be directly fed into the `Meta Model`.

.. autoclass:: qlib.model.meta.task.MetaTask
    :members:

Meta Dataset
=============

`Meta Dataset` controls the meta-information generating process. It is on the duty of providing data for training the `Meta Model`. Users should use `prepare_tasks` to retrieve a list of `Meta Task` instances.

.. autoclass:: qlib.model.meta.dataset.MetaTaskDataset
    :members:

Meta Model
=============

General Meta Model
------------------
`Meta Model` instance is the part that controls the workflow. The usage of the `Meta Model` includes:
1. Users train their `Meta Model` with the `fit` function. 
2. The `Meta Model` instance guides the workflow by giving useful information via the `inference` function.

.. autoclass:: qlib.model.meta.model.MetaModel
    :members:

Meta Task Model
------------------
This type of meta-model may interact with task definitions directly. Then, the `Meta Task Model` is the class for them to inherit from. They guide the base tasks by modifying the base task definitions. The function `prepare_tasks` can be used to obtain the modified base task definitions.

.. autoclass:: qlib.model.meta.model.MetaTaskModel
    :members:

Meta Guide Model
------------------
This type of meta-model participates in the training process of the base forecasting model. The meta-model may guide the base forecasting models during their training to improve their performances.

.. autoclass:: qlib.model.meta.model.MetaGuideModel
    :members:

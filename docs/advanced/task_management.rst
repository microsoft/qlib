.. _task_managment:

=================================
Task Management
=================================
.. currentmodule:: qlib


Introduction
=============

The `Workflow <../component/introduction.html>`_ part introduces how to run research workflow in a loosely-coupled way. But it can only execute one ``task`` when you use ``qrun``.
To automatically generate and execute different tasks, ``Task Management`` provides a whole process including `Task Generating`_, `Task Storing`_, `Task Running`_ and `Task Collecting`_. 
With this module, users can run their ``task`` automatically at different periods, in different losses, or even by different models.

An example of the entire process is shown `here <https://github.com/microsoft/qlib/tree/main/examples/taskmanager/task_manager_rolling.py>`_.

Task Generating
===============
A ``task`` consists of `Model`, `Dataset`, `Record` or anything added by users. 
The specific task template(/definition/config) can be viewed in 
`Task Section <../component/workflow.html#task-section>`_.
Even though the task template is fixed, users can customize their ``TaskGen`` to generate different ``task`` by task template.

Here is the base class of ``TaskGen``:

.. autoclass:: qlib.workflow.task.gen.TaskGen
    :members:

``Qlib`` provider a class `RollingGen <https://github.com/microsoft/qlib/tree/main/qlib/workflow/task/gen.py>`_ to generate a list of ``task`` of the dataset in different date segments.
This class allows users to verify the effect of data from different periods on the model in one experiment.

Task Storing
===============
To achieve higher efficiency and the possibility of cluster operation, ``Task Manager`` will store all tasks in `MongoDB <https://www.mongodb.com/>`_.
Users **MUST** finished the configuration of `MongoDB <https://www.mongodb.com/>`_ when using this module.

Users need to provide the URL and database name of ``task`` storing like this.

    .. code-block:: python

        from qlib.config import C
        C["mongo"] = {
            "task_url" : "mongodb://localhost:27017/", # your MongoDB url
            "task_db_name" : "rolling_db" # database name
        }

The CRUD methods of ``task`` can be found in TaskManager. 
More methods can be seen in the `Github <https://github.com/microsoft/qlib/tree/main/qlib/workflow/task/manage.py>`_.

.. autoclass:: qlib.workflow.task.manage.TaskManager
    :members:

Task Running
===============
After generating and storing those ``task``, it's time to run the ``task`` which are in the *WAITING* status.
``Qlib`` provides a method called ``run_task`` to run those ``task`` in task pool, however, users can also customize how tasks are executed.
An easy way to get the ``task_func`` is using ``qlib.model.trainer.task_train`` directly.
It will run the whole workflow defined by ``task``, which includes *Model*, *Dataset*, *Record*.

.. autofunction:: qlib.workflow.task.manage.run_task

Task Collecting
===============
To see the results of ``task`` after running or to update something, ``Qlib`` provides a ``TaskCollector`` to collect the tasks by filter condition (optional).
Here are some methods in this class.

.. autoclass:: qlib.workflow.task.collect.TaskCollector
    :members:

``Qlib`` provides a concrete `example <https://github.com/microsoft/qlib/tree/main/examples/taskmanager/task_manager_rolling_with_updating.py>`_, including a whole process of `Task Generating`_ (using `RollingGen <https://github.com/microsoft/qlib/tree/main/qlib/workflow/task/gen.py>`_), `Task Storing`_, `Task Running`_ and `Task Collecting`_.
Besides, the `example <https://github.com/microsoft/qlib/tree/main/examples/taskmanager/task_manager_rolling_with_updating.py>`_ uses a ``ModelUpdater`` inherited from ``TaskCollector``, which can update the inferences and retrain the model if it is out of date.
Actually, the model updating can be viewed as a subset of ``Online Serving``.
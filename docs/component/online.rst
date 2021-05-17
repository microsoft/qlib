.. _online:

=================================
Online Serving
=================================
.. currentmodule:: qlib


Introduction
=============

.. image:: ../_static/img/online_serving.png
    :align: center


In addition to backtesting, one way to test a model is effective is to make predictions in real market conditions or even do real trading based on those predictions.
``Online Serving`` is a set of modules for online models using the latest data,
which including `Online Manager <#Online Manager>`_, `Online Strategy <#Online Strategy>`_, `Online Tool <#Online Tool>`_, `Updater <#Updater>`_. 

`Here <https://github.com/microsoft/qlib/tree/main/examples/online_srv>`_ are several examples for reference, which demonstrate different features of ``Online Serving``.
If you have many models or `task` needs to be managed, please consider `Task Management <../advanced/task_management.html>`_.
The `examples <https://github.com/microsoft/qlib/tree/main/examples/online_srv>`_ are based on some components in `Task Management <../advanced/task_management.html>`_ such as ``TrainerRM`` or ``Collector``.

Online Manager
=============

.. automodule:: qlib.workflow.online.manager
    :members:

Online Strategy
=============

.. automodule:: qlib.workflow.online.strategy
    :members:

Online Tool
=============

.. automodule:: qlib.workflow.online.utils
    :members:

Updater
=============

.. automodule:: qlib.workflow.online.update
    :members:
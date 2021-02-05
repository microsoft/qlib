.. _serial:

=================================
Serialization
=================================
.. currentmodule:: qlib

Introduction
===================
``Qlib`` supports dumping the state of ``DataHandler``, ``DataSet``, ``Processor`` and ``Model``, etc. into a disk and reloading them. 

Serializable Class
========================

``Qlib`` provides a base class ``qlib.utils.serial.Serializable``, whose state can be dumped into or loaded from disk in `pickle` format. 
When users dump the state of a ``Serializable`` instance, the attributes of the instance whose name **does not** start with `_` will be saved on the disk.

Example
==========================
``Qlib``'s serializable class includes  ``DataHandler``, ``DataSet``, ``Processor`` and ``Model``, etc., which are subclass of  ``qlib.utils.serial.Serializable``. 
Specifically, ``qlib.data.dataset.DatasetH`` is one of them. Users can serialize ``DatasetH`` as follows.

.. code-block:: Python

    ##=============dump dataset=============
    dataset.to_pickle(path="dataset.pkl") # dataset is an instance of qlib.data.dataset.DatasetH

    ##=============reload dataset=============
    with open("dataset.pkl", "rb") as file_dataset:
        dataset = pickle.load(file_dataset)

.. note::
    Only state of ``DatasetH`` should be saved on the disk, such as some `mean` and `variance` used for data normalization, etc. 

    After reloading the ``DatasetH``, users need to reinitialize it. It means that users can reset some states of ``DatasetH`` or ``QlibDataHandler`` such as `instruments`, `start_time`, `end_time` and `segments`, etc.,  and generate new data according to the states (data is not state and should not be saved on the disk).

A more detailed example is in this `link <https://github.com/microsoft/qlib/tree/main/examples/highfreq>`_.


API
===================
Please refer to `Serializable API <../reference/api.html#module-qlib.utils.serial.Serializable>`_.

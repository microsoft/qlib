.. _server:

=============================
``Online`` & ``Offline`` mode
=============================
.. currentmodule:: qlib


Introduction
============

``Qlib`` supports ``Online`` mode and ``Offline`` mode. Only the ``Offline`` mode is introduced in this document.

The ``Online`` mode is designed to solve the following problems:

- Manage the data in a centralized way. Users don't have to manage data of different versions.
- Reduce the amount of cache to be generated.
- Make the data can be accessed in a remote way.

Qlib-Server
===========

``Qlib-Server`` is the assorted server system for ``Qlib``, which utilizes ``Qlib`` for basic calculations and provides extensive server system and cache mechanism. With QLibServer, the data provided for ``Qlib`` can be managed in a centralized manner. With ``Qlib-Server``, users can use ``Qlib`` in ``Online`` mode.



Reference
=========
If users are interested in ``Qlib-Server`` and ``Online`` mode, please refer to `Qlib-Server Project <https://github.com/microsoft/qlib-server>`_ and `Qlib-Server Document <https://qlib-server.readthedocs.io/en/latest/>`_.


Qlib FAQ
############

Qlib Frequently Asked Questions
================================
.. contents::
    :depth: 1
    :local:
    :backlinks: none

------


1. RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase...
------------------------------------------------------------------------------------------------------------------------------------

.. code-block:: console

    RuntimeError:
            An attempt has been made to start a new process before the
            current process has finished its bootstrapping phase.

            This probably means that you are not using fork to start your
            child processes and you have forgotten to use the proper idiom
            in the main module:

                if __name__ == '__main__':
                    freeze_support()
                    ...

            The "freeze_support()" line can be omitted if the program
            is not going to be frozen to produce an executable.

This is caused by the limitation of multiprocessing under windows OS. Please refer to `here <https://stackoverflow.com/a/24374798>`_ for more info.

**Solution**: To select a start method you use the ``D.features`` in the if __name__ == '__main__' clause of the main module. For example:

.. code-block:: python

    import qlib
    from qlib.data import D


    if __name__ == "__main__":
        qlib.init()
        instruments = ["SH600000"]
        fields = ["$close", "$change"]
        df = D.features(instruments, fields, start_time='2010-01-01', end_time='2012-12-31')
        print(df.head())



2. qlib.data.cache.QlibCacheException: It sees the key(...) of the redis lock has existed in your redis db now.
-----------------------------------------------------------------------------------------------------------------

It sees the key of the redis lock has existed in your redis db now. You can use the following command to clear your redis keys and rerun your commands

.. code-block:: console

    $ redis-cli
    > select 1
    > flushdb

If the issue is not resolved, use ``keys *`` to find if multiple keys exist. If so, try using ``flushall`` to clear all the keys.

.. note::

    ``qlib.config.redis_task_db`` defaults is ``1``, users can use ``qlib.init(redis_task_db=<other_db>)`` settings.


Also, feel free to post a new issue in our GitHub repository. We always check each issue carefully and try our best to solve them.

3. ModuleNotFoundError: No module named 'qlib.data._libs.rolling'
------------------------------------------------------------------------------------------------------------------------------------

.. code-block:: python

    #### Do not import qlib package in the repository directory in case of importing qlib from . without compiling #####
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    File "qlib/qlib/__init__.py", line 19, in init
        from .data.cache import H
    File "qlib/qlib/data/__init__.py", line 8, in <module>
        from .data import (
    File "qlib/qlib/data/data.py", line 20, in <module>
        from .cache import H
    File "qlib/qlib/data/cache.py", line 36, in <module>
        from .ops import Operators
    File "qlib/qlib/data/ops.py", line 19, in <module>
        from ._libs.rolling import rolling_slope, rolling_rsquare, rolling_resi
    ModuleNotFoundError: No module named 'qlib.data._libs.rolling'

- If the error occurs when importing ``qlib`` package with ``PyCharm`` IDE, users can execute the following command in the project root folder to compile Cython files and generate executable files:

    .. code-block:: bash

        python setup.py build_ext --inplace

- If the error occurs when importing ``qlib`` package with command ``python`` , users need to change the running directory to ensure that the script does not run in the project directory.


4. BadNamespaceError: / is not a connected namespace
------------------------------------------------------------------------------------------------------------------------------------

.. code-block:: python

      File "qlib_online.py", line 35, in <module>
        cal = D.calendar()
      File "e:\code\python\microsoft\qlib_latest\qlib\qlib\data\data.py", line 973, in calendar
        return Cal.calendar(start_time, end_time, freq, future=future)
      File "e:\code\python\microsoft\qlib_latest\qlib\qlib\data\data.py", line 798, in calendar
        self.conn.send_request(
      File "e:\code\python\microsoft\qlib_latest\qlib\qlib\data\client.py", line 101, in send_request
        self.sio.emit(request_type + "_request", request_content)
      File "G:\apps\miniconda\envs\qlib\lib\site-packages\python_socketio-5.3.0-py3.8.egg\socketio\client.py", line 369, in emit
        raise exceptions.BadNamespaceError(
      BadNamespaceError: / is not a connected namespace.

- The version of ``python-socketio`` in qlib needs to be the same as the version of ``python-socketio`` in qlib-server:

    .. code-block:: bash

        pip install -U python-socketio==<qlib-server python-socketio version>


5. TypeError: send() got an unexpected keyword argument 'binary'
------------------------------------------------------------------------------------------------------------------------------------

.. code-block:: python

      File "qlib_online.py", line 35, in <module>
        cal = D.calendar()
      File "e:\code\python\microsoft\qlib_latest\qlib\qlib\data\data.py", line 973, in calendar
        return Cal.calendar(start_time, end_time, freq, future=future)
      File "e:\code\python\microsoft\qlib_latest\qlib\qlib\data\data.py", line 798, in calendar
        self.conn.send_request(
      File "e:\code\python\microsoft\qlib_latest\qlib\qlib\data\client.py", line 101, in send_request
        self.sio.emit(request_type + "_request", request_content)
      File "G:\apps\miniconda\envs\qlib\lib\site-packages\socketio\client.py", line 263, in emit
        self._send_packet(packet.Packet(packet.EVENT, namespace=namespace,
      File "G:\apps\miniconda\envs\qlib\lib\site-packages\socketio\client.py", line 339, in _send_packet
        self.eio.send(ep, binary=binary)
      TypeError: send() got an unexpected keyword argument 'binary'


- The ``python-engineio`` version needs to be compatible with the ``python-socketio`` version, reference: https://github.com/miguelgrinberg/python-socketio#version-compatibility

    .. code-block:: bash

        pip install -U python-engineio==<compatible python-socketio version>
        # or
        pip install -U python-socketio==3.1.2 python-engineio==3.13.2


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


.. note::

    ``qlib.config.redis_task_db`` defaults is ``1``, users can use ``qlib.init(redis_task_db=<other_db>)`` settings.


Also, feel free to post a new issue in our GitHub repository. We always check each issue carefully and try our best to solve them.

.. _client:

Qlib Client-Server Framework
============================

.. currentmodule:: qlib

Introduction
------------
Client-Server is designed to solve following  problems

- Manage the data in a centralized way. Users don't have to manage data of different versions.
- Reduce the amount of cache to be generated.
- Make the data can be accessed in a remote way.

Therefore, we designed the client-server framework to solve these problems.
We will maintain a server and provide the data.

You have to initialize you qlib with specific config for using the client-server framework.
Here is a typical initialization process.

qlib ``init`` commonly used parameters; ``nfs-common`` must be installed on the server where the client is located, execute: ``sudo apt install nfs-common``:
    - ``provider_uri``: nfs-server path; the format is ``host: data_dir``, for example: ``172.23.233.89:/data2/gaochao/sync_qlib/qlib``. If using offline, it can be a local data directory
    - ``mount_path``: local data directory, ``provider_uri`` will be mounted to this directory
    - ``auto_mount``: whether to automatically mount ``provider_uri`` to ``mount_path`` during qlib ``init``; You can also mount it manually: sudo mount.nfs ``provider_uri`` ``mount_path``. If on PAI, it is recommended to set ``auto_mount=True``
    - ``flask_server``: data service host; if you are on the intranet, you can use the default host: 172.23.233.89
    - ``flask_port``: data service port


If running on 10.150.144.153 or 10.150.144.154 server, it's recommended to use the following code to ``init`` qlib:

.. code-block:: python

   >>> import qlib
   >>> qlib.init(auto_mount=False, mount_path='/data/csdesign/qlib')
   >>> from qlib.data import D
   >>> D.features(['SH600000'], ['$close'], start_time='20080101', end_time='20090101').head()
    [39336:MainThread](2019-05-28 21:35:42,800) INFO - Initialization - [__init__.py:16] - default_conf: client.
    [39336:MainThread](2019-05-28 21:35:42,801) INFO - Initialization - [__init__.py:54] - qlib successfully initialized based on client settings.
    [39336:MainThread](2019-05-28 21:35:42,801) INFO - Initialization - [__init__.py:56] - provider_uri=172.23.233.89:/data2/gaochao/sync_qlib/qlib
    [39336:Thread-68](2019-05-28 21:35:42,809) INFO - Client - [client.py:28] - Connect to server ws://172.23.233.89:9710
    [39336:Thread-72](2019-05-28 21:35:43,489) INFO - Client - [client.py:31] - Disconnect from server!
    Opening /data/csdesign/qlib/cache/d239a3b191daa9a5b1b19a59beb47b33 in read-only mode
    Out[5]:
                               $close
    instrument datetime
    SH600000   2008-01-02  119.079704
               2008-01-03  113.120125
               2008-01-04  117.878860
               2008-01-07  124.505539
               2008-01-08  125.395004


If running on PAI, it's recommended to use the following code to ``init`` qlib:

.. code-block:: python

   >>> import qlib
   >>> qlib.init(auto_mount=True, mount_path='/data/csdesign/qlib', provider_uri='172.23.233.89:/data2/gaochao/sync_qlib/qlib')
   >>> from qlib.data import D
   >>> D.features(['SH600000'], ['$close'], start_time='20080101', end_time='20090101').head()
    [39336:MainThread](2019-05-28 21:35:42,800) INFO - Initialization - [__init__.py:16] - default_conf: client.
    [39336:MainThread](2019-05-28 21:35:42,801) INFO - Initialization - [__init__.py:54] - qlib successfully initialized based on client settings.
    [39336:MainThread](2019-05-28 21:35:42,801) INFO - Initialization - [__init__.py:56] - provider_uri=172.23.233.89:/data2/gaochao/sync_qlib/qlib
    [39336:Thread-68](2019-05-28 21:35:42,809) INFO - Client - [client.py:28] - Connect to server ws://172.23.233.89:9710
    [39336:Thread-72](2019-05-28 21:35:43,489) INFO - Client - [client.py:31] - Disconnect from server!
    Opening /data/csdesign/qlib/cache/d239a3b191daa9a5b1b19a59beb47b33 in read-only mode
    Out[5]:
                               $close
    instrument datetime
    SH600000   2008-01-02  119.079704
               2008-01-03  113.120125
               2008-01-04  117.878860
               2008-01-07  124.505539
               2008-01-08  125.395004


If running on Windows, open **NFS** features and write correct **mount_path**, it's recommended to use the following code to ``init`` qlib:

1.windows System open NFS Features
    * Open ``Programs and Features``.
    * Click ``Turn Windows features on or off``.
    * Scroll down and check the option ``Services for NFS``, then click OK

    Reference address: https://graspingtech.com/mount-nfs-share-windows-10/
2.config correct mount_path
    * In windows, mount path must be not exist path and root path,
        * correct format path eg: `H`, `i`...
        * error format path eg: `C`, `C:/user/name`, `qlib_data`...

.. code-block:: python

   >>> import qlib
   >>> qlib.init(auto_mount=True, mount_path='H', provider_uri='172.23.233.89:/data2/gaochao/sync_qlib/qlib')
   >>> from qlib.data import D
   >>> D.features(['SH600000'], ['$close'], start_time='20080101', end_time='20090101').head()
    [39336:MainThread](2019-05-28 21:35:42,800) INFO - Initialization - [__init__.py:16] - default_conf: client.
    [39336:MainThread](2019-05-28 21:35:42,801) INFO - Initialization - [__init__.py:54] - qlib successfully initialized based on client settings.
    [39336:MainThread](2019-05-28 21:35:42,801) INFO - Initialization - [__init__.py:56] - provider_uri=172.23.233.89:/data2/gaochao/sync_qlib/qlib
    [39336:Thread-68](2019-05-28 21:35:42,809) INFO - Client - [client.py:28] - Connect to server ws://172.23.233.89:9710
    [39336:Thread-72](2019-05-28 21:35:43,489) INFO - Client - [client.py:31] - Disconnect from server!
    Opening /data/csdesign/qlib/cache/d239a3b191daa9a5b1b19a59beb47b33 in read-only mode
    Out[5]:
                               $close
    instrument datetime
    SH600000   2008-01-02  119.079704
               2008-01-03  113.120125
               2008-01-04  117.878860
               2008-01-07  124.505539
               2008-01-08  125.395004





The client will mount the data in `provider_uri` on `mount_path`. Then the server and client will communicate with flask and transporting data with this NFS.


If you have a local qlib data files and want to use the qlib data offline instead of online with client server framework.
It is also possible with  specific config.
You can created such a config. `client_config_local.yml`

.. code-block:: YAML

    provider_uri: /data/csdesign/qlib
    calendar_provider: 'LocalCalendarProvider'
    instrument_provider: 'LocalInstrumentProvider'
    feature_provider: 'LocalFeatureProvider'
    expression_provider: 'LocalExpressionProvider'
    dataset_provider: 'LocalDatasetProvider'
    provider: 'LocalProvider'
    dataset_cache: 'SimpleDatasetCache'
    local_cache_path: '~/.cache/qlib/'

`provider_uri` is the directory of your local data.

.. code-block:: python

   >>> import qlib
   >>> qlib.init_from_yaml_conf('client_config_local.yml')
   >>> from qlib.data import D
   >>> D.features(['SH600001'], ['$close'], start_time='20180101', end_time='20190101').head()
    21232:MainThread](2019-05-29 10:16:05,066) INFO - Initialization - [__init__.py:16] - default_conf: client.
    [21232:MainThread](2019-05-29 10:16:05,066) INFO - Initialization - [__init__.py:54] - qlib successfully initialized based on client settings.
    [21232:MainThread](2019-05-29 10:16:05,067) INFO - Initialization - [__init__.py:56] - provider_uri=/data/csdesign/qlib
    Out[9]:
                              $close
    instrument datetime
    SH600001   2008-01-02  21.082111
               2008-01-03  23.195362
               2008-01-04  23.874615
               2008-01-07  24.880930
               2008-01-08  24.277143

Limitations
-----------
1. The following API under the client-server module may not be as fast as the older off-line  API.
    - Cal.calendar
    - Inst.list_instruments
2. The rolling operation expression with parameter `0` can not be updated rightly under mechanism of the client-server framework.

API
***

The client is based on `python-socketio <https://python-socketio.readthedocs.io>`_ which is a framework that supports WebSocket client for Python language. The client can only propose requests and receive results, which do not include any calculating procedure.

Class
-----

.. automodule:: qlib.data.client

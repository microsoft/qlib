.. _initialization:
====================
Initialize Qlib
====================

.. currentmodule:: qlib


Initialize ``qlib`` Package
=========================

Please execute the following process to initialize ``qlib`` Package:

- Download and prepare the Data: execute the following command to download the stock data.
    .. code-block:: bash
    
        python scripts/get_data.py qlib_data_cn --target_dir ~/.qlib/qlib_data/cn_data

    Know more about how to use get_data.py, refer to `Raw Data  <../advanced/data.html#raw-data>`_.


- Run the initialization code: run the following code in python:

    .. code-block:: Python
<<<<<<< HEAD
        from qlib.config import REG_CN, REG_US
        mount_path = "~/.qlib/qlib_data/cn_data"  # target_dir
        qlib.init(mount_path=mount_path, region="REG_CN")

=======

        import qlib
        # region in [REG_CN, REG_US]
        from qlib.config import REG_CN
        mount_path = "~/.qlib/qlib_data/cn_data"  # target_dir
        qlib.init(mount_path=mount_path, region=REG_CN)
    
>>>>>>> c9205cac41923fe695edf8bd5728613d5c2f55c2


Parameters
===============================

In fact, in addition to'mount_path' and 'region', qlib.init has other parameters. The following are all the parameters of qlib.init:

- ``mount_path``: type: str. The local directory where the data loaded by 'get_data.py' is stored.
- ``region``:  type: str, optional parameter(default: `qlib.config.REG_CN`/'cn'>). If region == `qlib.config.REG_CN`, 'qlib' will be initialized in US stock mode. If region == `qlib.config.REG_US`, 'qlib' will be initialized in A-share mode.
    
    .. note:: 
        
        The value of'region' should be consistent with the data stored in'mount_path'. Currently,'scripts/get_data.py' only supports downloading A-share data. If users need to use the US stock mode, they need to prepare their own US stock data and store it in'mount_path'.
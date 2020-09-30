# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


__version__ = "0.5.1"

import os
import copy
import logging
import re
import subprocess
import platform
import yaml
from pathlib import Path

from .utils import can_use_cache


# init qlib
def init(default_conf="client", **kwargs):
    from .config import C, REG_CN, REG_US, QlibConfig
    from .data.data import register_all_wrappers
    from .log import get_module_logger, set_log_with_config
    from .data.cache import H

    C.reset()
    H.clear()

    _logging_config = C.logging_config
    if "logging_config" in kwargs:
        _logging_config = kwargs["logging_config"]

    # set global config
    if _logging_config:
        set_log_with_config(_logging_config)

    LOG = get_module_logger("Initialization", level=logging.INFO)
    LOG.info(f"default_conf: {default_conf}.")

    C.set_mode(default_conf)

    for k, v in kwargs.items():
        C[k] = v
        if k not in C:
            LOG.warning("Unrecognized config %s" % k)

    C.set_region(kwargs.get('region', C['region'] if 'region' in C else REG_CN ))
    C.resolve_path()

    if not (C["expression_cache"] is None and C["dataset_cache"] is None):
        # check redis
        if not can_use_cache():
            LOG.warning(
                f"redis connection failed(host={C['redis_host']} port={C['redis_port']}), cache will not be used!"
            )
            C["expression_cache"] = None
            C["dataset_cache"] = None

    # check path if server/local
    if C.get_uri_type() == QlibConfig.LOCAL_URI:
        if not os.path.exists(C["provider_uri"]):
            if C["auto_mount"]:
                LOG.error(
                    "Invalid provider uri: {}, please check if a valid provider uri has been set. This path does not exist.".format(
                        C["provider_uri"]
                    )
                )
            else:
                LOG.warning("auto_path is False, please make sure {} is mounted".format(C["mount_path"]))
    elif C.get_uri_type() == QlibConfig.NFS_URI:
        _mount_nfs_uri(C)
    else:
        raise NotImplementedError(f"This type of URI is not supported")

    LOG.info("qlib successfully initialized based on %s settings." % default_conf)
    register_all_wrappers()

    LOG.info(f"data_path={C.get_data_path()}")

    if "flask_server" in C:
        LOG.info(f"flask_server={C['flask_server']}, flask_port={C['flask_port']}")


def _mount_nfs_uri(C):
    from .log import get_module_logger
    LOG = get_module_logger("mount nfs", level=logging.INFO)

    # FIXME: the C["provider_uri"] is modified in this function
    # If it is not modified, we can pass only  provider_uri or mount_path instead of C
    mount_command = "sudo mount.nfs %s %s" % (C["provider_uri"], C["mount_path"])
    # If the provider uri looks like this 172.23.233.89//data/csdesign'
    # It will be a nfs path. The client provider will be used
    if not C["auto_mount"]:
        if not os.path.exists(C["mount_path"]):
            raise FileNotFoundError(
                "Invalid mount path: {}! Please mount manually: {} or Set init parameter `auto_mount=True`".format(
                    C["mount_path"], mount_command
                )
            )
    else:
        # Judging system type
        sys_type = platform.system()
        if "win" in sys_type.lower():
            # system: window
            exec_result = os.popen("mount -o anon %s %s" % (C["provider_uri"], C["mount_path"] + ":"))
            result = exec_result.read()
            if "85" in result:
                LOG.warning("already mounted or window mount path already exists")
            elif "53" in result:
                raise OSError("not find network path")
            elif "error" in result or "错误" in result:
                raise OSError("Invalid mount path")
            elif C["provider_uri"] in result:
                LOG.info("window success mount..")
            else:
                raise OSError(f"unknown error: {result}")

            # config mount path
            C["mount_path"] = C["mount_path"] + ":\\"
        else:
            # system: linux/Unix/Mac
            # check mount
            _remote_uri = C["provider_uri"]
            _remote_uri = _remote_uri[:-1] if _remote_uri.endswith("/") else _remote_uri
            _mount_path = C["mount_path"]
            _mount_path = _mount_path[:-1] if _mount_path.endswith("/") else _mount_path
            _check_level_num = 2
            _is_mount = False
            while _check_level_num:
                with subprocess.Popen(
                    'mount | grep "{}"'.format(_remote_uri),
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                ) as shell_r:
                    _command_log = shell_r.stdout.readlines()
                if len(_command_log) > 0:
                    for _c in _command_log:
                        _temp_mount = _c.decode("utf-8").split(" ")[2]
                        _temp_mount = _temp_mount[:-1] if _temp_mount.endswith("/") else _temp_mount
                        if _temp_mount == _mount_path:
                            _is_mount = True
                            break
                if _is_mount:
                    break
                _remote_uri = "/".join(_remote_uri.split("/")[:-1])
                _mount_path = "/".join(_mount_path.split("/")[:-1])
                _check_level_num -= 1

            if not _is_mount:
                try:
                    os.makedirs(C["mount_path"], exist_ok=True)
                except Exception:
                    raise OSError(
                        "Failed to create directory {}, please create {} manually!".format(
                            C["mount_path"], C["mount_path"]
                        )
                    )

                # check nfs-common
                command_res = os.popen("dpkg -l | grep nfs-common")
                command_res = command_res.readlines()
                if not command_res:
                    raise OSError(
                        "nfs-common is not found, please install it by execute: sudo apt install nfs-common"
                    )
                # manually mount
                command_status = os.system(mount_command)
                if command_status == 256:
                    raise OSError(
                        "mount {} on {} error! Needs SUDO! Please mount manually: {}".format(
                            C["provider_uri"], C["mount_path"], mount_command
                        )
                    )
                elif command_status == 32512:
                    # LOG.error("Command error")
                    raise OSError("mount {} on {} error! Command error".format(C["provider_uri"], C["mount_path"]))
                elif command_status == 0:
                    LOG.info("Mount finished")
            else:
                LOG.warning("{} on {} is already mounted".format(_remote_uri, _mount_path))


def init_from_yaml_conf(conf_path):
    """init_from_yaml_conf

    :param conf_path: A path to the qlib config in yml format
    """

    with open(conf_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    default_conf = config.pop("default_conf", "client")
    init(default_conf, **config)

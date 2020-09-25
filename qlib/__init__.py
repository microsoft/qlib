# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


__version__ = "0.5.0.dev"

import os
import copy
import logging
import re
import subprocess
import platform
from pathlib import Path

from .utils import can_use_cache


# init qlib
def init(default_conf="client", **kwargs):
    from .config import (
        C,
        _default_client_config,
        _default_server_config,
        _default_region_config,
        REG_CN,
    )
    from .data.data import register_all_wrappers
    from .log import get_module_logger, set_log_with_config

    _logging_config = C.logging_config
    if "logging_config" in kwargs:
        _logging_config = kwargs["logging_config"]

    # set global config
    if _logging_config:
        set_log_with_config(_logging_config)

    LOG = get_module_logger("Initialization", level=logging.INFO)
    LOG.info(f"default_conf: {default_conf}.")
    if default_conf == "server":
        base_config = copy.deepcopy(_default_server_config)
    elif default_conf == "client":
        base_config = copy.deepcopy(_default_client_config)
    else:
        raise ValueError("Unknown system type")
    if base_config:
        base_config.update(_default_region_config[kwargs.get("region", REG_CN)])
        for k, v in base_config.items():
            C[k] = v

    for k, v in kwargs.items():
        C[k] = v
        if k not in C:
            LOG.warning("Unrecognized config %s" % k)

    if default_conf == "client":
        C["mount_path"] = str(Path(C["mount_path"]).expanduser().resolve())
        if not (C["expression_cache"] is None and C["dataset_cache"] is None):
            # check redis
            if not can_use_cache():
                LOG.warning(
                    f"redis connection failed(host={C['redis_host']} port={C['redis_port']}), cache will not be used!"
                )
                C["expression_cache"] = None
                C["dataset_cache"] = None

    # check path if server/local
    if re.match("^[^/ ]+:.+", C["provider_uri"]) is None:
        C["provider_uri"] = str(Path(C["provider_uri"]).expanduser().resolve())
        if not os.path.exists(C["provider_uri"]):
            if C["auto_mount"]:
                LOG.error(
                    "Invalid provider uri: {}, please check if a valid provider uri has been set. This path does not exist.".format(
                        C["provider_uri"]
                    )
                )
            else:
                LOG.warning("auto_path is False, please make sure {} is mounted".format(C["mount_path"]))
    else:
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

    LOG.info("qlib successfully initialized based on %s settings." % default_conf)
    register_all_wrappers()
    try:
        if C["auto_mount"]:
            LOG.info(f"provider_uri={C['provider_uri']}")
        else:
            LOG.info(f"mount_path={C['mount_path']}")
    except KeyError:
        LOG.info(f"provider_uri={C['provider_uri']}")

    if "flask_server" in C:
        LOG.info(f"flask_server={C['flask_server']}, flask_port={C['flask_port']}")


def init_from_yaml_conf(conf_path):
    """init_from_yaml_conf

    :param conf_path: A path to the qlib config in yml format
    """
    import yaml

    with open(conf_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    default_conf = config.pop("default_conf", "client")
    init(default_conf, **config)

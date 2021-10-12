# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path

_version_path = Path(__file__).absolute().parent / "VERSION.txt"  # This file is copyed from setup.py
__version__ = _version_path.read_text(encoding="utf-8").strip()
__version__bak = __version__  # This version is backup for QlibConfig.reset_qlib_version
import os
from typing import Union
import yaml
import logging
import platform
import subprocess
from .log import get_module_logger


# init qlib
def init(default_conf="client", **kwargs):
    from .config import C
    from .data.cache import H

    # FIXME: this logger ignored the level in config
    logger = get_module_logger("Initialization", level=logging.INFO)

    skip_if_reg = kwargs.pop("skip_if_reg", False)
    if skip_if_reg and C.registered:
        # if we reinitialize Qlib during running an experiment `R.start`.
        # it will result in loss of the recorder
        logger.warning("Skip initialization because `skip_if_reg is True`")
        return

    H.clear()
    C.set(default_conf, **kwargs)

    # mount nfs
    for _freq, provider_uri in C.provider_uri.items():
        mount_path = C["mount_path"][_freq]
        # check path if server/local
        uri_type = C.dpm.get_uri_type(provider_uri)
        if uri_type == C.LOCAL_URI:
            if not Path(provider_uri).exists():
                if C["auto_mount"]:
                    logger.error(
                        f"Invalid provider uri: {provider_uri}, please check if a valid provider uri has been set. This path does not exist."
                    )
                else:
                    logger.warning(f"auto_path is False, please make sure {mount_path} is mounted")
        elif uri_type == C.NFS_URI:
            _mount_nfs_uri(provider_uri, mount_path, C["auto_mount"])
        else:
            raise NotImplementedError(f"This type of URI is not supported")

    C.register()

    if "flask_server" in C:
        logger.info(f"flask_server={C['flask_server']}, flask_port={C['flask_port']}")
    logger.info("qlib successfully initialized based on %s settings." % default_conf)
    data_path = {_freq: C.dpm.get_data_uri(_freq) for _freq in C.dpm.provider_uri.keys()}
    logger.info(f"data_path={data_path}")


def _mount_nfs_uri(provider_uri, mount_path, auto_mount: bool = False):

    LOG = get_module_logger("mount nfs", level=logging.INFO)
    if mount_path is None:
        raise ValueError(f"Invalid mount path: {mount_path}!")
    # FIXME: the C["provider_uri"] is modified in this function
    # If it is not modified, we can pass only  provider_uri or mount_path instead of C
    mount_command = "sudo mount.nfs %s %s" % (provider_uri, mount_path)
    # If the provider uri looks like this 172.23.233.89//data/csdesign'
    # It will be a nfs path. The client provider will be used
    if not auto_mount:
        if not Path(mount_path).exists():
            raise FileNotFoundError(
                f"Invalid mount path: {mount_path}! Please mount manually: {mount_command} or Set init parameter `auto_mount=True`"
            )
    else:
        # Judging system type
        sys_type = platform.system()
        if "win" in sys_type.lower():
            # system: window
            exec_result = os.popen("mount -o anon %s %s" % (provider_uri, mount_path + ":"))
            result = exec_result.read()
            if "85" in result:
                LOG.warning(f"{provider_uri} on Windows:{mount_path} is already mounted")
            elif "53" in result:
                raise OSError("not find network path")
            elif "error" in result or "错误" in result:
                raise OSError("Invalid mount path")
            elif provider_uri in result:
                LOG.info("window success mount..")
            else:
                raise OSError(f"unknown error: {result}")

        else:
            # system: linux/Unix/Mac
            # check mount
            _remote_uri = provider_uri[:-1] if provider_uri.endswith("/") else provider_uri
            _mount_path = mount_path[:-1] if mount_path.endswith("/") else mount_path
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
                    Path(mount_path).mkdir(parents=True, exist_ok=True)
                except Exception:
                    raise OSError(f"Failed to create directory {mount_path}, please create {mount_path} manually!")

                # check nfs-common
                command_res = os.popen("dpkg -l | grep nfs-common")
                command_res = command_res.readlines()
                if not command_res:
                    raise OSError("nfs-common is not found, please install it by execute: sudo apt install nfs-common")
                # manually mount
                command_status = os.system(mount_command)
                if command_status == 256:
                    raise OSError(
                        f"mount {provider_uri} on {mount_path} error! Needs SUDO! Please mount manually: {mount_command}"
                    )
                elif command_status == 32512:
                    # LOG.error("Command error")
                    raise OSError(f"mount {provider_uri} on {mount_path} error! Command error")
                elif command_status == 0:
                    LOG.info("Mount finished")
            else:
                LOG.warning(f"{_remote_uri} on {_mount_path} is already mounted")


def init_from_yaml_conf(conf_path, **kwargs):
    """init_from_yaml_conf

    :param conf_path: A path to the qlib config in yml format
    """

    with open(conf_path) as f:
        config = yaml.safe_load(f)
    config.update(kwargs)
    default_conf = config.pop("default_conf", "client")
    init(default_conf, **config)


def get_project_path(config_name="config.yaml", cur_path: Union[Path, str, None] = None) -> Path:
    """
    If users are building a project follow the following pattern.
    - Qlib is a sub folder in project path
    - There is a file named `config.yaml` in qlib.

    For example:
        If your project file system stucuture follows such a pattern

            <project_path>/
              - config.yaml
              - ...some folders...
                - qlib/

        This folder will return <project_path>

        NOTE: link is not supported here.


    This method is often used when
    - user want to use a relative config path instead of hard-coding qlib config path in code

    Raises
    ------
    FileNotFoundError:
        If project path is not found
    """
    if cur_path is None:
        cur_path = Path(__file__).absolute().resolve()
    cur_path = Path(cur_path)
    while True:
        if (cur_path / config_name).exists():
            return cur_path
        if cur_path == cur_path.parent:
            raise FileNotFoundError("We can't find the project path")
        cur_path = cur_path.parent


def auto_init(**kwargs):
    """
    This function will init qlib automatically with following priority
    - Find the project configuration and init qlib
        - The parsing process will be affected by the `conf_type` of the configuration file
    - Init qlib with default config
    - Skip initialization if already initialized

    :**kwargs: it may contain following parameters
                cur_path: the start path to find the project path

    Here are two examples of the configuration

    Example 1)
    If you want create a new project-specific config based on a shared configure, you can use  `conf_type: ref`

    .. code-block:: yaml

        conf_type: ref
        qlib_cfg: '<shared_yaml_config_path>'
        # following configs in `qlib_cfg_update` is project=specific
        qlib_cfg_update:
            exp_manager:
                class: "MLflowExpManager"
                module_path: "qlib.workflow.expm"
                kwargs:
                    uri: "file://<your mlflow experiment path>"
                    default_exp_name: "Experiment"

    Example 2)
    If you wan to create simple a stand alone config, you can use following config(a.k.a `conf_type: origin`)

    .. code-block:: python

        exp_manager:
            class: "MLflowExpManager"
            module_path: "qlib.workflow.expm"
            kwargs:
                uri: "file://<your mlflow experiment path>"
                default_exp_name: "Experiment"

    """
    kwargs["skip_if_reg"] = kwargs.get("skip_if_reg", True)

    try:
        pp = get_project_path(cur_path=kwargs.pop("cur_path", None))
    except FileNotFoundError:
        init(**kwargs)
    else:
        conf_pp = pp / "config.yaml"
        with conf_pp.open() as f:
            conf = yaml.safe_load(f)

        conf_type = conf.get("conf_type", "origin")
        if conf_type == "origin":
            # The type of config is just like original qlib config
            init_from_yaml_conf(conf_pp, **kwargs)
        elif conf_type == "ref":
            # This config type will be more convenient in following scenario
            # - There is a shared configure file and you don't want to edit it inplace.
            # - The shared configure may be updated later and you don't want to copy it.
            # - You have some customized config.
            qlib_conf_path = conf["qlib_cfg"]
            qlib_conf_update = conf.get("qlib_cfg_update")
            init_from_yaml_conf(qlib_conf_path, **qlib_conf_update, **kwargs)
        logger = get_module_logger("Initialization")
        logger.info(f"Auto load project config: {conf_pp}")

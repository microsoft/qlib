# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import os
import pickle
import re
import sys
import copy
import json
import yaml
import redis
import bisect
import shutil
import difflib
import inspect
import hashlib
import warnings
import datetime
import requests
import tempfile
import importlib
import contextlib
import collections
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union, Tuple, Any, Text, Optional, Callable
from types import ModuleType
from urllib.parse import urlparse
from .file import get_or_create_path, save_multiple_parts_file, unpack_archive_with_buffer, get_tmp_file_with_buffer
from ..config import C
from ..log import get_module_logger, set_log_with_config

log = get_module_logger("utils")


#################### Server ####################
def get_redis_connection():
    """get redis connection instance."""
    return redis.StrictRedis(host=C.redis_host, port=C.redis_port, db=C.redis_task_db)


#################### Data ####################
def read_bin(file_path: Union[str, Path], start_index, end_index):
    file_path = Path(file_path.expanduser().resolve())
    with file_path.open("rb") as f:
        # read start_index
        ref_start_index = int(np.frombuffer(f.read(4), dtype="<f")[0])
        si = max(ref_start_index, start_index)
        if si > end_index:
            return pd.Series(dtype=np.float32)
        # calculate offset
        f.seek(4 * (si - ref_start_index) + 4)
        # read nbytes
        count = end_index - si + 1
        data = np.frombuffer(f.read(4 * count), dtype="<f")
        series = pd.Series(data, index=pd.RangeIndex(si, si + len(data)))
    return series


def np_ffill(arr: np.array):
    """
    forward fill a 1D numpy array

    Parameters
    ----------
    arr : np.array
        Input numpy 1D array
    """
    mask = np.isnan(arr.astype(float))  # np.isnan only works on np.float
    # get fill index
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]


#################### Search ####################
def lower_bound(data, val, level=0):
    """multi fields list lower bound.

    for single field list use `bisect.bisect_left` instead
    """
    left = 0
    right = len(data)
    while left < right:
        mid = (left + right) // 2
        if val <= data[mid][level]:
            right = mid
        else:
            left = mid + 1
    return left


def upper_bound(data, val, level=0):
    """multi fields list upper bound.

    for single field list use `bisect.bisect_right` instead
    """
    left = 0
    right = len(data)
    while left < right:
        mid = (left + right) // 2
        if val >= data[mid][level]:
            left = mid + 1
        else:
            right = mid
    return left


#################### HTTP ####################
def requests_with_retry(url, retry=5, **kwargs):
    while retry > 0:
        retry -= 1
        try:
            res = requests.get(url, timeout=1, **kwargs)
            assert res.status_code in {200, 206}
            return res
        except AssertionError:
            continue
        except Exception as e:
            log.warning("exception encountered {}".format(e))
            continue
    raise Exception("ERROR: requests failed!")


#################### Parse ####################
def parse_config(config):
    # Check whether need parse, all object except str do not need to be parsed
    if not isinstance(config, str):
        return config
    # Check whether config is file
    if os.path.exists(config):
        with open(config, "r") as f:
            return yaml.safe_load(f)
    # Check whether the str can be parsed
    try:
        return yaml.safe_load(config)
    except BaseException:
        raise ValueError("cannot parse config!")


#################### Other ####################
def drop_nan_by_y_index(x, y, weight=None):
    # x, y, weight: DataFrame
    # Find index of rows which do not contain Nan in all columns from y.
    mask = ~y.isna().any(axis=1)
    # Get related rows from x, y, weight.
    x = x[mask]
    y = y[mask]
    if weight is not None:
        weight = weight[mask]
    return x, y, weight


def hash_args(*args):
    # json.dumps will keep the dict keys always sorted.
    string = json.dumps(args, sort_keys=True, default=str)  # frozenset
    return hashlib.md5(string.encode()).hexdigest()


def parse_field(field):
    # Following patterns will be matched:
    # - $close -> Feature("close")
    # - $close5 -> Feature("close5")
    # - $open+$close -> Feature("open")+Feature("close")
    # TODO: this maybe used in the feature if we want to support the computation of different frequency data
    # - $close@5min -> Feature("close", "5min")

    if not isinstance(field, str):
        field = str(field)
    for pattern, new in [(r"\$(\w+)", rf'Feature("\1")'), (r"(\w+\s*)\(", r"Operators.\1(")]:  # Features  # Operators
        field = re.sub(pattern, new, field)
    return field


def get_module_by_module_path(module_path: Union[str, ModuleType]):
    """Load module path

    :param module_path:
    :return:
    """
    if isinstance(module_path, ModuleType):
        module = module_path
    else:
        if module_path.endswith(".py"):
            module_name = re.sub("^[^a-zA-Z_]+", "", re.sub("[^0-9a-zA-Z_]", "", module_path[:-3].replace("/", "_")))
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            module_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_path)
    return module


def split_module_path(module_path: str) -> Tuple[str, str]:
    """

    Parameters
    ----------
    module_path : str
        e.g. "a.b.c.ClassName"

    Returns
    -------
    Tuple[str, str]
        e.g. ("a.b.c", "ClassName")
    """
    *m_path, cls = module_path.split(".")
    m_path = ".".join(m_path)
    return m_path, cls


def get_callable_kwargs(config: Union[dict, str], default_module: Union[str, ModuleType] = None) -> (type, dict):
    """
    extract class/func and kwargs from config info

    Parameters
    ----------
    config : [dict, str]
        similar to config
        please refer to the doc of init_instance_by_config

    default_module : Python module or str
        It should be a python module to load the class type
        This function will load class from the config['module_path'] first.
        If config['module_path'] doesn't exists, it will load the class from default_module.

    Returns
    -------
    (type, dict):
        the class/func object and it's arguments.
    """
    if isinstance(config, dict):
        key = "class" if "class" in config else "func"
        if isinstance(config[key], str):
            # 1) get module and class
            # - case 1): "a.b.c.ClassName"
            # - case 2): {"class": "ClassName", "module_path": "a.b.c"}
            m_path, cls = split_module_path(config[key])
            if m_path == "":
                m_path = config.get("module_path", default_module)
            module = get_module_by_module_path(m_path)

            # 2) get callable
            _callable = getattr(module, cls)  # may raise AttributeError
        else:
            _callable = config[key]  # the class type itself is passed in
        kwargs = config.get("kwargs", {})
    elif isinstance(config, str):
        # a.b.c.ClassName
        m_path, cls = split_module_path(config)
        module = get_module_by_module_path(default_module if m_path == "" else m_path)

        _callable = getattr(module, cls)
        kwargs = {}
    else:
        raise NotImplementedError(f"This type of input is not supported")
    return _callable, kwargs


get_cls_kwargs = get_callable_kwargs  # NOTE: this is for compatibility for the previous version


def init_instance_by_config(
    config: Union[str, dict, object],
    default_module=None,
    accept_types: Union[type, Tuple[type]] = (),
    try_kwargs: Dict = {},
    **kwargs,
) -> Any:
    """
    get initialized instance with config

    Parameters
    ----------
    config : Union[str, dict, object]
        dict example.
            case 1)
            {
                'class': 'ClassName',
                'kwargs': dict, #  It is optional. {} will be used if not given
                'model_path': path, # It is optional if module is given
            }
            case 2)
            {
                'class': <The class it self>,
                'kwargs': dict, #  It is optional. {} will be used if not given
            }
        str example.
            1) specify a pickle object
                - path like 'file:///<path to pickle file>/obj.pkl'
            2) specify a class name
                - "ClassName":  getattr(module, "ClassName")() will be used.
            3) specify module path with class name
                - "a.b.c.ClassName" getattr(<a.b.c.module>, "ClassName")() will be used.
        object example:
            instance of accept_types
    default_module : Python module
        Optional. It should be a python module.
        NOTE: the "module_path" will be override by `module` arguments

        This function will load class from the config['module_path'] first.
        If config['module_path'] doesn't exists, it will load the class from default_module.

    accept_types: Union[type, Tuple[type]]
        Optional. If the config is a instance of specific type, return the config directly.
        This will be passed into the second parameter of isinstance.

    try_kwargs: Dict
        Try to pass in kwargs in `try_kwargs` when initialized the instance
        If error occurred, it will fail back to initialization without try_kwargs.

    Returns
    -------
    object:
        An initialized object based on the config info
    """
    if isinstance(config, accept_types):
        return config

    if isinstance(config, str):
        # path like 'file:///<path to pickle file>/obj.pkl'
        pr = urlparse(config)
        if pr.scheme == "file":
            with open(os.path.join(pr.netloc, pr.path), "rb") as f:
                return pickle.load(f)

    klass, cls_kwargs = get_callable_kwargs(config, default_module=default_module)

    try:
        return klass(**cls_kwargs, **try_kwargs, **kwargs)
    except (TypeError,):
        # TypeError for handling errors like
        # 1: `XXX() got multiple values for keyword argument 'YYY'`
        # 2: `XXX() got an unexpected keyword argument 'YYY'
        return klass(**cls_kwargs, **kwargs)


@contextlib.contextmanager
def class_casting(obj: object, cls: type):
    """
    Python doesn't provide the downcasting mechanism.
    We use the trick here to downcast the class

    Parameters
    ----------
    obj : object
        the object to be cast
    cls : type
        the target class type
    """
    orig_cls = obj.__class__
    obj.__class__ = cls
    yield
    obj.__class__ = orig_cls


def compare_dict_value(src_data: dict, dst_data: dict):
    """Compare dict value

    :param src_data:
    :param dst_data:
    :return:
    """

    class DateEncoder(json.JSONEncoder):
        # FIXME: This class can only be accurate to the day. If it is a minute,
        # there may be a bug
        def default(self, o):
            if isinstance(o, (datetime.datetime, datetime.date)):
                return o.strftime("%Y-%m-%d %H:%M:%S")
            return json.JSONEncoder.default(self, o)

    src_data = json.dumps(src_data, indent=4, sort_keys=True, cls=DateEncoder)
    dst_data = json.dumps(dst_data, indent=4, sort_keys=True, cls=DateEncoder)
    diff = difflib.ndiff(src_data, dst_data)
    changes = [line for line in diff if line.startswith("+ ") or line.startswith("- ")]
    return changes


def remove_repeat_field(fields):
    """remove repeat field

    :param fields: list; features fields
    :return: list
    """
    fields = copy.deepcopy(fields)
    _fields = set(fields)
    return sorted(_fields, key=fields.index)


def remove_fields_space(fields: [list, str, tuple]):
    """remove fields space

    :param fields: features fields
    :return: list or str
    """
    if isinstance(fields, str):
        return fields.replace(" ", "")
    return [i.replace(" ", "") for i in fields if isinstance(i, str)]


def normalize_cache_fields(fields: [list, tuple]):
    """normalize cache fields

    :param fields: features fields
    :return: list
    """
    return sorted(remove_repeat_field(remove_fields_space(fields)))


def normalize_cache_instruments(instruments):
    """normalize cache instruments

    :return: list or dict
    """
    if isinstance(instruments, (list, tuple, pd.Index, np.ndarray)):
        instruments = sorted(list(instruments))
    else:
        # dict type stockpool
        if "market" in instruments:
            pass
        else:
            instruments = {k: sorted(v) for k, v in instruments.items()}
    return instruments


def is_tradable_date(cur_date):
    """judgy whether date is a tradable date
    ----------
    date : pandas.Timestamp
        current date
    """
    from ..data import D

    return str(cur_date.date()) == str(D.calendar(start_time=cur_date, future=True)[0].date())


def get_date_range(trading_date, left_shift=0, right_shift=0, future=False):
    """get trading date range by shift

    Parameters
    ----------
    trading_date: pd.Timestamp
    left_shift: int
    right_shift: int
    future: bool

    """

    from ..data import D

    start = get_date_by_shift(trading_date, left_shift, future=future)
    end = get_date_by_shift(trading_date, right_shift, future=future)

    calendar = D.calendar(start, end, future=future)
    return calendar


def get_date_by_shift(trading_date, shift, future=False, clip_shift=True, freq="day", align: Optional[str] = None):
    """get trading date with shift bias will cur_date
        e.g. : shift == 1,  return next trading date
               shift == -1, return previous trading date
    ----------
    trading_date : pandas.Timestamp
        current date
    shift : int
    clip_shift: bool
    align : Optional[str]
        When align is None, this function will raise ValueError if `trading_date` is not a trading date
        when align is "left"/"right", it will try to align to left/right nearest trading date before shifting when `trading_date` is not a trading date

    """
    from qlib.data import D

    cal = D.calendar(future=future, freq=freq)
    trading_date = pd.to_datetime(trading_date)
    if align is None:
        if trading_date not in list(cal):
            raise ValueError("{} is not trading day!".format(str(trading_date)))
        _index = bisect.bisect_left(cal, trading_date)
    elif align == "left":
        _index = bisect.bisect_right(cal, trading_date) - 1
    elif align == "right":
        _index = bisect.bisect_left(cal, trading_date)
    else:
        raise ValueError(f"align with value `{align}` is not supported")
    shift_index = _index + shift
    if shift_index < 0 or shift_index >= len(cal):
        if clip_shift:
            shift_index = np.clip(shift_index, 0, len(cal) - 1)
        else:
            raise IndexError(f"The shift_index({shift_index}) of the trading day ({trading_date}) is out of range")
    return cal[shift_index]


def get_next_trading_date(trading_date, future=False):
    """get next trading date
    ----------
    cur_date : pandas.Timestamp
        current date
    """
    return get_date_by_shift(trading_date, 1, future=future)


def get_pre_trading_date(trading_date, future=False):
    """get previous trading date
    ----------
    date : pandas.Timestamp
        current date
    """
    return get_date_by_shift(trading_date, -1, future=future)


def transform_end_date(end_date=None, freq="day"):
    """handle the end date with various format

    If end_date is -1, None, or end_date is greater than the maximum trading day, the last trading date is returned.
    Otherwise, returns the end_date

    ----------
    end_date: str
        end trading date
    date : pandas.Timestamp
        current date
    """
    from ..data import D

    last_date = D.calendar(freq=freq)[-1]
    if end_date is None or (str(end_date) == "-1") or (pd.Timestamp(last_date) < pd.Timestamp(end_date)):
        log.warning(
            "\nInfo: the end_date in the configuration file is {}, "
            "so the default last date {} is used.".format(end_date, last_date)
        )
        end_date = last_date
    return end_date


def get_date_in_file_name(file_name):
    """Get the date(YYYY-MM-DD) written in file name
    Parameter
            file_name : str
       :return
            date : str
                'YYYY-MM-DD'
    """
    pattern = "[0-9]{4}-[0-9]{2}-[0-9]{2}"
    date = re.search(pattern, str(file_name)).group()
    return date


def split_pred(pred, number=None, split_date=None):
    """split the score file into two part
    Parameter
    ---------
        pred : pd.DataFrame (index:<instrument, datetime>)
            A score file of stocks
        number: the number of dates for pred_left
        split_date: the last date of the pred_left
    Return
    -------
        pred_left : pd.DataFrame (index:<instrument, datetime>)
            The first part of original score file
        pred_right : pd.DataFrame (index:<instrument, datetime>)
            The second part of original score file
    """
    if number is None and split_date is None:
        raise ValueError("`number` and `split date` cannot both be None")
    dates = sorted(pred.index.get_level_values("datetime").unique())
    dates = list(map(pd.Timestamp, dates))
    if split_date is None:
        date_left_end = dates[number - 1]
        date_right_begin = dates[number]
        date_left_start = None
    else:
        split_date = pd.Timestamp(split_date)
        date_left_end = split_date
        date_right_begin = split_date + pd.Timedelta(days=1)
        if number is None:
            date_left_start = None
        else:
            end_idx = bisect.bisect_right(dates, split_date)
            date_left_start = dates[end_idx - number]
    pred_temp = pred.sort_index()
    pred_left = pred_temp.loc(axis=0)[:, date_left_start:date_left_end]
    pred_right = pred_temp.loc(axis=0)[:, date_right_begin:]
    return pred_left, pred_right


def time_to_slc_point(t: Union[None, str, pd.Timestamp]) -> Union[None, pd.Timestamp]:
    """
    Time slicing in Qlib or Pandas is a frequently-used action.
    However, user often input all kinds of data format to represent time.
    This function will help user to convert these inputs into a uniform format which is friendly to time slicing.

    Parameters
    ----------
    t : Union[None, str, pd.Timestamp]
        original time

    Returns
    -------
    Union[None, pd.Timestamp]:
    """
    if t is None:
        # None represents unbounded in Qlib or Pandas(e.g. df.loc[slice(None, "20210303")]).
        return t
    else:
        return pd.Timestamp(t)


def can_use_cache():
    res = True
    r = get_redis_connection()
    try:
        r.client()
    except redis.exceptions.ConnectionError:
        res = False
    finally:
        r.close()
    return res


def exists_qlib_data(qlib_dir):
    qlib_dir = Path(qlib_dir).expanduser()
    if not qlib_dir.exists():
        return False

    calendars_dir = qlib_dir.joinpath("calendars")
    instruments_dir = qlib_dir.joinpath("instruments")
    features_dir = qlib_dir.joinpath("features")
    # check dir
    for _dir in [calendars_dir, instruments_dir, features_dir]:
        if not (_dir.exists() and list(_dir.iterdir())):
            return False
    # check calendar bin
    for _calendar in calendars_dir.iterdir():

        if ("_future" not in _calendar.name) and (
            not list(features_dir.rglob(f"*.{_calendar.name.split('.')[0]}.bin"))
        ):
            return False

    # check instruments
    code_names = set(map(lambda x: x.name.lower(), features_dir.iterdir()))
    _instrument = instruments_dir.joinpath("all.txt")
    miss_code = set(pd.read_csv(_instrument, sep="\t", header=None).loc[:, 0].apply(str.lower)) - set(code_names)
    if miss_code and any(map(lambda x: "sht" not in x, miss_code)):
        return False

    return True


def check_qlib_data(qlib_config):
    inst_dir = Path(qlib_config["provider_uri"]).joinpath("instruments")
    for _p in inst_dir.glob("*.txt"):
        assert len(pd.read_csv(_p, sep="\t", nrows=0, header=None).columns) == 3, (
            f"\nThe {str(_p.resolve())} of qlib data is not equal to 3 columns:"
            f"\n\tIf you are using the data provided by qlib: "
            f"https://qlib.readthedocs.io/en/latest/component/data.html#qlib-format-dataset"
            f"\n\tIf you are using your own data, please dump the data again: "
            f"https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format"
        )


def lazy_sort_index(df: pd.DataFrame, axis=0) -> pd.DataFrame:
    """
    make the df index sorted

    df.sort_index() will take a lot of time even when `df.is_lexsorted() == True`
    This function could avoid such case

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame:
        sorted dataframe
    """
    idx = df.index if axis == 0 else df.columns
    # NOTE: MultiIndex.is_lexsorted() is a deprecated method in Pandas 1.3.0 and is suggested to be replaced by MultiIndex.is_monotonic_increasing (see discussion here: https://github.com/pandas-dev/pandas/issues/32259). However, in case older versions of Pandas is implemented, MultiIndex.is_lexsorted() is necessary to prevent certain fatal errors.
    if idx.is_monotonic_increasing and not (isinstance(idx, pd.MultiIndex) and not idx.is_lexsorted()):
        return df
    else:
        return df.sort_index(axis=axis)


FLATTEN_TUPLE = "_FLATTEN_TUPLE"


def flatten_dict(d, parent_key="", sep=".") -> dict:
    """
    Flatten a nested dict.

        >>> flatten_dict({'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y' : 10}}, 'd': [1, 2, 3]})
        >>> {'a': 1, 'c.a': 2, 'c.b.x': 5, 'd': [1, 2, 3], 'c.b.y': 10}

        >>> flatten_dict({'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y' : 10}}, 'd': [1, 2, 3]}, sep=FLATTEN_TUPLE)
        >>> {'a': 1, ('c','a'): 2, ('c','b','x'): 5, 'd': [1, 2, 3], ('c','b','y'): 10}

    Args:
        d (dict): the dict waiting for flatting
        parent_key (str, optional): the parent key, will be a prefix in new key. Defaults to "".
        sep (str, optional): the separator for string connecting. FLATTEN_TUPLE for tuple connecting.

    Returns:
        dict: flatten dict
    """
    items = []
    for k, v in d.items():
        if sep == FLATTEN_TUPLE:
            new_key = (parent_key, k) if parent_key else k
        else:
            new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_item_from_obj(config: dict, name_path: str) -> object:
    """
    Follow the name_path to get values from config
    For example:
    If we follow the example in in the Parameters section,
        Timestamp('2008-01-02 00:00:00') will be returned

    Parameters
    ----------
    config : dict
        e.g.
        {'dataset': {'class': 'DatasetH',
          'kwargs': {'handler': {'class': 'Alpha158',
                                 'kwargs': {'end_time': '2020-08-01',
                                            'fit_end_time': '<dataset.kwargs.segments.train.1>',
                                            'fit_start_time': '<dataset.kwargs.segments.train.0>',
                                            'instruments': 'csi100',
                                            'start_time': '2008-01-01'},
                                 'module_path': 'qlib.contrib.data.handler'},
                     'segments': {'test': (Timestamp('2017-01-03 00:00:00'),
                                           Timestamp('2019-04-08 00:00:00')),
                                  'train': (Timestamp('2008-01-02 00:00:00'),
                                            Timestamp('2014-12-31 00:00:00')),
                                  'valid': (Timestamp('2015-01-05 00:00:00'),
                                            Timestamp('2016-12-30 00:00:00'))}}
        }}
    name_path : str
        e.g.
        "dataset.kwargs.segments.train.1"

    Returns
    -------
    object
        the retrieved object
    """
    cur_cfg = config
    for k in name_path.split("."):
        if isinstance(cur_cfg, dict):
            cur_cfg = cur_cfg[k]
        elif k.isdigit():
            cur_cfg = cur_cfg[int(k)]
        else:
            raise ValueError(f"Error when getting {k} from cur_cfg")
    return cur_cfg


def fill_placeholder(config: dict, config_extend: dict):
    """
    Detect placeholder in config and fill them with config_extend.
    The item of dict must be single item(int, str, etc), dict and list. Tuples are not supported.
    There are two type of variables:
    - user-defined variables :
        e.g. when config_extend is `{"<MODEL>": model, "<DATASET>": dataset}`, "<MODEL>" and "<DATASET>" in `config` will be replaced with `model` `dataset`
    - variables extracted from `config` :
        e.g. the variables like "<dataset.kwargs.segments.train.0>" will be replaced with the values from `config`

    Parameters
    ----------
    config : dict
        the parameter dict will be filled
    config_extend : dict
        the value of all placeholders

    Returns
    -------
    dict
        the parameter dict
    """
    # check the format of config_extend
    for placeholder in config_extend.keys():
        assert re.match(r"<[^<>]+>", placeholder)

    # bfs
    top = 0
    tail = 1
    item_queue = [config]
    while top < tail:
        now_item = item_queue[top]
        top += 1
        if isinstance(now_item, list):
            item_keys = range(len(now_item))
        elif isinstance(now_item, dict):
            item_keys = now_item.keys()
        for key in item_keys:
            if isinstance(now_item[key], list) or isinstance(now_item[key], dict):
                item_queue.append(now_item[key])
                tail += 1
            elif isinstance(now_item[key], str):
                if now_item[key] in config_extend.keys():
                    now_item[key] = config_extend[now_item[key]]
                else:
                    m = re.match(r"<(?P<name_path>[^<>]+)>", now_item[key])
                    if m is not None:
                        now_item[key] = get_item_from_obj(config, m.groupdict()["name_path"])
    return config


def auto_filter_kwargs(func: Callable) -> Callable:
    """
    this will work like a decoration function

    The decrated function will ignore and give warning when the parameter is not acceptable

    Parameters
    ----------
    func : Callable
        The original function

    Returns
    -------
    Callable:
        the new callable function
    """

    def _func(*args, **kwargs):
        spec = inspect.getfullargspec(func)
        new_kwargs = {}
        for k, v in kwargs.items():
            # if `func` don't accept variable keyword arguments like `**kwargs` and have not according named arguments
            if spec.varkw is None and k not in spec.args:
                log.warning(f"The parameter `{k}` with value `{v}` is ignored.")
            else:
                new_kwargs[k] = v
        return func(*args, **new_kwargs)

    return _func


#################### Wrapper #####################
class Wrapper:
    """Wrapper class for anything that needs to set up during qlib.init"""

    def __init__(self):
        self._provider = None

    def register(self, provider):
        self._provider = provider

    def __repr__(self):
        return "{name}(provider={provider})".format(name=self.__class__.__name__, provider=self._provider)

    def __getattr__(self, key):
        if self.__dict__.get("_provider", None) is None:
            raise AttributeError("Please run qlib.init() first using qlib")
        return getattr(self._provider, key)


def register_wrapper(wrapper, cls_or_obj, module_path=None):
    """register_wrapper

    :param wrapper: A wrapper.
    :param cls_or_obj:  A class or class name or object instance.
    """
    if isinstance(cls_or_obj, str):
        module = get_module_by_module_path(module_path)
        cls_or_obj = getattr(module, cls_or_obj)
    obj = cls_or_obj() if isinstance(cls_or_obj, type) else cls_or_obj
    wrapper.register(obj)


def load_dataset(path_or_obj, index_col=[0, 1]):
    """load dataset from multiple file formats"""
    if isinstance(path_or_obj, pd.DataFrame):
        return path_or_obj
    if not os.path.exists(path_or_obj):
        raise ValueError(f"file {path_or_obj} doesn't exist")
    _, extension = os.path.splitext(path_or_obj)
    if extension == ".h5":
        return pd.read_hdf(path_or_obj)
    elif extension == ".pkl":
        return pd.read_pickle(path_or_obj)
    elif extension == ".csv":
        return pd.read_csv(path_or_obj, parse_dates=True, index_col=index_col)
    raise ValueError(f"unsupported file type `{extension}`")


def code_to_fname(code: str):
    """stock code to file name

    Parameters
    ----------
    code: str
    """
    # NOTE: In windows, the following name is I/O device, and the file with the corresponding name cannot be created
    # reference: https://superuser.com/questions/86999/why-cant-i-name-a-folder-or-file-con-in-windows
    replace_names = ["CON", "PRN", "AUX", "NUL"]
    replace_names += [f"COM{i}" for i in range(10)]
    replace_names += [f"LPT{i}" for i in range(10)]

    prefix = "_qlib_"
    if str(code).upper() in replace_names:
        code = prefix + str(code)

    return code


def fname_to_code(fname: str):
    """file name to stock code

    Parameters
    ----------
    fname: str
    """

    prefix = "_qlib_"
    if fname.startswith(prefix):
        fname = fname.lstrip(prefix)
    return fname

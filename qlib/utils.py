# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import os
import re
import copy
import json
import yaml
import redis
import bisect
import shutil
import difflib
import hashlib
import datetime
import requests
import tempfile
import importlib
import contextlib
import numpy as np
import pandas as pd
from pathlib import Path

from .config import C
from .log import get_module_logger

log = get_module_logger("utils")


#################### Server ####################
def get_redis_connection():
    """get redis connection instance."""
    return redis.StrictRedis(host=C.redis_host, port=C.redis_port, db=C.redis_task_db)


#################### Data ####################
def read_bin(file_path, start_index, end_index):
    with open(file_path, "rb") as f:
        # read start_index
        ref_start_index = int(np.frombuffer(f.read(4), dtype="<f")[0])
        si = max(ref_start_index, start_index)
        if si > end_index:
            return pd.Series()
        # calculate offset
        f.seek(4 * (si - ref_start_index) + 4)
        # read nbytes
        count = end_index - si + 1
        data = np.frombuffer(f.read(4 * count), dtype="<f")
        series = pd.Series(data, index=pd.RangeIndex(si, si + len(data)))
    return series


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
            return yaml.load(f)
    # Check whether the str can be parsed
    try:
        return yaml.load(config)
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
    if not isinstance(field, str):
        field = str(field)
    return re.sub(r"\$(\w+)", r'Feature("\1")', field)


def get_module_by_module_path(module_path):
    """Load module path

    :param module_path:
    :return:
    """

    if  module_path.endswith(".py"):
        module_spec = importlib.util.spec_from_file_location("", module_path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)

    return module


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


def create_save_path(save_path=None):
    """Create save path

    :param save_path:
    :return:
    """
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        temp_dir = os.path.expanduser("~/tmp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        _, save_path = tempfile.mkstemp(dir=temp_dir)
    return save_path


@contextlib.contextmanager
def save_multiple_parts_file(filename, format="gztar"):
    """Save multiple parts file

    Implementation process:
        1. get the absolute path to 'filename'
        2. create a 'filename' directory
        3. user does something with file_path('filename/')
        4. remove 'filename' directory
        5. make_archive 'filename' directory, and rename 'archive file' to filename

    :param filename: result model path
    :param format: archive format: one of "zip", "tar", "gztar", "bztar", or "xztar"
    :return: real model path

    Usage::

        >>> # The following code will create an archive file('~/tmp/test_file') containing 'test_doc_i'(i is 0-10) files.
        >>> with save_multiple_parts_file('~/tmp/test_file') as filename_dir:
        ...   for i in range(10):
        ...       temp_path = os.path.join(filename_dir, 'test_doc_{}'.format(str(i)))
        ...       with open(temp_path) as fp:
        ...           fp.write(str(i))
        ...

    """

    if filename.startswith("~"):
        filename = os.path.expanduser(filename)

    file_path = os.path.abspath(filename)

    # Create model dir
    if os.path.exists(file_path):
        raise FileExistsError("ERROR: file exists: {}, cannot be create the directory.".format(file_path))

    os.makedirs(file_path)

    # return model dir
    yield file_path

    # filename dir to filename.tar.gz file
    tar_file = shutil.make_archive(file_path, format=format, root_dir=file_path)

    # Remove filename dir
    if os.path.exists(file_path):
        shutil.rmtree(file_path)

    # filename.tar.gz rename to filename
    os.rename(tar_file, file_path)


@contextlib.contextmanager
def unpack_archive_with_buffer(buffer, format="gztar"):
    """Unpack archive with archive buffer
    After the call is finished, the archive file and directory will be deleted.

    Implementation process:
        1. create 'tempfile' in '~/tmp/' and directory
        2. 'buffer' write to 'tempfile'
        3. unpack archive file('tempfile')
        4. user does something with file_path('tempfile/')
        5. remove 'tempfile' and 'tempfile directory'

    :param buffer: bytes
    :param format: archive format: one of "zip", "tar", "gztar", "bztar", or "xztar"
    :return: unpack archive directory path

    Usage::

        >>> # The following code is to print all the file names in 'test_unpack.tar.gz'
        >>> with open('test_unpack.tar.gz') as fp:
        ...     buffer = fp.read()
        ...
        >>> with unpack_archive_with_buffer(buffer) as temp_dir:
        ...     for f_n in os.listdir(temp_dir):
        ...         print(f_n)
        ...

    """
    temp_dir = os.path.expanduser("~/tmp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=temp_dir) as fp:
        fp.write(buffer)
        file_path = fp.name

    try:
        tar_file = file_path + ".tar.gz"
        os.rename(file_path, tar_file)
        # Create dir
        os.makedirs(file_path)
        shutil.unpack_archive(tar_file, format=format, extract_dir=file_path)

        # Return temp dir
        yield file_path

    except Exception as e:
        log.error(str(e))
    finally:
        # Remove temp tar file
        if os.path.exists(tar_file):
            os.unlink(tar_file)

        # Remove temp model dir
        if os.path.exists(file_path):
            shutil.rmtree(file_path)


@contextlib.contextmanager
def get_tmp_file_with_buffer(buffer):
    temp_dir = os.path.expanduser("~/tmp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    with tempfile.NamedTemporaryFile("wb", delete=True, dir=temp_dir) as fp:
        fp.write(buffer)
        file_path = fp.name
        yield file_path


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
    from .data import D

    return str(cur_date.date()) == str(D.calendar(start_time=cur_date, future=True)[0].date())


def get_date_range(trading_date, shift, future=False):
    """get trading date range by shift

    :param trading_date:
    :param shift: int
    :param future: bool
    :return:
    """
    from .data import D

    calendar = D.calendar(future=future)
    if pd.to_datetime(trading_date) not in list(calendar):
        raise ValueError("{} is not trading day!".format(str(trading_date)))
    day_index = bisect.bisect_left(calendar, trading_date)
    if 0 <= (day_index + shift) < len(calendar):
        if shift > 0:
            return calendar[day_index + 1 : day_index + 1 + shift]
        else:
            return calendar[day_index + shift : day_index]
    else:
        return calendar


def get_date_by_shift(trading_date, shift, future=False):
    """get trading date with shift bias wil cur_date
        e.g. : shift == 1,  return next trading date
               shift == -1, return previous trading date
    ----------
    trading_date : pandas.Timestamp
        current date
    shift : int
    """
    return get_date_range(trading_date, shift, future)[0 if shift < 0 else -1] if shift != 0 else trading_date


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
    """get previous trading date
    If end_date is -1, None, or end_date is greater than the maximum trading day, the last trading date is returned.
    Otherwise, returns the end_date
    ----------
    end_date: str
        end trading date
    date : pandas.Timestamp
        current date
    """
    from .data import D

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
        if not list(features_dir.rglob(f"*.{_calendar.name.split('.')[0]}.bin")):
            return False

    # check instruments
    code_names = set(map(lambda x: x.name.lower(), features_dir.iterdir()))
    _instrument = instruments_dir.joinpath("all.txt")
    miss_code = set(pd.read_csv(_instrument, sep="\t", header=None).loc[:, 0].apply(str.lower)) - set(code_names)
    if miss_code and any(map(lambda x: "sht" not in x, miss_code)):
        return False

    return True

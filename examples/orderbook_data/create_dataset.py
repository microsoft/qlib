# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
NOTE:
- This scripts is a demo to import example data import Qlib
- !!!!!!!!!!!!!!!TODO!!!!!!!!!!!!!!!!!!!:
    - Its structure is not well designed and very ugly, your contribution is welcome to make importing dataset easier
"""
from datetime import date, datetime as dt
import os
from pathlib import Path
import random
import shutil
import time
import traceback

from arctic import Arctic, chunkstore
import arctic
from arctic import Arctic, CHUNK_STORE
from arctic.chunkstore.chunkstore import CHUNK_SIZE
import fire
from joblib import Parallel, delayed, parallel
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.core.indexes.datetimes import date_range
from pymongo.mongo_client import MongoClient

DIRNAME = Path(__file__).absolute().resolve().parent

# CONFIG
N_JOBS = -1  # leaving one kernel free
LOG_FILE_PATH = DIRNAME / "log_file"
DATA_PATH = DIRNAME / "raw_data"
DATABASE_PATH = DIRNAME / "orig_data"
DATA_INFO_PATH = DIRNAME / "data_info"
DATA_FINISH_INFO_PATH = DIRNAME / "./data_finish_info"
DOC_TYPE = ["Tick", "Order", "OrderQueue", "Transaction", "Day", "Minute"]
MAX_SIZE = 3000 * 1024 * 1024 * 1024
ALL_STOCK_PATH = DATABASE_PATH / "all.txt"
ARCTIC_SRV = "127.0.0.1"


def get_library_name(doc_type):
    if str.lower(doc_type) == str.lower("Tick"):
        return "ticks"
    else:
        return str.lower(doc_type)


def is_stock(exchange_place, code):
    if exchange_place == "SH" and code[0] != "6":
        return False
    if exchange_place == "SZ" and code[0] != "0" and code[:2] != "30":
        return False
    return True


def add_one_stock_daily_data(filepath, type, exchange_place, arc, date):
    """
    exchange_place: "SZ" OR "SH"
    type: "tick", "orderbook", ...
    filepath: the path of csv
    arc: arclink created by a process
    """
    code = os.path.split(filepath)[-1].split(".csv")[0]
    if exchange_place == "SH" and code[0] != "6":
        return
    if exchange_place == "SZ" and code[0] != "0" and code[:2] != "30":
        return

    df = pd.read_csv(filepath, encoding="gbk", dtype={"code": str})
    code = os.path.split(filepath)[-1].split(".csv")[0]

    def format_time(day, hms):
        day = str(day)
        hms = str(hms)
        if hms[0] == "1":  # >=10,
            return (
                "-".join([day[0:4], day[4:6], day[6:8]]) + " " + ":".join([hms[:2], hms[2:4], hms[4:6] + "." + hms[6:]])
            )
        else:
            return (
                "-".join([day[0:4], day[4:6], day[6:8]]) + " " + ":".join([hms[:1], hms[1:3], hms[3:5] + "." + hms[5:]])
            )

    ## Discard the entire row if wrong data timestamp encoutered.
    timestamp = list(zip(list(df["date"]), list(df["time"])))
    error_index_list = []
    for index, t in enumerate(timestamp):
        try:
            pd.Timestamp(format_time(t[0], t[1]))
        except Exception:
            error_index_list.append(index)  ## The row number of the error line

    # to-do: writting to logs

    if len(error_index_list) > 0:
        print("error: {}, {}".format(filepath, len(error_index_list)))

    df = df.drop(error_index_list)
    timestamp = list(zip(list(df["date"]), list(df["time"])))  ## The cleaned timestamp
    # generate timestamp
    pd_timestamp = pd.DatetimeIndex(
        [pd.Timestamp(format_time(timestamp[i][0], timestamp[i][1])) for i in range(len(df["date"]))]
    )
    df = df.drop(columns=["date", "time", "name", "code", "wind_code"])
    # df = pd.DataFrame(data=df.to_dict("list"), index=pd_timestamp)
    df["date"] = pd.to_datetime(pd_timestamp)
    df.set_index("date", inplace=True)

    if str.lower(type) == "orderqueue":
        ## extract ab1~ab50
        df["ab"] = [
            ",".join([str(int(row["ab" + str(i + 1)])) for i in range(0, row["ab_items"])])
            for timestamp, row in df.iterrows()
        ]
        df = df.drop(columns=["ab" + str(i) for i in range(1, 51)])

    type = get_library_name(type)
    # arc.initialize_library(type, lib_type=CHUNK_STORE)
    lib = arc[type]

    symbol = "".join([exchange_place, code])
    if symbol in lib.list_symbols():
        print("update {0}, date={1}".format(symbol, date))
        if df.empty == True:
            return error_index_list
        lib.update(symbol, df, chunk_size="D")
    else:
        print("write {0}, date={1}".format(symbol, date))
        lib.write(symbol, df, chunk_size="D")
    return error_index_list


def add_one_stock_daily_data_wrapper(filepath, type, exchange_place, index, date):
    pid = os.getpid()
    code = os.path.split(filepath)[-1].split(".csv")[0]
    arc = Arctic(ARCTIC_SRV)
    try:
        if index % 100 == 0:
            print("index = {}, filepath = {}".format(index, filepath))
        error_index_list = add_one_stock_daily_data(filepath, type, exchange_place, arc, date)
        if error_index_list is not None and len(error_index_list) > 0:
            f = open(os.path.join(LOG_FILE_PATH, "temp_timestamp_error_{0}_{1}_{2}.txt".format(pid, date, type)), "a+")
            f.write("{}, {}, {}\n".format(filepath, error_index_list, exchange_place + "_" + code))
            f.close()

    except Exception as e:
        info = traceback.format_exc()
        print("error:" + str(e))
        f = open(os.path.join(LOG_FILE_PATH, "temp_fail_{0}_{1}_{2}.txt".format(pid, date, type)), "a+")
        f.write("fail:" + str(filepath) + "\n" + str(e) + "\n" + str(info) + "\n")
        f.close()

    finally:
        arc.reset()


def add_data(tick_date, doc_type, stock_name_dict):
    pid = os.getpid()

    if doc_type not in DOC_TYPE:
        print("doc_type not in {}".format(DOC_TYPE))
        return
    try:
        begin_time = time.time()
        os.system(f"cp {DATABASE_PATH}/{tick_date + '_{}.tar.gz'.format(doc_type)} {DATA_PATH}/")

        os.system(
            f"tar -xvzf {DATA_PATH}/{tick_date + '_{}.tar.gz'.format(doc_type)} -C {DATA_PATH}/ {tick_date + '_' + doc_type}/SH"
        )
        os.system(
            f"tar -xvzf {DATA_PATH}/{tick_date + '_{}.tar.gz'.format(doc_type)} -C {DATA_PATH}/ {tick_date + '_' + doc_type}/SZ"
        )
        os.system(f"chmod 777 {DATA_PATH}")
        os.system(f"chmod 777 {DATA_PATH}/{tick_date + '_' + doc_type}")
        os.system(f"chmod 777 {DATA_PATH}/{tick_date + '_' + doc_type}/SH")
        os.system(f"chmod 777 {DATA_PATH}/{tick_date + '_' + doc_type}/SZ")
        os.system(f"chmod 777 {DATA_PATH}/{tick_date + '_' + doc_type}/SH/{tick_date}")
        os.system(f"chmod 777 {DATA_PATH}/{tick_date + '_' + doc_type}/SZ/{tick_date}")

        print("tick_date={}".format(tick_date))

        temp_data_path_sh = os.path.join(DATA_PATH, tick_date + "_" + doc_type, "SH", tick_date)
        temp_data_path_sz = os.path.join(DATA_PATH, tick_date + "_" + doc_type, "SZ", tick_date)
        is_files_exist = {"sh": os.path.exists(temp_data_path_sh), "sz": os.path.exists(temp_data_path_sz)}

        sz_files = (
            (
                set([i.split(".csv")[0] for i in os.listdir(temp_data_path_sz) if i[:2] == "30" or i[0] == "0"])
                & set(stock_name_dict["SZ"])
            )
            if is_files_exist["sz"]
            else set()
        )
        sz_file_nums = len(sz_files) if is_files_exist["sz"] else 0
        sh_files = (
            (
                set([i.split(".csv")[0] for i in os.listdir(temp_data_path_sh) if i[0] == "6"])
                & set(stock_name_dict["SH"])
            )
            if is_files_exist["sh"]
            else set()
        )
        sh_file_nums = len(sh_files) if is_files_exist["sh"] else 0
        print("sz_file_nums:{}, sh_file_nums:{}".format(sz_file_nums, sh_file_nums))

        f = (DATA_INFO_PATH / "data_info_log_{}_{}".format(doc_type, tick_date)).open("w+")
        f.write("sz:{}, sh:{}, date:{}:".format(sz_file_nums, sh_file_nums, tick_date) + "\n")
        f.close()

        if sh_file_nums > 0:
            # write is not thread-safe, update may be thread-safe
            Parallel(n_jobs=N_JOBS)(
                delayed(add_one_stock_daily_data_wrapper)(
                    os.path.join(temp_data_path_sh, name + ".csv"), doc_type, "SH", index, tick_date
                )
                for index, name in enumerate(list(sh_files))
            )
        if sz_file_nums > 0:
            # write is not thread-safe, update may be thread-safe
            Parallel(n_jobs=N_JOBS)(
                delayed(add_one_stock_daily_data_wrapper)(
                    os.path.join(temp_data_path_sz, name + ".csv"), doc_type, "SZ", index, tick_date
                )
                for index, name in enumerate(list(sz_files))
            )

        os.system(f"rm -f {DATA_PATH}/{tick_date + '_{}.tar.gz'.format(doc_type)}")
        os.system(f"rm -rf {DATA_PATH}/{tick_date + '_' + doc_type}")
        total_time = time.time() - begin_time
        f = (DATA_FINISH_INFO_PATH / "data_info_finish_log_{}_{}".format(doc_type, tick_date)).open("w+")
        f.write("finish: date:{}, consume_time:{}, end_time: {}".format(tick_date, total_time, time.time()) + "\n")
        f.close()

    except Exception as e:
        info = traceback.format_exc()
        print("date error:" + str(e))
        f = open(os.path.join(LOG_FILE_PATH, "temp_fail_{0}_{1}_{2}.txt".format(pid, tick_date, doc_type)), "a+")
        f.write("fail:" + str(tick_date) + "\n" + str(e) + "\n" + str(info) + "\n")
        f.close()


class DSCreator:
    """Dataset creator"""

    def clear(self):
        client = MongoClient(ARCTIC_SRV)
        client.drop_database("arctic")

    def initialize_library(self):
        arc = Arctic(ARCTIC_SRV)
        for doc_type in DOC_TYPE:
            arc.initialize_library(get_library_name(doc_type), lib_type=CHUNK_STORE)

    def _get_empty_folder(self, fp: Path):
        fp = Path(fp)
        if fp.exists():
            shutil.rmtree(fp)
        fp.mkdir(parents=True, exist_ok=True)

    def import_data(self, doc_type_l=["Tick", "Transaction", "Order"]):
        # clear all the old files
        for fp in LOG_FILE_PATH, DATA_INFO_PATH, DATA_FINISH_INFO_PATH, DATA_PATH:
            self._get_empty_folder(fp)

        arc = Arctic(ARCTIC_SRV)
        for doc_type in DOC_TYPE:
            # arc.initialize_library(get_library_name(doc_type), lib_type=CHUNK_STORE)
            arc.set_quota(get_library_name(doc_type), MAX_SIZE)
        arc.reset()

        # doc_type = 'Day'
        for doc_type in doc_type_l:
            date_list = list(set([int(path.split("_")[0]) for path in os.listdir(DATABASE_PATH) if doc_type in path]))
            date_list.sort()
            date_list = [str(date) for date in date_list]

            f = open(ALL_STOCK_PATH, "r")
            stock_name_list = [lines.split("\t")[0] for lines in f.readlines()]
            f.close()
            stock_name_dict = {
                "SH": [stock_name[2:] for stock_name in stock_name_list if "SH" in stock_name],
                "SZ": [stock_name[2:] for stock_name in stock_name_list if "SZ" in stock_name],
            }

            lib_name = get_library_name(doc_type)
            a = Arctic(ARCTIC_SRV)
            # a.initialize_library(lib_name, lib_type=CHUNK_STORE)

            stock_name_exist = a[lib_name].list_symbols()
            lib = a[lib_name]
            initialize_count = 0
            for stock_name in stock_name_list:
                if stock_name not in stock_name_exist:
                    initialize_count += 1
                    # A placeholder for stocks
                    pdf = pd.DataFrame(index=[pd.Timestamp("1900-01-01")])
                    pdf.index.name = "date"  # an col named date is necessary
                    lib.write(stock_name, pdf)
            print("initialize count: {}".format(initialize_count))
            print("tasks: {}".format(date_list))
            a.reset()

            # date_list = [files.split("_")[0] for files in os.listdir("./raw_data_price") if "tar" in files]
            # print(len(date_list))
            date_list = ["20201231"]  # for test
            Parallel(n_jobs=min(2, len(date_list)))(
                delayed(add_data)(date, doc_type, stock_name_dict) for date in date_list
            )


if __name__ == "__main__":
    fire.Fire(DSCreator)

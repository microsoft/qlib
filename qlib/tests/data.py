# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import sys
import qlib
import shutil
import zipfile
import requests
import datetime
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from qlib.utils import exists_qlib_data


class GetData:
    DATASET_VERSION = "v2"
    REMOTE_URL = "http://fintech.msra.cn/stock_data/downloads"
    QLIB_DATA_NAME = "{dataset_name}_{region}_{interval}_{qlib_version}.zip"

    def __init__(self, delete_zip_file=False):
        """

        Parameters
        ----------
        delete_zip_file : bool, optional
            Whether to delete the zip file, value from True or False, by default False
        """
        self.delete_zip_file = delete_zip_file

    def normalize_dataset_version(self, dataset_version: str = None):
        if dataset_version is None:
            dataset_version = self.DATASET_VERSION
        return dataset_version

    def merge_remote_url(self, file_name: str, dataset_version: str = None):
        return f"{self.REMOTE_URL}/{self.normalize_dataset_version(dataset_version)}/{file_name}"

    def _download_data(
        self, file_name: str, target_dir: [Path, str], delete_old: bool = True, dataset_version: str = None
    ):
        target_dir = Path(target_dir).expanduser()
        target_dir.mkdir(exist_ok=True, parents=True)
        # saved file name
        _target_file_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file_name
        target_path = target_dir.joinpath(_target_file_name)

        url = self.merge_remote_url(file_name, dataset_version)
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        if resp.status_code != 200:
            raise requests.exceptions.HTTPError()

        chunk_size = 1024
        logger.warning(
            f"The data for the example is collected from Yahoo Finance. Please be aware that the quality of the data might not be perfect. (You can refer to the original data source: https://finance.yahoo.com/lookup.)"
        )
        logger.info(f"{file_name} downloading......")
        with tqdm(total=int(resp.headers.get("Content-Length", 0))) as p_bar:
            with target_path.open("wb") as fp:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    fp.write(chunk)
                    p_bar.update(chunk_size)

        self._unzip(target_path, target_dir, delete_old)
        if self.delete_zip_file:
            target_path.unlink()

    def check_dataset(self, file_name: str, dataset_version: str = None):
        url = self.merge_remote_url(file_name, dataset_version)
        resp = requests.get(url, stream=True)
        status = True
        if resp.status_code == 404:
            status = False
        return status

    @staticmethod
    def _unzip(file_path: Path, target_dir: Path, delete_old: bool = True):
        if delete_old:
            logger.warning(
                f"will delete the old qlib data directory(features, instruments, calendars, features_cache, dataset_cache): {target_dir}"
            )
            GetData._delete_qlib_data(target_dir)
        logger.info(f"{file_path} unzipping......")
        with zipfile.ZipFile(str(file_path.resolve()), "r") as zp:
            for _file in tqdm(zp.namelist()):
                zp.extract(_file, str(target_dir.resolve()))

    @staticmethod
    def _delete_qlib_data(file_dir: Path):
        rm_dirs = []
        for _name in ["features", "calendars", "instruments", "features_cache", "dataset_cache"]:
            _p = file_dir.joinpath(_name)
            if _p.exists():
                rm_dirs.append(str(_p.resolve()))
        if rm_dirs:
            flag = input(
                f"Will be deleted: "
                f"\n\t{rm_dirs}"
                f"\nIf you do not need to delete {file_dir}, please change the <--target_dir>"
                f"\nAre you sure you want to delete, yes(Y/y), no (N/n):"
            )
            if str(flag) not in ["Y", "y"]:
                sys.exit()
            for _p in rm_dirs:
                logger.warning(f"delete: {_p}")
                shutil.rmtree(_p)

    def qlib_data(
        self,
        name="qlib_data",
        target_dir="~/.qlib/qlib_data/cn_data",
        version=None,
        interval="1d",
        region="cn",
        delete_old=True,
        exists_skip=False,
    ):
        """download cn qlib data from remote

        Parameters
        ----------
        target_dir: str
            data save directory
        name: str
            dataset name, value from [qlib_data, qlib_data_simple], by default qlib_data
        version: str
            data version, value from [v1, ...], by default None(use script to specify version)
        interval: str
            data freq, value from [1d], by default 1d
        region: str
            data region, value from [cn, us], by default cn
        delete_old: bool
            delete an existing directory, by default True
        exists_skip: bool
            exists skip, by default False

        Examples
        ---------
        # get 1d data
        python get_data.py qlib_data --name qlib_data --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn

        # get 1min data
        python get_data.py qlib_data --name qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --interval 1min --region cn
        -------

        """
        if exists_skip and exists_qlib_data(target_dir):
            logger.warning(
                f"Data already exists: {target_dir}, the data download will be skipped\n"
                f"\tIf downloading is required: `exists_skip=False` or `change target_dir`"
            )
            return

        qlib_version = ".".join(re.findall(r"(\d+)\.+", qlib.__version__))

        def _get_file_name(v):
            return self.QLIB_DATA_NAME.format(
                dataset_name=name, region=region.lower(), interval=interval.lower(), qlib_version=v
            )

        file_name = _get_file_name(qlib_version)
        if not self.check_dataset(file_name, version):
            file_name = _get_file_name("latest")
        self._download_data(file_name.lower(), target_dir, delete_old, dataset_version=version)

    def csv_data_cn(self, target_dir="~/.qlib/csv_data/cn_data"):
        """download cn csv data from remote

        Parameters
        ----------
        target_dir: str
            data save directory

        Examples
        ---------
        python get_data.py csv_data_cn --target_dir ~/.qlib/csv_data/cn_data
        -------

        """
        file_name = "csv_data_cn.zip"
        self._download_data(file_name, target_dir)

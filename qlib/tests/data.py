# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import zipfile
import requests
from tqdm import tqdm
from pathlib import Path
from loguru import logger


class GetData:
    REMOTE_URL = "http://fintech.msra.cn/stock_data/downloads"

    def __init__(self, delete_zip_file=False):
        """

        Parameters
        ----------
        delete_zip_file : bool, optional
            Whether to delete the zip file, value from True or False, by default False
        """
        self.delete_zip_file = delete_zip_file

    def _download_data(self, file_name: str, target_dir: [Path, str]):
        target_dir = Path(target_dir).expanduser()
        target_dir.mkdir(exist_ok=True, parents=True)

        url = f"{self.REMOTE_URL}/{file_name}"
        target_path = target_dir.joinpath(file_name)

        resp = requests.get(url, stream=True)
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

        self._unzip(target_path, target_dir)
        if self.delete_zip_file:
            target_path.unlink()

    @staticmethod
    def _unzip(file_path: Path, target_dir: Path):
        logger.info(f"{file_path} unzipping......")
        with zipfile.ZipFile(str(file_path.resolve()), "r") as zp:
            for _file in tqdm(zp.namelist()):
                zp.extract(_file, str(target_dir.resolve()))

    def qlib_data(
        self, name="qlib_data", target_dir="~/.qlib/qlib_data/cn_data", version="latest", interval="1d", region="cn"
    ):
        """download cn qlib data from remote

        Parameters
        ----------
        target_dir: str
            data save directory
        name: str
            dataset name, value from [qlib_data, qlib_data_simple], by default qlib_data
        version: str
            data version, value from [v0, v1, ..., latest], by default latest
        interval: str
            data freq, value from [1d], by default 1d
        region: str
            data region, value from [cn, us], by default cn

        Examples
        ---------
        python get_data.py qlib_data --name qlib_data --target_dir ~/.qlib/qlib_data/cn_data --version latest --interval 1d --region cn
        -------

        """
        file_name = f"{name}_{region.lower()}_{interval.lower()}_{version}.zip"
        self._download_data(file_name.lower(), target_dir)

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

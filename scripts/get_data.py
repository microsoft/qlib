#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import fire
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

        chuck_size = 1024
        logger.info(f"{file_name} downloading......")
        with tqdm(total=int(resp.headers.get("Content-Length", 0))) as p_bar:
            with target_path.open("wb") as fp:
                for chuck in resp.iter_content(chunk_size=chuck_size):
                    fp.write(chuck)
                    p_bar.update(chuck_size)

        self._unzip(target_path, target_dir)
        if self.delete_zip_file:
            target_path.unlike()

    @staticmethod
    def _unzip(file_path: Path, target_dir: Path):
        logger.info(f"{file_path} unzipping......")
        with zipfile.ZipFile(str(file_path.resolve()), "r") as zp:
            for _file in tqdm(zp.namelist()):
                zp.extract(_file, str(target_dir.resolve()))

    def qlib_data_cn(self, target_dir="~/.qlib/qlib_data/cn_data", version="v1"):
        """download cn qlib data from remote

        Parameters
        ----------
        target_dir: str
            data save directory
        version: str
            data version, value from [v0, v1], by default v1

        Examples
        ---------
        python get_data.py qlib_data_cn --target_dir ~/.qlib/qlib_data/cn_data --version v1
        -------

        """
        file_name = f"qlib_data_cn_{version}.zip"
        self._download_data(file_name, target_dir)

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


if __name__ == "__main__":
    fire.Fire(GetData)

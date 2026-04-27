# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: skip-file
# flake8: noqa

import pathlib
import pandas as pd
import shutil
import os
from ruamel.yaml import YAML
from ...backtest.account import Account
from .user import User
from .utils import load_instance, save_instance
from ...utils import init_instance_by_config


class UserManager:
    def __init__(self, user_data_path, save_report=True):
        """
        This module is designed to manager the users in online system
        all users' data were assumed to be saved in user_data_path
            Parameter
                user_data_path : string
                    data path that all users' data were saved in

        variables:
            data_path : string
                data path that all users' data were saved in
            users_file : string
                A path of the file record the add_date of users
            save_report : bool
                whether to save report after each trading process
            users : dict{}
                [user_id]->User()
                the python dict save instances of User() for each user_id
            user_record : pd.Dataframe
                user_id(string), add_date(string)
                indicate the add_date for each users
        """
        self.data_path = pathlib.Path(user_data_path)
        self.users_file = self.data_path / "users.csv"
        self.save_report = save_report
        self.users = {}
        self.user_record = None

    @staticmethod
    def _validate_user_id(user_id):
        """
        Validate user_id to prevent path traversal / absolute paths.

        user_id is used as a directory name under self.data_path and must not
        contain path separators, parent traversal, or drive/UNC prefixes.
        """
        if not isinstance(user_id, str):
            raise TypeError("user_id must be a string")
        if user_id == "" or "\x00" in user_id:
            raise ValueError("Invalid user_id")

        # Forbid any multi-part paths (e.g. "../x", "a/b", "C:\\x", "\\\\server\\share").
        parts = pathlib.PurePath(user_id).parts
        if len(parts) != 1 or parts[0] in (".", ".."):
            raise ValueError("Invalid user_id")

        # Extra guard: explicit separators (platform-specific and cross-platform)
        if "/" in user_id or "\\" in user_id:
            raise ValueError("Invalid user_id")

        return user_id

    def _user_path(self, user_id):
        """
        Return the resolved user directory path under self.data_path.

        Ensures the resulting path is contained within self.data_path even if
        symlinks are involved.
        """
        user_id = self._validate_user_id(user_id)
        base = self.data_path.resolve(strict=False)
        candidate = (self.data_path / user_id).resolve(strict=False)

        # Ensure candidate is inside base (and on same drive on Windows).
        try:
            common = os.path.commonpath([str(base), str(candidate)])
        except ValueError:
            # Different drives or invalid paths
            raise ValueError("Invalid user_id") from None

        if common != str(base):
            raise ValueError("Invalid user_id")
        if candidate == base:
            raise ValueError("Invalid user_id")

        return candidate

    def load_users(self):
        """
        load all users' data into manager
        """
        self.users = {}
        self.user_record = pd.read_csv(self.users_file, index_col=0)
        for user_id in self.user_record.index:
            self.users[user_id] = self.load_user(user_id)

    def load_user(self, user_id):
        """
        return a instance of User() represents a user to be processed
            Parameter
                user_id : string
            :return
                user : User()
        """
        user_path = self._user_path(user_id)
        account_path = user_path
        strategy_file = user_path / "strategy_{}.pickle".format(user_id)
        model_file = user_path / "model_{}.pickle".format(user_id)
        cur_user_list = list(self.users)
        if user_id in cur_user_list:
            raise ValueError("User {} has been loaded".format(user_id))
        else:
            trade_account = Account(0)
            trade_account.load_account(account_path)
            strategy = load_instance(strategy_file)
            model = load_instance(model_file)
            user = User(account=trade_account, strategy=strategy, model=model)
            return user

    def save_user_data(self, user_id):
        """
        save a instance of User() to user data path
            Parameter
                user_id : string
        """
        if not user_id in self.users:
            raise ValueError("Cannot find user {}".format(user_id))
        user_path = self._user_path(user_id)
        self.users[user_id].account.save_account(user_path)
        save_instance(
            self.users[user_id].strategy,
            user_path / "strategy_{}.pickle".format(user_id),
        )
        save_instance(
            self.users[user_id].model,
            user_path / "model_{}.pickle".format(user_id),
        )

    def add_user(self, user_id, config_file, add_date):
        """
        add the new user {user_id} into user data
        will create a new folder named "{user_id}" in user data path
            Parameter
                user_id : string
                init_cash : int
                config_file : str/pathlib.Path()
                   path of config file
        """
        config_file = pathlib.Path(config_file)
        if not config_file.exists():
            raise ValueError("Cannot find config file {}".format(config_file))
        user_path = self._user_path(user_id)
        if user_path.exists():
            raise ValueError("User data for {} already exists".format(user_id))

        with config_file.open("r") as fp:
            yaml = YAML(typ="safe", pure=True)
            config = yaml.load(fp)
        # load model
        model = init_instance_by_config(config["model"])

        # load strategy
        strategy = init_instance_by_config(config["strategy"])
        init_args = strategy.get_init_args_from_model(model, add_date)
        strategy.init(**init_args)

        # init Account
        trade_account = Account(init_cash=config["init_cash"])

        # save user
        user_path.mkdir()
        save_instance(model, user_path / "model_{}.pickle".format(user_id))
        save_instance(strategy, user_path / "strategy_{}.pickle".format(user_id))
        trade_account.save_account(user_path)
        user_record = pd.read_csv(self.users_file, index_col=0)
        user_record.loc[user_id] = [add_date]
        user_record.to_csv(self.users_file)

    def remove_user(self, user_id):
        """
        remove user {user_id} in current user dataset
        will delete the folder "{user_id}" in user data path
            :param
                user_id : string
        """
        user_path = self._user_path(user_id)
        if not user_path.exists():
            raise ValueError("Cannot find user data {}".format(user_id))
        shutil.rmtree(user_path)
        user_record = pd.read_csv(self.users_file, index_col=0)
        user_record.drop([user_id], inplace=True)
        user_record.to_csv(self.users_file)

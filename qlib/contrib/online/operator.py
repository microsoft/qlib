# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import fire
import pandas as pd
import pathlib
import qlib
import logging

from ...data import D
from ...log import get_module_logger
from ...utils import get_pre_trading_date, is_tradable_date
from ..evaluate import risk_analysis
from ..backtest.backtest import update_account

from .manager import UserManager
from .utils import prepare
from .utils import create_user_folder
from .executor import load_order_list, save_order_list
from .executor import SimulatorExecutor
from .executor import save_score_series, load_score_series


class Operator:
    def __init__(self, client: str):
        """
        Parameters
        ----------
            client: str
                The qlib client config file(.yaml)
        """
        self.logger = get_module_logger("online operator", level=logging.INFO)
        self.client = client

    @staticmethod
    def init(client, path, date=None):
        """Initial UserManager(), get predict date and trade date
        Parameters
        ----------
            client: str
                The qlib client config file(.yaml)
            path : str
                Path to save user account.
            date : str (YYYY-MM-DD)
                Trade date, when the generated order list will be traded.
        Return
        ----------
            um: UserManager()
            pred_date: pd.Timestamp
            trade_date: pd.Timestamp
        """
        qlib.init_from_yaml_conf(client)
        um = UserManager(user_data_path=pathlib.Path(path))
        um.load_users()
        if not date:
            trade_date, pred_date = None, None
        else:
            trade_date = pd.Timestamp(date)
            if not is_tradable_date(trade_date):
                raise ValueError("trade date is not tradable date".format(trade_date.date()))
            pred_date = get_pre_trading_date(trade_date, future=True)
        return um, pred_date, trade_date

    def add_user(self, id, config, path, date):
        """Add a new user into the a folder to run 'online' module.

        Parameters
        ----------
        id : str
            User id, should be unique.
        config : str
            The file path (yaml) of user config
        path : str
            Path to save user account.
        date : str (YYYY-MM-DD)
            The date that user account was added.
        """
        create_user_folder(path)
        qlib.init_from_yaml_conf(self.client)
        um = UserManager(user_data_path=path)
        add_date = D.calendar(end_time=date)[-1]
        if not is_tradable_date(add_date):
            raise ValueError("add date is not tradable date".format(add_date.date()))
        um.add_user(user_id=id, config_file=config, add_date=add_date)

    def remove_user(self, id, path):
        """Remove user from folder used in 'online' module.

        Parameters
        ----------
        id : str
            User id, should be unique.
        path : str
            Path to save user account.
        """
        um = UserManager(user_data_path=path)
        um.remove_user(user_id=id)

    def generate(self, date, path):
        """Generate order list that will be traded at 'date'.

        Parameters
        ----------
        date : str (YYYY-MM-DD)
            Trade date, when the generated order list will be traded.
        path : str
            Path to save user account.
        """
        um, pred_date, trade_date = self.init(self.client, path, date)
        for user_id, user in um.users.items():
            dates, trade_exchange = prepare(um, pred_date, user_id)
            # get and save the score at predict date
            input_data = user.model.get_data_with_date(pred_date)
            score_series = user.model.predict(input_data)
            save_score_series(score_series, (pathlib.Path(path) / user_id), trade_date)

            # update strategy (and model)
            user.strategy.update(score_series, pred_date, trade_date)

            # generate and save order list
            order_list = user.strategy.generate_order_list(
                score_series=score_series,
                current=user.account.current,
                trade_exchange=trade_exchange,
                trade_date=trade_date,
            )
            save_order_list(
                order_list=order_list,
                user_path=(pathlib.Path(path) / user_id),
                trade_date=trade_date,
            )
            self.logger.info("Generate order list at {} for {}".format(trade_date, user_id))
            um.save_user_data(user_id)

    def execute(self, date, exchange_config, path):
        """Execute the orderlist at 'date'.

        Parameters
        ----------
           date : str (YYYY-MM-DD)
               Trade date, that the generated order list will be traded.
           exchange_config: str
               The file path (yaml) of exchange config
           path : str
               Path to save user account.
        """
        um, pred_date, trade_date = self.init(self.client, path, date)
        for user_id, user in um.users.items():
            dates, trade_exchange = prepare(um, trade_date, user_id, exchange_config)
            executor = SimulatorExecutor(trade_exchange=trade_exchange)
            if str(dates[0].date()) != str(pred_date.date()):
                raise ValueError(
                    "The account data is not newest! last trading date {}, today {}".format(
                        dates[0].date(), trade_date.date()
                    )
                )

            # load and execute the order list
            # will not modify the trade_account after executing
            order_list = load_order_list(user_path=(pathlib.Path(path) / user_id), trade_date=trade_date)
            trade_info = executor.execute(order_list=order_list, trade_account=user.account, trade_date=trade_date)
            executor.save_executed_file_from_trade_info(
                trade_info=trade_info,
                user_path=(pathlib.Path(path) / user_id),
                trade_date=trade_date,
            )
            self.logger.info("execute order list at {} for {}".format(trade_date.date(), user_id))

    def update(self, date, path, type="SIM"):
        """Update account at 'date'.

        Parameters
        ----------
        date : str (YYYY-MM-DD)
            Trade date, that the generated order list will be traded.
        path : str
            Path to save user account.
        type : str
            which executor was been used to execute the order list
            'SIM': SimulatorExecutor()
        """
        if type not in ["SIM", "YC"]:
            raise ValueError("type is invalid, {}".format(type))
        um, pred_date, trade_date = self.init(self.client, path, date)
        for user_id, user in um.users.items():
            dates, trade_exchange = prepare(um, trade_date, user_id)
            if type == "SIM":
                executor = SimulatorExecutor(trade_exchange=trade_exchange)
            else:
                raise ValueError("not found executor")
            # dates[0] is the last_trading_date
            if str(dates[0].date()) > str(pred_date.date()):
                raise ValueError(
                    "The account data is not newest! last trading date {}, today {}".format(
                        dates[0].date(), trade_date.date()
                    )
                )
            # load trade info and update account
            trade_info = executor.load_trade_info_from_executed_file(
                user_path=(pathlib.Path(path) / user_id), trade_date=trade_date
            )
            score_series = load_score_series((pathlib.Path(path) / user_id), trade_date)
            update_account(user.account, trade_info, trade_exchange, trade_date)

            report = user.account.report.generate_report_dataframe()
            self.logger.info(report)
            um.save_user_data(user_id)
            self.logger.info("Update account state {} for {}".format(trade_date, user_id))

    def simulate(self, id, config, exchange_config, start, end, path, bench="SH000905"):
        """Run the ( generate_order_list -> execute_order_list -> update_account) process everyday
            from start date to end date.

        Parameters
        ----------
        id : str
            user id, need to be unique
        config : str
            The file path (yaml) of user config
        exchange_config: str
            The file path (yaml) of exchange config
        start : str "YYYY-MM-DD"
            The start date to run the online simulate
        end : str "YYYY-MM-DD"
            The end date to run the online simulate
        path : str
            Path to save user account.
        bench : str
            The benchmark that our result compared with.
            'SH000905' for csi500, 'SH000300' for csi300
        """
        # Clear the current user if exists, then add a new user.
        create_user_folder(path)
        um = self.init(self.client, path, None)[0]
        start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
        try:
            um.remove_user(user_id=id)
        except BaseException:
            pass
        um.add_user(user_id=id, config_file=config, add_date=pd.Timestamp(start_date))

        # Do the online simulate
        um.load_users()
        user = um.users[id]
        dates, trade_exchange = prepare(um, end_date, id, exchange_config)
        executor = SimulatorExecutor(trade_exchange=trade_exchange)
        for pred_date, trade_date in zip(dates[:-2], dates[1:-1]):
            user_path = pathlib.Path(path) / id

            # 1. load and save score_series
            input_data = user.model.get_data_with_date(pred_date)
            score_series = user.model.predict(input_data)
            save_score_series(score_series, (pathlib.Path(path) / id), trade_date)

            # 2. update strategy (and model)
            user.strategy.update(score_series, pred_date, trade_date)

            # 3. generate and save order list
            order_list = user.strategy.generate_order_list(
                score_series=score_series,
                current=user.account.current,
                trade_exchange=trade_exchange,
                trade_date=trade_date,
            )
            save_order_list(order_list=order_list, user_path=user_path, trade_date=trade_date)

            # 4. auto execute order list
            order_list = load_order_list(user_path=user_path, trade_date=trade_date)
            trade_info = executor.execute(trade_account=user.account, order_list=order_list, trade_date=trade_date)
            executor.save_executed_file_from_trade_info(
                trade_info=trade_info, user_path=user_path, trade_date=trade_date
            )
            # 5. update account state
            trade_info = executor.load_trade_info_from_executed_file(user_path=user_path, trade_date=trade_date)
            update_account(user.account, trade_info, trade_exchange, trade_date)
        report = user.account.report.generate_report_dataframe()
        self.logger.info(report)
        um.save_user_data(id)
        self.show(id, path, bench)

    def show(self, id, path, bench="SH000905"):
        """show the newly report (mean, std, information_ratio, annualized_return)

        Parameters
        ----------
        id : str
            user id, need to be unique
        path : str
            Path to save user account.
        bench : str
            The benchmark that our result compared with.
            'SH000905' for csi500, 'SH000300' for csi300
        """
        um = self.init(self.client, path, None)[0]
        if id not in um.users:
            raise ValueError("Cannot find user ".format(id))
        bench = D.features([bench], ["$change"]).loc[bench, "$change"]
        report = um.users[id].account.report.generate_report_dataframe()
        report["bench"] = bench
        analysis_result = {}
        r = (report["return"] - report["bench"]).dropna()
        analysis_result["excess_return_without_cost"] = risk_analysis(r)
        r = (report["return"] - report["bench"] - report["cost"]).dropna()
        analysis_result["excess_return_with_cost"] = risk_analysis(r)
        print("Result:")
        print("excess_return_without_cost:")
        print(analysis_result["excess_return_without_cost"])
        print("excess_return_with_cost:")
        print(analysis_result["excess_return_with_cost"])


def run():
    fire.Fire(Operator)


if __name__ == "__main__":
    run()

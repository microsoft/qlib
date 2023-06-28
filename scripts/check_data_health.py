import enum
import logging
import os
from typing import List, Optional, Dict, Tuple

import fire
import pandas as pd
from tqdm import tqdm


class DataHealthChecker:
    """Checks a dataset for data completeness and correctness. The data will be converted to a pd.DataFrame and checked for the following problems:
    - any of the columns ["open", "high", "low", "close", "volume"] are missing
    - any data is missing
    - any step change in the OHLCV columns is above a threshold (default: 0.5 for price, 3 for volume)
    - any factor is missing
    """

    class DataProblem(enum.Enum):
        MISSING_REQUIRED_COLUMN = 1
        MISSING_DATA = 2
        LARGE_STEP_CHANGE = 3
        MISSING_FACTOR = 4

    def __init__(
        self,
        csv_path=None,
        qlib_dir=None,
        large_step_threshold_price=0.5,
        large_step_threshold_volume=3,
    ):
        assert csv_path or qlib_dir, "One of csv_path or qlib_dir should be provided."
        assert not (
            csv_path and qlib_dir
        ), "Only one of csv_path or qlib_dir should be provided."

        self.data = {}
        self.problems = {}
        self.large_step_threshold_price = large_step_threshold_price
        self.large_step_threshold_volume = large_step_threshold_volume

        if csv_path:
            assert os.path.isdir(csv_path), f"{csv_path} should be a directory."
            files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
            for filename in tqdm(files, desc="Loading data"):
                df = pd.read_csv(os.path.join(csv_path, filename))
                self.data[filename] = df

        elif qlib_dir:
            # todo: add support for qlib_dir
            pass

    def check_missing_data(
        self, filename: str, df: pd.DataFrame
    ) -> Optional[Tuple[DataProblem, List[str]]]:
        """Check if any data is missing in the DataFrame."""
        if df.isnull().values.any():
            missing_data_columns = (
                df.isnull().sum()[df.isnull().sum() > 0].index.tolist()
            )
            logging.warning(
                f"{filename}: Missing data in columns {missing_data_columns}."
            )
            return self.DataProblem.MISSING_DATA, missing_data_columns

    def check_large_step_changes(
        self, filename: str, df: pd.DataFrame
    ) -> Optional[Tuple[DataProblem, List[str]]]:
        """Check if there are any large step changes above the threshold in the OHLCV columns."""
        affected_columns = []
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                pct_change = df[col].pct_change().abs()
                threshold = (
                    self.large_step_threshold_volume
                    if col == "volume"
                    else self.large_step_threshold_price
                )
                if pct_change.max() > threshold:
                    large_steps = pct_change[pct_change > threshold]
                    logging.warning(
                        f"{filename}: Relative step changes above threshold {threshold} in column '{col}' at indices {large_steps.index.tolist()}."
                    )
                    affected_columns.append(col)
        if affected_columns:
            return self.DataProblem.LARGE_STEP_CHANGE, affected_columns

    def check_required_columns(
        self, filename: str, df: pd.DataFrame
    ) -> Optional[Tuple[DataProblem, List[str]]]:
        """Check if any of the required columns (OLHCV) are missing in the DataFrame."""
        required_columns = ["open", "high", "low", "close", "volume"]
        if not all(column in df.columns for column in required_columns):
            missing_required_columns = [
                column for column in required_columns if column not in df.columns
            ]
            logging.error(
                f"{filename}: Missing columns {missing_required_columns} of required columns {required_columns}."
            )
            return self.DataProblem.MISSING_REQUIRED_COLUMN, missing_required_columns

    def check_missing_factor(
        self, filename: str, df: pd.DataFrame
    ) -> Optional[Tuple[DataProblem, List[str]]]:
        # todo
        pass

    def check_data(self):
        checks = [
            self.check_missing_data,
            self.check_large_step_changes,
            self.check_required_columns,
            self.check_missing_factor,
        ]
        for filename, df in self.data.items():
            for check in checks:
                problem = check(filename, df)
                if problem:
                    self.problems.setdefault(filename, []).append(problem)
        self._print_report(self.problems)

    def _print_report(self, problems: Dict[str, List[Tuple[DataProblem, str]]]):
        """Count the number of problems for each type and print the report together with the affected columns."""
        if problems:
            problem_stats_by_type = {}
            for _, problem_tuples in problems.items():
                for name, affected_columns in problem_tuples:
                    stats = problem_stats_by_type.setdefault(
                        name, {"count": 0, "affected_columns": set()}
                    )
                    stats["count"] += 1
                    stats["affected_columns"].update(affected_columns)
            print("\n-----------------------------")
            print("Summary of data health check:")
            print(f"Files checked: {len(self.data)}")
            padding = max(len(problem.name) for problem in self.DataProblem)
            for problem in self.DataProblem:
                padded_name = problem.name.ljust(padding + 2, " ")
                print(f"• {padded_name}", end="")
                if problem in problem_stats_by_type:
                    print(f"{problem_stats_by_type[problem]['count']}")
                    print(
                        f"  affected columns{' ' * max(padding - 14, 0)}{problem_stats_by_type[problem]['affected_columns']}"
                    )
                else:
                    print("0")
        else:
            logging.info("Data check passed. No problems found.")


if __name__ == "__main__":
    fire.Fire(DataHealthChecker)

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Default configs for TFT experiments.

Contains the default output paths for data, serialised models and predictions
for the main experiments used in the publication.
"""

import os

import data_formatters.qlib_Alpha158


class ExperimentConfig:
    """Defines experiment configs and paths to outputs.

    Attributes:
      root_folder: Root folder to contain all experimental outputs.
      experiment: Name of experiment to run.
      data_folder: Folder to store data for experiment.
      model_folder: Folder to store serialised models.
      results_folder: Folder to store results.
      data_csv_path: Path to primary data csv file used in experiment.
      hyperparam_iterations: Default number of random search iterations for
        experiment.
    """

    default_experiments = ["Alpha158"]

    def __init__(self, experiment="volatility", root_folder=None):
        """Creates configs based on default experiment chosen.

        Args:
          experiment: Name of experiment.
          root_folder: Root folder to save all outputs of training.
        """

        if experiment not in self.default_experiments:
            raise ValueError("Unrecognised experiment={}".format(experiment))

        # Defines all relevant paths
        if root_folder is None:
            root_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "outputs")
            print("Using root folder {}".format(root_folder))

        self.root_folder = root_folder
        self.experiment = experiment
        self.data_folder = os.path.join(root_folder, "data", experiment)
        self.model_folder = os.path.join(root_folder, "saved_models", experiment)
        self.results_folder = os.path.join(root_folder, "results", experiment)

        # Creates folders if they don't exist
        for relevant_directory in [self.root_folder, self.data_folder, self.model_folder, self.results_folder]:
            if not os.path.exists(relevant_directory):
                os.makedirs(relevant_directory)

    @property
    def data_csv_path(self):
        csv_map = {
            "Alpha158": "Alpha158.csv",
        }

        return os.path.join(self.data_folder, csv_map[self.experiment])

    @property
    def hyperparam_iterations(self):

        return 240 if self.experiment == "volatility" else 60

    def make_data_formatter(self):
        """Gets a data formatter object for experiment.

        Returns:
          Default DataFormatter per experiment.
        """

        data_formatter_class = {
            "Alpha158": data_formatters.qlib_Alpha158.Alpha158Formatter,
        }

        return data_formatter_class[self.experiment]()

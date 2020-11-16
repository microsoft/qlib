# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
import pickle


class Serializable:
    """
    Serializable behaves like pickle.
    But it only saves the state whose name **does not** start with `_`
    """

    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def __setstate__(self, state: dict):
        self.__dict__.update(state)

    def to_pickle(self, path: [Path, str]):
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

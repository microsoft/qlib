# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Group can group a set of object based on `group_func` and change them to a dict.
"""

from qlib.model.ens.ensemble import Ensemble, RollingEnsemble
from typing import Callable, Union
from joblib import Parallel, delayed


class Group:
    """Group the objects based on dict"""

    def __init__(self, group_func=None, ens: Ensemble = None):
        """init Group.

        Args:
            group_func (Callable, optional): Given a dict and return the group key and one of group elements.

                For example: {(A,B,C1): object, (A,B,C2): object} -> {(A,B): {C1: object, C2: object}}

            Defaults to None.

            ens (Ensemble, optional): If not None, do ensemble for grouped value after grouping.
        """
        self._group_func = group_func
        self._ens_func = ens

    def group(self, *args, **kwargs) -> dict:
        # TODO: such design is weird when `_group_func` is the only configurable part in the class
        if isinstance(getattr(self, "_group_func", None), Callable):
            return self._group_func(*args, **kwargs)
        else:
            raise NotImplementedError(f"Please specify valid `group_func`.")

    def reduce(self, *args, **kwargs) -> dict:
        if isinstance(getattr(self, "_ens_func", None), Callable):
            return self._ens_func(*args, **kwargs)
        else:
            raise NotImplementedError(f"Please specify valid `_ens_func`.")

    def __call__(self, ungrouped_dict: dict, n_jobs=1, verbose=0, *args, **kwargs) -> dict:
        """Group the ungrouped_dict into different groups.

        Args:
            ungrouped_dict (dict): the ungrouped dict waiting for grouping like {name: things}

        Returns:
            dict: grouped_dict like {G1: object, G2: object}
        """

        # NOTE: The multiprocessing will raise error if you use `Serializable`
        # Because the `Serializable` will affect the behaviours of pickle
        grouped_dict = self.group(ungrouped_dict, *args, **kwargs)

        key_l = []
        job_l = []
        for key, value in grouped_dict.items():
            key_l.append(key)
            job_l.append(delayed(Group.reduce)(self, value))
        return dict(zip(key_l, Parallel(n_jobs=n_jobs, verbose=verbose)(job_l)))


class RollingGroup(Group):
    """group the rolling dict"""

    def group(self, rolling_dict: dict) -> dict:
        """Given an rolling dict likes {(A,B,R): things}, return the grouped dict likes {(A,B): {R:things}}

        NOTE: There is a assumption which is the rolling key is at the end of key tuple, because the rolling results always need to be ensemble firstly.

        Args:
            rolling_dict (dict): an rolling dict. If the key is not a tuple, then do nothing.

        Returns:
            dict: grouped dict
        """
        grouped_dict = {}
        for key, values in rolling_dict.items():
            if isinstance(key, tuple):
                grouped_dict.setdefault(key[:-1], {})[key[-1]] = values
        return grouped_dict

    def __init__(self):
        super().__init__(ens=RollingEnsemble())

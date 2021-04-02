from qlib.model.ens.ensemble import Ensemble, RollingEnsemble
from typing import Callable, Union


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
        self._group = group_func
        self._ens = ens

    def __call__(self, ungrouped_dict: dict, *args, **kwargs):
        """Group the ungrouped_dict into different groups.

        Args:
            ungrouped_dict (dict): the ungrouped dict waiting for grouping like {name: things}

        Returns:
            dict: grouped_dict like {G1: object, G2: object}
        """
        if isinstance(getattr(self, "_group", None), Callable):
            grouped_dict = self._group(ungrouped_dict, *args, **kwargs)
            if self._ens is not None:
                ens_dict = {}
                for key, value in grouped_dict.items():
                    ens_dict[key] = self._ens(value)
                grouped_dict = ens_dict
            return grouped_dict
        else:
            raise NotImplementedError(f"Please specify valid merge_func.")


class RollingGroup(Group):
    """group the rolling dict"""

    @staticmethod
    def rolling_group(rolling_dict: dict):
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

    def __init__(self, group_func=None, ens: Ensemble = RollingEnsemble()):
        super().__init__(group_func=group_func, ens=ens)
        if group_func is None:
            self._group = RollingGroup.rolling_group
import pandas as pd
import numpy as np
from gym.spaces import Discrete, Box, Tuple, MultiDiscrete
import math
import json

from .obs_rule import RuleObs


class TeacherObs(RuleObs):
    """
    The Observation used for OPD method.

    Consist of public state(raw feature), private state, seqlen

    """

    def get_obs(
        self, raw_df, feature_dfs, t, interval, position, target, is_buy, max_step_num, interval_num, *args, **kargs,
    ):
        if t == -1:
            self.private_states = []
        public_state = self.get_feature_res(feature_dfs, t, interval, whole_day=True)
        private_state = np.array([position / target, (t + 1) / max_step_num])
        self.private_states.append(private_state)
        list_private_state = np.concatenate(self.private_states)
        list_private_state = np.concatenate(
            (list_private_state, [0.0] * 2 * (interval_num + 1 - len(self.private_states)),)
        )
        seqlen = np.array([interval])
        assert not (
            np.isnan(list_private_state).any() | np.isinf(list_private_state).any()
        ), f"{private_state}, {target}"
        assert not (np.isnan(public_state).any() | np.isinf(public_state).any()), f"{public_state}"
        return np.concatenate((public_state, list_private_state, seqlen))


class RuleTeacher(RuleObs):
    """ """

    def get_obs(
        self, raw_df, feature_dfs, t, interval, position, target, is_buy, max_step_num, interval_num, *args, **kargs,
    ):
        if t == -1:
            self.private_states = []
        public_state = feature_dfs[0].reshape(-1)[: 6 * 240]
        private_state = np.array([position / target, (t + 1) / max_step_num])
        teacher_action = self.get_feature_res(feature_dfs, t, interval)[-self.features[1]["size"] :]
        self.private_states.append(private_state)
        list_private_state = np.concatenate(self.private_states)
        list_private_state = np.concatenate(
            (list_private_state, [0.0] * 2 * (interval_num + 1 - len(self.private_states)),)
        )
        seqlen = np.array([interval])
        return np.concatenate((teacher_action, public_state, list_private_state, seqlen))

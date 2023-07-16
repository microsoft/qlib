from collections import defaultdict
from dataclasses import dataclass, field
import copy
from pathlib import Path
from typing import List, Optional, Union
from qlib.finco.log import FinCoLog
from qlib.typehint import Literal

from qlib.finco.utils import similarity


@dataclass
class Design:
    plan: str
    classes: str
    decision: str


@dataclass
class Exp:
    """Experiment"""
    # compoments
    dataset: Optional[Design] = None
    datahandler: Optional[Design] = None
    model: Optional[Design] = None
    record: Optional[Design] = None
    strategy: Optional[Design] = None
    backtest: Optional[Design] = None

    # basic
    template: Optional[Path] = None

    # rolling strategy. None indicates no rolling
    rolling: Optional[Literal["base", "ddgda"]] = None


@dataclass
class StructContext:
    """Part of the context have clear meaning and structure, so they will be saved here and can be easily retrieved and understood"""
    # TODO: move more content in WorkflowContextManager.context to here
    workspace: Path
    exp_list: List[Exp] = field(default_factory=list)  # the planned experiments


class WorkflowContextManager:
    """Context Manager stores the context of the workflow"""
    """All context are key value pairs which saves the input, output and status of the whole workflow"""

    def __init__(self, workspace: Path) -> None:
        self.context = {}
        self.logger = FinCoLog()
        # this context is public
        self.struct_context = StructContext(workspace)  # TODO: move more content in context to here
        self.set_context("workspace", workspace)  # TODO: remove me

    def set_context(self, key, value):
        if key in self.context:
            self.logger.warning("The key already exists in the context, the value will be overwritten")
        self.context[key] = value

    def get_context(self, key):
        # NOTE: if the key doesn't exist, return None. In the future, we may raise an error to detect abnormal behavior
        if key not in self.context:
            self.logger.warning("The key doesn't exist in the context")
            return None
        return self.context[key]

    def update_context(self, key, new_value):
        # NOTE: if the key doesn't exist, return None. In the future, we may raise an error to detect abnormal behavior
        if key not in self.context:
            self.logger.warning("The key doesn't exist in the context")
        self.context.update({key: new_value})

    def get_all_context(self):
        """return a deep copy of the context"""
        """TODO: do we need to return a deep copy?"""
        return copy.deepcopy(self.context)

    def retrieve(self, query: str) -> dict:
        if query in self.context.keys():
            return {query: self.context.get(query)}

        # Note: retrieve information from context by string similarity maybe abandon in future
        scores = {}
        for k, v in self.context.items():
            scores.update({k: max(similarity(query, k), similarity(query, v))})
        max_score_key = max(scores, key=scores.get)
        return {max_score_key: self.context.get(max_score_key)}

    def clear(self, reserve: list = None):
        if reserve is None:
            reserve = []

        _context = {k: self.get_context(k) for k in reserve}
        self.context = _context

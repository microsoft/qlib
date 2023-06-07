import sys
import copy
from pathlib import Path
import shutil

from qlib.log import get_module_logger
from qlib.finco.conf import Config
from qlib.finco.utils import parse_json
from qlib.finco.task import WorkflowTask, PlanTask, ActionTask, SummarizeTask, RecorderTask


class WorkflowContextManager:
    """Context Manager stores the context of the workflow"""

    """All context are key value pairs which saves the input, output and status of the whole workflow"""

    def __init__(self) -> None:
        self.context = {}
        self.logger = get_module_logger("fincoWorkflowContextManager")

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


class WorkflowManager:
    """This manange the whole task automation workflow including tasks and actions"""

    def __init__(self, workspace=None) -> None:
        if workspace is None:
            self._workspace = Path.cwd() / "finco_workspace"
        else:
            self._workspace = Path(workspace)
        self._confirm_and_rm()
        self._context = WorkflowContextManager()
        self._context.set_context("workspace", self._workspace)
        self.default_user_prompt = "Please help me build a low turnover strategy that focus more on longterm return in China a stock market. I want to construct a new dataset covers longer history"

    def _confirm_and_rm(self):
        # if workspace exists, please confirm and remove it. Otherwise exit.
        if self._workspace.exists():
            flag = input(
                f"Will be deleted: "
                f"\n\t{self._workspace}"
                f"\nIf you do not need to delete {self._workspace}, please change the workspace dir or rename existing files "
                f"\nAre you sure you want to delete, yes(Y/y), no (N/n):"
            )
            if str(flag) not in ["Y", "y"]:
                sys.exit()
            else:
                # remove self._workspace
                shutil.rmtree(self._workspace)

    def set_context(self, key, value):
        """Direct call set_context method of the context manager"""
        self._context.set_context(key, value)

    def get_context(self) -> WorkflowContextManager:
        return self._context

    def run(self, prompt: str) -> Path:
        """
        The workflow manager is supposed to generate a codebase based on the prompt

        Parameters
        ----------
        prompt: str
            the prompt user gives

        Returns
        -------
        Path
            The workflow manager is expected to produce output that includes a codebase containing generated code, results, and reports in a designated location.
            The path is returned

            The output path should follow a specific format:
            - TODO: design
              There is a summarized report where user can start from.
        """

        # NOTE: The following items are not designed to make the workflow very flexible.
        # - The generated tasks can't be changed after geting new information from the execution retuls.
        #   - But it is required in some cases, if we want to build a external dataset, it maybe have to plan like autogpt...

        cfg = Config()

        # NOTE: default user prompt might be changed in the future and exposed to the user
        if prompt is None:
            self.set_context("user_prompt", self.default_user_prompt)
        else:
            self.set_context("user_prompt", prompt)

        # NOTE: list may not be enough for general task list
        task_list = [WorkflowTask(), RecorderTask(), SummarizeTask()]
        while len(task_list):
            # task list is not long, so sort it is not a big problem
            # TODO: sort the task list based on the priority of the task
            # task_list = sorted(task_list, key=lambda x: x.task_type)
            t = task_list.pop(0)
            t.assign_context_manager(self._context)
            res = t.execute()
            if not cfg.continous_mode:
                res = t.interact()
            t.summarize()
            if isinstance(t, (WorkflowTask, PlanTask, ActionTask, SummarizeTask)):
                task_list = res + task_list
            else:
                raise NotImplementedError(f"Unsupported Task type {t}")
        return self._workspace

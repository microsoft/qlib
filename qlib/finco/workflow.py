import sys
import copy
import shutil
from pathlib import Path
from typing import List

from qlib.finco.task import HighLevelPlanTask, SummarizeTask, TrainTask
from qlib.finco.prompt_template import PromptTemplate, Template
from qlib.finco.log import FinCoLog, LogColors
from qlib.finco.utils import similarity
from qlib.finco.llm import APIBackend
from qlib.finco.conf import Config
from qlib.finco.knowledge import KnowledgeBase, Topic


class WorkflowContextManager:
    """Context Manager stores the context of the workflow"""

    """All context are key value pairs which saves the input, output and status of the whole workflow"""

    def __init__(self) -> None:
        self.context = {}
        self.logger = FinCoLog()

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


class WorkflowManager:
    """This manage the whole task automation workflow including tasks and actions"""

    def __init__(self, workspace=None) -> None:
        self.logger = FinCoLog()

        if workspace is None:
            self._workspace = Path.cwd() / "finco_workspace"
        else:
            self._workspace = Path(workspace)
        self.conf = Config()
        self._confirm_and_rm()

        self.prompt_template = PromptTemplate()
        self.context = WorkflowContextManager()
        self.context.set_context("workspace", self._workspace)
        self.default_user_prompt = "Please help me build a low turnover strategy that focus more on longterm return in China A csi300. Please help to use lightgbm model."

    def _confirm_and_rm(self):
        # if workspace exists, please confirm and remove it. Otherwise exit.
        if self._workspace.exists() and not self.conf.continuous_mode:
            self.logger.info(title="Interact")
            flag = input(
                LogColors().render(
                    f"Will be deleted: \n\t{self._workspace}\n"
                    f"If you do not need to delete {self._workspace},"
                    f" please change the workspace dir or rename existing files\n"
                    f"Are you sure you want to delete, yes(Y/y), no (N/n):",
                    color=LogColors.WHITE)
            )
            if str(flag) not in ["Y", "y"]:
                sys.exit()
            else:
                # remove self._workspace
                shutil.rmtree(self._workspace)
        elif self._workspace.exists() and self.conf.continuous_mode:
            shutil.rmtree(self._workspace)

    def set_context(self, key, value):
        """Direct call set_context method of the context manager"""
        self.context.set_context(key, value)

    def get_context(self) -> WorkflowContextManager:
        return self.context

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

        # NOTE: default user prompt might be changed in the future and exposed to the user
        if prompt is None:
            self.set_context("user_prompt", self.default_user_prompt)
        else:
            self.set_context("user_prompt", prompt)
        self.logger.info(f"user_prompt: {self.get_context().get_context('user_prompt')}", title="Start")

        # NOTE: list may not be enough for general task list
        task_list = [HighLevelPlanTask(), SummarizeTask()]
        task_finished = []
        while len(task_list):
            task_list_info = [str(task) for task in task_list]

            # task list is not long, so sort it is not a big problem
            # TODO: sort the task list based on the priority of the task
            # task_list = sorted(task_list, key=lambda x: x.task_type)
            t = task_list.pop(0)
            self.logger.info(f"Task finished: {[str(task) for task in task_finished]}",
                             f"Task in queue: {task_list_info}",
                             f"Executing task: {str(t)}",
                             title="Task")

            t.assign_context_manager(self.context)
            res = t.execute()
            t.summarize()
            task_finished.append(t)
            self.context.set_context("task_finished", task_finished)
            self.logger.plain_info(f"{str(t)} finished.\n\n\n")

            task_list = res + task_list

        return self._workspace


class LearnManager:
    __DEFAULT_TOPICS = ["IC", "MaxDropDown"]

    def __init__(self):
        self.epoch = 0
        self.wm = WorkflowManager()

        self.topics = [Topic(name=topic, describe=self.wm.prompt_template.get(f"Topic_{topic}")) for topic in
                       self.__DEFAULT_TOPICS]
        self.knowledge_base = KnowledgeBase(workdir=Path.cwd().joinpath('knowledge'))
        self.knowledge_base.execute_knowledge.add([])
        self.knowledge_base.query(knowledge_type="infrastructure", content="resolve_path")

    def run(self, prompt):
        # todo: add early stop condition
        for i in range(10):
            self.wm.run(prompt)
            self.learn()
            self.epoch += 1

    def learn(self):
        workspace = self.wm.context.get_context("workspace")

        def _drop_duplicate_task(_task: List):
            unique_task = {}
            for obj in _task:
                task_name = obj.__class__.__name__
                if task_name not in unique_task:
                    unique_task[task_name] = obj
            return list(unique_task.values())

        # one task maybe run several times in workflow
        task_finished = _drop_duplicate_task(self.wm.context.get_context("task_finished"))

        user_prompt = self.wm.context.get_context("user_prompt")
        summary = self.wm.context.get_context("summary")

        [topic.summarize(self.knowledge_base.get_knowledge()) for topic in self.topics]
        knowledge_of_topics = [{topic.name: topic.knowledge} for topic in self.topics]

        for task in task_finished:
            prompt_workflow_selection = self.wm.prompt_template.get(f"{self.__class__.__name__}_user").render(
                summary=summary, brief=knowledge_of_topics,
                task_finished=[str(t) for t in task_finished],
                task=task.__class__.__name__, system=task.system.render(), user_prompt=user_prompt
            )

            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=prompt_workflow_selection,
                system_prompt=self.wm.prompt_template.get(f"{self.__class__.__name__}_system").render()
            )

            # todo: response assertion
            task.prompt_template.update(key=f"{task.__class__.__name__}_system", value=Template(response))

        self.wm.prompt_template.save(Path.joinpath(workspace, f"prompts/checkpoint_{self.epoch}.yml"))
        self.wm.context.clear(reserve=["workspace"])

import sys
import shutil
from typing import List

from pathlib import Path

from qlib.finco.task import IdeaTask, SummarizeTask
from qlib.finco.prompt_template import PromptTemplate, Template
from qlib.finco.log import FinCoLog, LogColors
from qlib.finco.llm import APIBackend
from qlib.finco.conf import Config
from qlib.finco.knowledge import KnowledgeBase, Topic
from qlib.finco.context import WorkflowContextManager


# TODO: it is not necessary in current phase
# class TaskDAG:
#     """
#     This is a Task manager. it maintains a graph and a stack stucture to manager the task
#     The reason why the DGA relationship is maintained outside instead of inside the task is that
#     - To make the creating of task simpler(user don't have to care about the relation-ship)
#     - To manage the relation ship when poping and executing the tasks is relatively easier instead of scattering them everywhere
#     """
#     def __init__(self) -> None:
#         self._finished = []
#         self._stack = []
#         self._dag = defaultdict(list)  # from id(object) -> list of id(object)
#
#     def pop(self):
#         return  self._stack.pop(0)
#
#     def push(self, task: Union[Task, List[Task]], parent: Optional[Task] = None):
#         if isinstance(task, Task):
#             task = [task]
#         if parent is not None:
#             self._dag
#
#     def done(self) -> bool:
#         return len(self._stack) == 0


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
        self.context = WorkflowContextManager(workspace=self._workspace)
        self.context.set_context("workspace", self._workspace)
        self.default_user_prompt = "build an A-share stock market daily portfolio in quantitative investment and minimize the maximum drawdown."

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
                    color=LogColors.WHITE,
                )
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
            self.set_context("user_intention", self.default_user_prompt)
        else:
            self.set_context("user_intention", prompt)
        self.logger.info(f"user_intention: {self.get_context().get_context('user_intention')}", title="Start")

        # NOTE: list may not be enough for general task list
        task_list = [IdeaTask(), SummarizeTask()]
        task_finished = []
        while len(task_list):
            task_list_info = [str(task) for task in task_list]

            # task list is not long, so sort it is not a big problem
            # TODO: sort the task list based on the priority of the task
            # task_list = sorted(task_list, key=lambda x: x.task_type)
            t = task_list.pop(0)
            self.logger.info(
                f"Task finished: {[str(task) for task in task_finished]}",
                f"Task in queue: {task_list_info}",
                f"Executing task: {str(t)}",
                title="Task",
            )

            t.assign_context_manager(self.context)
            res = t.execute()
            t.summarize()
            task_finished.append(t)
            self.context.set_context("task_finished", task_finished)
            self.logger.plain_info(f"{str(t)} finished.\n\n\n")

            task_list = res + task_list

        return self._workspace


class LearnManager:
    __DEFAULT_TOPICS = ["IC", "MaxDropDown", "RollingModel"]

    def __init__(self):
        self.epoch = 0
        self.wm = WorkflowManager()

        self.topics = [
            Topic(name=topic, describe=self.wm.prompt_template.get(f"Topic_{topic}")) for topic in self.__DEFAULT_TOPICS
        ]
        self.knowledge_base = KnowledgeBase()

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

        user_intention = self.wm.context.get_context("user_intention")
        summary = self.wm.context.get_context("summary")

        [topic.summarize(self.knowledge_base.practice_knowledge.knowledge[-2:]) for topic in self.topics]
        [self.knowledge_base.practice_knowledge.add([{"practice_knowledge": topic.knowledge}]) for topic in self.topics]
        knowledge_of_topics = [{topic.name: topic.knowledge} for topic in self.topics]

        for task in task_finished:
            prompt_workflow_selection = self.wm.prompt_template.get(f"{self.__class__.__name__}_user").render(
                summary=summary,
                brief=knowledge_of_topics,
                task_finished=[str(t) for t in task_finished],
                task=task.__class__.__name__, system=task.system.render(), user_intention=user_intention
            )

            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=prompt_workflow_selection,
                system_prompt=self.wm.prompt_template.get(f"{self.__class__.__name__}_system").render(),
            )

            # todo: response assertion
            task.prompt_template.update(key=f"{task.__class__.__name__}_system", value=Template(response))

        self.wm.prompt_template.save(Path.joinpath(workspace, f"prompts/checkpoint_{self.epoch}.yml"))
        self.wm.context.clear(reserve=["workspace"])

import os

from pathlib import Path
from typing import Any, List
from qlib.typehint import Literal


class Task:
    """
    The user's intention, which was initially represented by a prompt, is achieved through a sequence of tasks.

    Some thoughts:
    - Do we have to split create a new concept of Action besides Task?
        - Most actions directly modify the disk, with their interfaces taking in and outputting text. The LLM's interface similarly takes in and outputs text.
        - Some actions will run some commands.

    Maybe we can just categorizing tasks by following?
    - Planning task (it is at a high level and difficult to execute directly; therefore, it should be further divided):
    - Action Task
        - CMD Task: it is expected to run a cmd
        - Edit Task: it is supposed to edit the code base directly.
    """

    def __init__(self, context=None) -> None:
        pass

    def summarize(self) -> str:
        """After the execution of the task, it is supposed to generated some context about the execution"""
        return ""

    def update_context(self, latest_context):
        ...

    def execution(self) -> Any:
        """The execution results of the task"""
        pass


class PlanTask(Task):
    def execute(self) -> List[Task]:
        return []


class WorkflowTask(PlanTask):
    """make the choice which main workflow (RL, SL) will be used"""

    def execute(self):
        ...


class SLTask(PlanTask):
    def exeute(self):
        """
        return a list of interested tasks
        Copy the template project maybe a part of the task
        """
        return []


class ActionTask(Task):
    def execute(self) -> Literal["fail", "success"]:
        return "success"


class SummarizeTask(Task):
    def execution(self) -> Any:
        output_path = ''

    def parse2txt(self, path) -> List:
        file_list = []
        path = Path.cwd().joinpath(path)
        for root, dirs, files in os.walk(path):
            for filename in files:
                file_path = os.path.join(root, filename)
                print(file_path)
                file_list.append(file_path)

        result = []
        for file in file_list:
            postfix = file.split('.')[-1]
            if postfix in ['txt', 'py', 'log']:
                with open(file) as f:
                    content = f.read()
                    print(content)
                    result.append({'postfix': postfix, 'content': content})
        return result


class WorkflowManager:
    """This manange the whole task automation workflow including tasks and actions"""

    def __init__(self, name="project", output_path=None) -> None:

        if output_path is None:
            self._output_path = Path.cwd() / name
        else:
            self._output_path = Path(output_path)
        self._context = []

    def add_context(self, task_res):
        self._context.append(task_res)

    def get_context(self):
        """TODO: context manger?"""

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

        # NOTE: list may not be enough for general task list
        task_list = [WorkflowTask(prompt)]
        while len(task_list):
            # task_list.ap
            t = task_list.pop(0)
            t.update_context(self.get_context())
            res = t.execute()
            if isinstance(t, PlanTask):
                task_list.extend(res)
            elif isinstance(t, ActionTask):
                if res != "success":
                    ...
                    # TODO: handle the unexpected execution Error
            else:
                raise NotImplementedError("Unsupported action type")
            self.add_context(t.summarize())
        return self._output_path

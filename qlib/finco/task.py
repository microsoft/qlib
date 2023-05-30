from pathlib import Path
from typing import Any, List
from qlib.log import get_module_logger
from qlib.typehint import Literal
from qlib.finco.conf import Config
from qlib.finco.llm import try_create_chat_completion
from qlib.finco.utils import parse_json
from jinja2 import Template

import abc
import copy
import logging


class Task():
    """
    The user's intention, which was initially represented by a prompt, is achieved through a sequence of tasks.
    This class doesn't have to be abstract, but it is abstract in the sense that it is not supposed to be instantiated directly because it doesn't have any implementation.

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

    
    TASK_TYPE_WORKFLOW = 0
    TASK_TYPE_PLAN = 1
    TASK_TYPE_ACTION = 2
    TASK_TYPE_SUMMARIZE = 3

    ## all subclass should implement this method to determine task type
    @abc.abstractclassmethod
    def __init__(self, task_type) -> None:
        self.task_type = task_type
        self._context_manager = None
        self.executed = False
    
    def summarize(self) -> str:
        """After the execution of the task, it is supposed to generated some context about the execution"""
        raise NotImplementedError

    """assign the workflow context manager to the task"""
    """then all tasks can use this context manager to share the same context"""
    def assign_context_manager(self, context_manager):
        ...
        self._context_manager = context_manager

    def execution(self, **kwargs) -> Any:
        """The execution results of the task"""
        raise NotImplementedError

    def interact(self) -> Any:
        """The user can interact with the task"""
        """All sub classes should implement the interact method to determine the next task"""
        """In continous mode, this method will not be called and the next task will be determined by the execution method only"""
        raise NotImplementedError("The interact method is not implemented, but workflow not in continous mode")

class WorkflowTask(Task):
    """This task is supposed to be the first task of the workflow"""
    def __init__(self,) -> None:
        super().__init__(Task.TASK_TYPE_WORKFLOW)
        self.__DEFAULT_WORKFLOW_SYSTEM_PROMPT = """
        Your task is to determine the workflow in Qlib (supervised learning or reinforcemtn learning) ensureing the workflow can meet the user's requirements.

        The user will provide the requirements, you will provide only the output the choice in exact format specified below with no explanation or conversation.

        Example input 1:
        Help me build a build a low turnover quant investment strategy that focus more on long turn return in China a stock market.

        Example output 1:
        workflow: supervised learning

        Example input 2:
        Help me build a build a pipeline to determine the best selling point of a stock in a day or half a day in USA stock market.

        Example output 2:
        workflow: reinforcemtn learning
        """

        self.__DEFAULT_WORKFLOW_USER_PROMPT = (
            "User input: '{{user_prompt}}'\n"
            "Please provide the workflow in Qlib (supervised learning or reinforcemtn learning) ensureing the workflow can meet the user's requirements.\n"
            "Response only with the output in the exact format specified in the system prompt, with no explanation or conversation.\n"
        )
        self.__DEFAULT_USER_PROMPT = "Please help me build a low turnover strategy that focus more on longterm return in China a stock market."
        self.logger = get_module_logger("fincoWorkflowTask", level=logging.INFO)

    """make the choice which main workflow (RL, SL) will be used"""
    def execute(self,) -> List[Task]:
        user_prompt = self._context_manager.get_context("user_prompt")
        user_prompt = user_prompt if user_prompt is not None else self.__DEFAULT_USER_PROMPT
        system_prompt = self.__DEFAULT_WORKFLOW_SYSTEM_PROMPT
        prompt_workflow_selection = Template(
            self.__DEFAULT_WORKFLOW_USER_PROMPT
        ).render(user_prompt=user_prompt)
        messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": prompt_workflow_selection,
        },
    ]
        response = try_create_chat_completion(messages=messages)
        workflow = response.split(":")[1].strip().lower()
        self.executed = True
        self._context_manager.set_context("workflow", workflow)
        if workflow == "supervised learning":
            return [SLTask()]
        elif workflow == "reinforcement learning":
            return [RLTask()]
        else:
            raise ValueError(f"The workflow: {workflow} is not supported")
    
    def interact(self) -> Any:
        assert self.executed == True, "The workflow task has not been executed yet"
        ## TODO use logger
        self.logger.info(
            f"The workflow has been determined to be ---{self._context_manager.get_context('workflow')}---"
        )
        self.logger.info(
            "Enter 'y' to authorise command,'s' to run self-feedback commands, "
            "'n' to exit program, or enter feedback for WorkflowTask"
        )
        answer = input()
        if answer.lower().strip() == "y":
            return
        else:
            # TODO add self feedback
            raise ValueError("The input cannot be interpreted as a valid input")
        

class PlanTask(Task):
    def execute(self, prompt) -> List[Task]:
        return []

class SLTask(PlanTask):
    def __init__(self,) -> None:
        super().__init__(Task.TASK_TYPE_PLAN)

    def exeute(self):
        """
        return a list of interested tasks
        Copy the template project maybe a part of the task
        """
        return []
    
class RLTask(PlanTask):
    def __init__(self,) -> None:
        super().__init__(Task.TASK_TYPE_PLAN)
    def exeute(self):
        """
        return a list of interested tasks
        Copy the template project maybe a part of the task
        """
        return []


class ActionTask(Task):
    def execute(self) -> Literal["fail", "success"]:
        return "success"
    
"""Context Manager stores the context of the workflow"""
"""All context are key value pairs which saves the input, output and status of the whole workflow"""
class WorkflowContextManager():
    def __init__(self) -> None:
        self.context = {}
        self.logger = get_module_logger("fincoWorkflowContextManager")

    def set_context(self, key, value):
        if key in self.context:
            self.logger.warning("The key already exists in the context, the value will be overwritten")
        self.context[key] = value
    
    def get_context(self, key):
        if key not in self.context:
            self.logger.warning("The key doesn't exist in the context")
            return None
        return self.context[key]
    
    """return a deep copy of the context"""
    """TODO: do we need to return a deep copy?"""
    def get_all_context(self):
        return copy.deepcopy(self.context)


class WorkflowManager:
    """This manange the whole task automation workflow including tasks and actions"""

    def __init__(self, name="project", output_path=None) -> None:

        if output_path is None:
            self._output_path = Path.cwd() / name
        else:
            self._output_path = Path(output_path)
        self._context = WorkflowContextManager()

    """Direct call set_context method of the context manager"""
    def set_context(self, key, value):
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

        # NOTE: list may not be enough for general task list
        self.set_context("user_prompt", prompt)
        task_list = [WorkflowTask()]
        while len(task_list):
            """task list is not long, so sort it is not a big problem"""
            task_list = sorted(task_list, key=lambda x: x.task_type)
            t = task_list.pop(0)
            t.assign_context_manager(self._context)
            res = t.execute()
            if not cfg.continous_mode:
                res = t.interact()
            if t.task_type == Task.TASK_TYPE_WORKFLOW or t.task_type == Task.TASK_TYPE_PLAN:
                task_list.extend(res)
            elif t.task_type == Task.TASK_TYPE_ACTION:
                if res != "success":
                    ...
                    # TODO: handle the unexpected execution Error
            else:
                raise NotImplementedError("Unsupported action type")
            self.add_context(t.summarize())
        return self._output_path

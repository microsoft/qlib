import os

from pathlib import Path
from typing import Any, List
from qlib.log import get_module_logger
from qlib.typehint import Literal
from qlib.finco.llm import try_create_chat_completion
from jinja2 import Template

import abc
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

    ## all subclass should implement this method to determine task type
    @abc.abstractclassmethod
    def __init__(self) -> None:
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
        super().__init__()
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
        response = ""
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
        try:
            answer = input("You answer is:")
        except KeyboardInterrupt:
            self.logger.info("User has exited the program")
            exit()
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
        super().__init__()

    def exeute(self):
        """
        return a list of interested tasks
        Copy the template project maybe a part of the task
        """
        return []
    
class RLTask(PlanTask):
    def __init__(self,) -> None:
        super().__init__()
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
    __DEFAULT_OUTPUT_PATH = "./"

    __DEFAULT_WORKFLOW_SYSTEM_PROMPT = """
    Your task is to help user to analysis the output of qlib, your information including the strategy's backtest index 
    and runtime log. You may receive some scripts of the code as well, you can use them to analysis the output. 
    If there are any abnormal areas in the log or scripts, please also point them out.
    
    Example output 1:
    The backtest indexes show that your strategy's max draw down is a bit large, 
    You can try diversifying your positions across different assets.
    """
    __DEFAULT_WORKFLOW_USER_PROMPT = "Here is my information: '{{information}}'\n{{user_prompt}}"
    __DEFAULT_USER_PROMPT = "Please summarize them and give me some advice."

    def __init__(self):
        super().__init__()

    def execution(self) -> Any:
        user_prompt = self._context_manager.get_context("user_prompt")
        user_prompt = user_prompt if user_prompt is not None else self.__DEFAULT_USER_PROMPT
        system_prompt = self.__DEFAULT_WORKFLOW_SYSTEM_PROMPT
        output_path = self._context_manager.get_context("output_path")
        output_path = output_path if output_path is not None else self.__DEFAULT_OUTPUT_PATH
        information = self.parse2txt(output_path)

        prompt_workflow_selection = Template(self.__DEFAULT_WORKFLOW_USER_PROMPT).render(information=information,
                                                                                         user_prompt=user_prompt)
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
        return response

    def summarize(self) -> str:
        return ''

    def interact(self) -> Any:
        return

    @staticmethod
    def parse2txt(path) -> List:
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

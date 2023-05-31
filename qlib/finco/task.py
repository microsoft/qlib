import os

from pathlib import Path
from typing import Any, List
from jinja2 import Template
import abc
import re
import logging

from qlib.log import get_module_logger
from qlib.finco.utils import build_messages_and_create_chat_completion

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

    def __init__(self) -> None:
        self._context_manager = None
        self.executed = False
        self.logger : logging.Logger = get_module_logger(f"finco.{self.__class__.__name__}")
    
    def summarize(self) -> str:
        """After the execution of the task, it is supposed to generated some context about the execution"""
        """This function might be converted to abstract method in the future"""
        self.logger.info("The method has nothing to summarize")

    def assign_context_manager(self, context_manager):
        """assign the workflow context manager to the task"""
        """then all tasks can use this context manager to share the same context"""
        self._context_manager = context_manager
    
    def save_chat_history_to_context_manager(self, user_input, response):
        chat_history = self._context_manager.get_context("chat_history")
        if chat_history is None:
            chat_history = []
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})
        self._context_manager.update_context("chat_history", chat_history)

    @abc.abstractclassmethod
    def execute(self, **kwargs) -> Any:
        """The execution results of the task"""
        """All sub classes should implement the execute method to determine the next task"""
        raise NotImplementedError

    @abc.abstractclassmethod
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
Your task is to determine the workflow in Qlib (supervised learning or reinforcement learning) ensuring the workflow can meet the user's requirements.

The user will provide the requirements, you will provide only the output the choice in exact format specified below with no explanation or conversation.

Example input 1:
Help me build a low turnover quant investment strategy that focus more on long turn return in China a stock market.

Example output 1:
workflow: supervised learning

Example input 2:
Help me build a pipeline to determine the best selling point of a stock in a day or half a day in USA stock market.

Example output 2:
workflow: reinforcement learning
        """

        self.__DEFAULT_WORKFLOW_USER_PROMPT = (
            "User input: '{{user_prompt}}'\n"
            "Please provide the workflow in Qlib (supervised learning or reinforcement learning) ensureing the workflow can meet the user's requirements.\n"
            "Response only with the output in the exact format specified in the system prompt, with no explanation or conversation.\n"
        )

    def execute(self,) -> List[Task]:
        """make the choice which main workflow (RL, SL) will be used"""
        user_prompt = self._context_manager.get_context("user_prompt")
        system_prompt = self.__DEFAULT_WORKFLOW_SYSTEM_PROMPT
        prompt_workflow_selection = Template(
            self.__DEFAULT_WORKFLOW_USER_PROMPT
        ).render(user_prompt=user_prompt)
        response = build_messages_and_create_chat_completion(prompt_workflow_selection, system_prompt)
        self.save_chat_history_to_context_manager(prompt_workflow_selection, response)
        # TODO: use the above line instead of the following line before release!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # response = 'workflow: supervised learning'
        workflow = response.split(":")[1].strip().lower()
        self.executed = True
        self._context_manager.set_context("workflow", workflow)
        if workflow == "supervised learning":
            return [SLPlanTask()]
        elif workflow == "reinforcement learning":
            return [RLPlanTask()]
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
    pass

class SLPlanTask(PlanTask):
    def __init__(self,) -> None:
        super().__init__()
        self.__DEFAULT_WORKFLOW_SYSTEM_PROMPT = """
Your task is to determine the 5 crucial components in Qlib (Dataset, Model, Record, Strategy, Backtest) ensuring the workflow can meet the user's requirements.

For each component, you first point out whether to use default module in Qlib or implement the new module (Default or Personized). Default module means the class has already be implemented by Qlib which can be found in document and source code. Default class can be directed called from config file without additional implementation. Personized module means new python class is implemented and called from config file. You should always provide the reason of your choice.

The user will provide the requirements, you will provide only the output the choice in exact format specified below with no explanation or conversation. You only response 5 components in the order of dataset, model, record, strategy, backtest with no other addition.

Example input:
Help me build a low turnover quant investment strategy that focus more on long turn return in China a stock market. I have some data in csv format and I want to merge them with the data in Qlib.

Example output:
components:
- Dataset: (Personized) I will implement a CustomDataset inherited from qlib.data.dataset and exposed a api to load user's csv file. I will check the format of user's data and align them with Qlib data. Because it is a suitable dataset to get a long turn return in China A stock market.

- Model: (Default) I will use LGBModel in qlib.contrib.model.gbdt and choose more robust hyperparameters to focus on long-term return. Because tree model is more stable than NN models and is more unlikely to be over converged.

- Record: (Default) I will use SignalRecord in qlib.workflow.record_temp and SigAnaRecord in qlib.workflow.record_temp to save all the signals and the analysis results. Because user needs to check the metrics to determine whether the system meets the requirements.

- Strategy: (Default) I will use TopkDropoutStrategy in qlib.contrib.strategy. Because it is a more robust strategy which saves turnover fee.

- Backtest: (Default) I will use the default backtest module in Qlib. Because it can tell the user a more real performance result of the model we build.
        """
        self.__DEFAULT_WORKFLOW_USER_PROMPT = (
            "User input: '{{user_prompt}}'\n"
            "Please provide the 5 crucial components in Qlib (dataset, model, record, strategy, backtest) ensureing the workflow can meet the user's requirements.\n"
            "Response only with the output in the exact format specified in the system prompt, with no explanation or conversation.\n"
        )

    def execute(self):
        workflow = self._context_manager.get_context("workflow")
        assert workflow == "supervised learning", "The workflow is not supervised learning"

        user_prompt = self._context_manager.get_context("user_prompt")
        assert user_prompt is not None, "The user prompt is not provided"
        system_prompt = self.__DEFAULT_WORKFLOW_SYSTEM_PROMPT
        prompt_plan_all = Template(
            self.__DEFAULT_WORKFLOW_USER_PROMPT
        ).render(user_prompt=user_prompt)
        response = build_messages_and_create_chat_completion(prompt_plan_all, system_prompt)
        self.save_chat_history_to_context_manager(prompt_plan_all, response)
        # TODO: use upper lines instead of the following line before release!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # response = 'components:\n- Dataset: (Default) I will use a default dataset in Qlib for China A stock market. Because it is a suitable dataset that already contains the required data.\n\n- Model: (Default) I will use LGBModel in qlib.contrib.model.gbdt and choose more robust hyperparameters to focus on long-term return. Because tree model is more stable than NN models and is more unlikely to be over converged.\n\n- Record: (Default) I will use SignalRecord in qlib.workflow.record_temp and SigAnaRecord in qlib.workflow.record_temp to save all the signals and the analysis results. Because user needs to check the metrics to determine whether the system meets the requirements.\n\n- Strategy: (Default) I will use TopkDropoutStrategy in qlib.contrib.strategy. Because it is a more robust strategy which saves turnover.\n\n- Backtest: (Default) I will use the default backtest module in Qlib. Because it can tell the user a more real performance result of the model we build.'
        if "components" not in response:
            self.logger.warning("The response is not in the correct format, which probably means the answer is not correct")
        
        regex_dict = {
            "Dataset":re.compile("Dataset: \((.*?)\) (.*?)\n"),
            "Model":re.compile("Model: \((.*?)\) (.*?)\n"),
            "Record":re.compile("Record: \((.*?)\) (.*?)\n"),
            "Strategy":re.compile("Strategy: \((.*?)\) (.*?)\n"),
            "Backtest":re.compile("Backtest: \((.*?)\) (.*?)$"),
        }
        new_task = []
        for name, regex in regex_dict.items():
            res = re.search(regex, response)
            if not res:
                self.logger.error(f"The search for {name} decision failed")
            else:
                self._context_manager.set_context(f"{name}_decision", res.group(1))
                self._context_manager.set_context(f"{name}_plan", res.group(2))
                assert res.group(1) in ["Default", "Personized"]
                if res.group(1) == "Default":
                    new_task.append(ConfigActionTask(name))
                elif res.group(1) == "Personized":
                    new_task.append(ConfigActionTask(name), ImplementActionTask(name))
        return new_task
    
class RLPlanTask(PlanTask):
    def __init__(self,) -> None:
        super().__init__()
        self.logger.error("The RL task is not implemented yet")
        exit()

    def execute(self):
        """
        return a list of interested tasks
        Copy the template project maybe a part of the task
        """
        return []


class ActionTask(Task):
    pass

class ConfigActionTask(ActionTask):
    def __init__(self, component) -> None:
        super().__init__()
        self.target_componet = component
        self.__DEFAULT_CONFIG_ACTION_SYSTEM_PROMPT = """
Your task is to write the config of target component in Qlib(Dataset, Model, Record, Strategy, Backtest).

Config means the yaml file in Qlib. You can find the default config in qlib/contrib/config_template. You can also find the config in Qlib document.

The user has provided the requirements and made plan and reason to each component. You should strictly follow user's plan and you should provide the reason of your hyperparameter choices if exist and some suggestion if user wants to finetune the hyperparameters after the config. Default means you should only use classes in Qlib without any other new code while Personized has no such restriction. class in Qlib means Qlib has implemented the class and you can find it in Qlib document or source code.

You only need to write the config of the target component in the exact format specified below with no explanation or conversation.

Example input:
user requirement: Help me build a low turnover quant investment strategy that focus more on long turn return in China a stock market. I have some data in csv format and I want to merge them with the data in Qlib.
user plan:
- Dataset: (Personized) I will implement a CustomDataset imherited from qlib.data.dataset and exposed a api to load user's csv file. I will check the format of user's data and align them with Qlib data. Because it is a suitable dataset to get a long turn return in China A stock market.
- Model: (Default) I will use LGBModel in qlib.contrib.model.gbdt and choose more robust hyperparameters to focus on long-term return. Because tree model is more stable than NN models and is more unlikely to be over converged.
- Record: (Default) I will use SignalRecord in qlib.workflow.record_temp and SigAnaRecord in qlib.workflow.record_temp to save all the signals and the analysis results. Because user needs to check the metrics to determine whether the system meets the requirements.
- Strategy: (Default) I will use TopkDropoutStrategy in qlib.contrib.strategy. Because it is a more robust strategy which saves turnover fee.
- Backtest: (Default) I will use the default backtest module in Qlib. Because it can tell the user a more real performance result of the model we build.
target component: Model

Example output:
Config:
```yaml
model:
class: LGBModel
module_path: qlib.contrib.model.gbdt
kwargs:
    loss: mse
    colsample_bytree: 0.8879
    learning_rate: 0.2
    subsample: 0.8789
    lambda_l1: 205.6999
    lambda_l2: 580.9768
    max_depth: 8
    num_leaves: 210
    num_threads: 20
```
Reason: I choose the hyperparameters above because they are the default hyperparameters in Qlib and they are more robust than other hyperparameters.
Improve suggestion: You can try to tune the num_leaves in range [100, 300], max_depth in [5, 10], learning_rate in [0.01, 1] and other hyperparameters in the config. Since you're trying to get a long tern return, if you have enough computation resource, you can try to use a larger num_leaves and max_depth and a smaller learning_rate.
        """

        self.__CONFIG_ACTION_SYSTEM_PROMPT_TEMPLATE = (
"""
user requirement: {{user_requirement}}
user plan:
- Dataset: ({{dataset_decision}}) {{dataset_plan}}
- Model: ({{model_decision}}) {{model_plan}}
- Record: ({{record_decision}}) {{record_plan}}
- Strategy: ({{strategy_decision}}) {{strategy_plan}}
- Backtest: ({{backtest_decision}}) {{backtest_plan}}
target component: {{target_component}}
"""
        )

    def execute(self):
        user_prompt = self._context_manager.get_context("user_prompt")
        component_list = ["Dataset", "Model", "Record", "Strategy", "Backtest"]
        prompt_element_dict = dict()
        for component in component_list:
            prompt_element_dict[f"{component}_decision"] = self._context_manager.get_context(f"{component}_decision")
            prompt_element_dict[f"{component}_plan"] = self._context_manager.get_context(f"{component}_plan")
        
        assert None not in prompt_element_dict.values(), "Some decision or plan is not set by plan maker"

        config_prompt = Template(self.__CONFIG_ACTION_SYSTEM_PROMPT_TEMPLATE).render(
            user_requirement=user_prompt,
            dataset_decision=prompt_element_dict["Dataset_decision"],
            dataset_plan=prompt_element_dict["Dataset_plan"],
            model_decision=prompt_element_dict["Model_decision"],
            model_plan=prompt_element_dict["Model_plan"],
            record_decision=prompt_element_dict["Record_decision"],
            record_plan=prompt_element_dict["Record_plan"],
            strategy_decision=prompt_element_dict["Strategy_decision"],
            strategy_plan=prompt_element_dict["Strategy_plan"],
            backtest_decision=prompt_element_dict["Backtest_decision"],
            backtest_plan=prompt_element_dict["Backtest_plan"],
            target_component=self.target_componet
        )
        response = build_messages_and_create_chat_completion(config_prompt, self.__DEFAULT_CONFIG_ACTION_SYSTEM_PROMPT)
        self.save_chat_history_to_context_manager(config_prompt, response)
        
        res = re.search(r"Config:(.*)Reason:(.*)Improve suggestion:(.*)", response, re.S)
        assert res is not None and len(res.groups()) == 3, "The response of config action task is not in the correct format"

        config = re.search(r"```yaml(.*)```", res.group(1), re.S)
        assert config is not None, "The config part of config action task response is not in the correct format"
        config = config.group(1)
        reason = res.group(2)
        improve_suggestion = res.group(3)

        self._context_manager.set_context(f"{self.target_componet}_config", config)
        self._context_manager.set_context(f"{self.target_componet}_reason", reason)
        self._context_manager.set_context(f"{self.target_componet}_improve_suggestion", improve_suggestion)

        return []
class ImplementActionTask(ActionTask):
    def __init__(self) -> None:
        super().__init__()
    
    def execute(self):
        """
        return a list of interested tasks
        Copy the template project maybe a part of the task
        """
        return []

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

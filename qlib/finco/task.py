import os

from pathlib import Path
import io
from typing import Any, List, Union
from jinja2 import Template
import ruamel.yaml as yaml
import abc
import re
import logging
import subprocess
import platform

from qlib.log import get_module_logger
from qlib.finco.llm import APIBackend
from qlib.finco.tpl import get_tpl_path
from qlib.workflow.record_temp import HFSignalRecord, SignalRecord
from qlib.contrib.analyzer import HFAnalyzer, SignalAnalyzer
from qlib.utils import init_instance_by_config
from qlib.workflow import R


class Task:
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
        self.logger: logging.Logger = get_module_logger(f"finco.{self.__class__.__name__}")

    def summarize(self) -> str:
        """After the execution of the task, it is supposed to generated some context about the execution"""
        """This function might be converted to abstract method in the future"""
        self.logger.info("The method has nothing to summarize")

    def assign_context_manager(self, context_manager):
        """assign the workflow context manager to the task"""
        """then all tasks can use this context manager to share the same context"""
        self._context_manager = context_manager

    def save_chat_history_to_context_manager(self, user_input, response, system_prompt):
        chat_history = self._context_manager.get_context("chat_history")
        if chat_history is None:
            chat_history = []
        chat_history.append({"role": "system", "content": system_prompt})
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

    def __init__(
        self,
    ) -> None:
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

    def execute(
        self,
    ) -> List[Task]:
        """make the choice which main workflow (RL, SL) will be used"""
        user_prompt = self._context_manager.get_context("user_prompt")
        prompt_workflow_selection = Template(self.__DEFAULT_WORKFLOW_USER_PROMPT).render(user_prompt=user_prompt)
        response = APIBackend().build_messages_and_create_chat_completion(
            prompt_workflow_selection, self.__DEFAULT_WORKFLOW_SYSTEM_PROMPT
        )
        self.save_chat_history_to_context_manager(
            prompt_workflow_selection, response, self.__DEFAULT_WORKFLOW_SYSTEM_PROMPT
        )
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
    def __init__(
        self,
    ) -> None:
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
        prompt_plan_all = Template(self.__DEFAULT_WORKFLOW_USER_PROMPT).render(user_prompt=user_prompt)
        response = APIBackend().build_messages_and_create_chat_completion(
            prompt_plan_all, self.__DEFAULT_WORKFLOW_SYSTEM_PROMPT
        )
        self.save_chat_history_to_context_manager(prompt_plan_all, response, self.__DEFAULT_WORKFLOW_SYSTEM_PROMPT)
        if "components" not in response:
            self.logger.warning(
                "The response is not in the correct format, which probably means the answer is not correct"
            )

        regex_dict = {
            "Dataset": re.compile("Dataset: \((.*?)\) (.*?)\n"),
            "Model": re.compile("Model: \((.*?)\) (.*?)\n"),
            "Record": re.compile("Record: \((.*?)\) (.*?)\n"),
            "Strategy": re.compile("Strategy: \((.*?)\) (.*?)\n"),
            "Backtest": re.compile("Backtest: \((.*?)\) (.*?)$"),
        }
        new_task = []
        # 1) create a workspace
        # TODO: we have to make choice between `sl` and  `sl-cfg`
        new_task.append(
            CMDTask(
                cmd_intention=f"Copy folder from {get_tpl_path() / 'sl'} to {self._context_manager.get_context('workspace')}"
            )
        )

        # 2) CURD on the workspace
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
                    new_task.extend([ConfigActionTask(name), ImplementActionTask(name)])
        return new_task


class RLPlanTask(PlanTask):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.logger.error("The RL task is not implemented yet")
        exit()

    def execute(self):
        """
        return a list of interested tasks
        Copy the template project maybe a part of the task
        """
        return []


class RecorderTask(Task):
    """
    This Recorder task is responsible for analysing data such as index and distribution.
    """

    __ANALYZERS_PROJECT = {HFAnalyzer.__name__: HFSignalRecord, SignalAnalyzer.__name__: SignalRecord}
    __ANALYZERS_DOCS = {HFAnalyzer.__name__: HFAnalyzer.__doc__, SignalAnalyzer.__name__: SignalAnalyzer.__doc__}
    # __ANALYZERS_PROJECT = {SignalAnalyzer.__name__: SignalRecord}
    # __ANALYZERS_DOCS = {SignalAnalyzer.__name__: SignalAnalyzer.__doc__}

    __DEFAULT_WORKFLOW_SYSTEM_PROMPT = f"""
        You are an expert system administrator.
        Your task is to select the best analysis class based on user intent from this list:
        {list(__ANALYZERS_DOCS.keys())}
        Their description are:
        {__ANALYZERS_DOCS}
        
        Response only with the Analyser name provided above with no explanation or conversation. if there are more than
        one analyser, separate them by ","
        
    """

    __DEFAULT_WORKFLOW_USER_PROMPT = """{{user_prompt}}, 
    The analyzers you select should separate by ",", such as: "HFAnalyzer", "SignalAnalyzer"
    """

    def __init__(self):
        super().__init__()
        self._output = None

    def execute(self):
        prompt = Template(self.__DEFAULT_WORKFLOW_USER_PROMPT).render(
            user_prompt=self._context_manager.get_context("user_prompt"))
        be = APIBackend()
        be.debug_mode = False
        response = be.build_messages_and_create_chat_completion(prompt, self.__DEFAULT_WORKFLOW_SYSTEM_PROMPT)

        # it's better to move to another Task
        workflow_config = self._context_manager.get_context("workflow_config") \
            if self._context_manager.get_context("workflow_config") else "workflow_config.yaml"
        workspace = self._context_manager.get_context('workspace')
        with workspace.joinpath(workflow_config).open() as f:
            workflow = yaml.safe_load(f)

        model = init_instance_by_config(workflow["task"]["model"])
        dataset = init_instance_by_config(workflow["task"]["dataset"])

        with R.start(experiment_name="finCo"):
            model.fit(dataset)
            R.save_objects(trained_model=model)

            # prediction
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()

        analysers = response.split(',')
        if isinstance(analysers, list):
            self.logger.info(f"selected analysers: {analysers}")
            tasks = []
            for analyser in analysers:
                if analyser in self.__ANALYZERS_PROJECT.keys():
                    tasks.append(self.__ANALYZERS_PROJECT.get(analyser)(workspace=workspace, model=model,
                                                                        dataset=dataset, recorder=recorder))

            for task in tasks:
                resp = task.analyse()
                self._context_manager.set_context(task.__class__.__name__, resp)

        return []


class ActionTask(Task):
    pass


class CMDTask(ActionTask):
    """
    This CMD task is responsible for ensuring compatibility across different operating systems.
    """

    __DEFAULT_WORKFLOW_SYSTEM_PROMPT = """
You are an expert system administrator.
Your task is to convert the user's intention into a specific runnable command for a particular system.
Example input:
- User intention: Copy the folder from  a/b/c to d/e/f
- User OS: Linux
Example output:
cp -r a/b/c d/e/f
"""
    __DEFAULT_WORKFLOW_USER_PROMPT = """
Example input:
- User intention: "{{cmd_intention}}"
- User OS: "{{user_os}}"
Example output:
"""

    def __init__(self, cmd_intention: str, cwd=None):
        self.cwd = cwd
        self.cmd_intention = cmd_intention
        self._output = None

    def execute(self):
        prompt = Template(self.__DEFAULT_WORKFLOW_USER_PROMPT).render(
            cmd_intention=self.cmd_intention, user_os=platform.system()
        )
        response = APIBackend().build_messages_and_create_chat_completion(prompt, self.__DEFAULT_WORKFLOW_SYSTEM_PROMPT)
        self._output = subprocess.check_output(response, shell=True, cwd=self.cwd)
        return []

    def summarize(self):
        if self._output is not None:
            # TODO: it will be overrides by later commands
            # utf8 can't decode normally on Windows
            self._context_manager.set_context(self.__class__.__name__, self._output.decode("ANSI"))


class ConfigActionTask(ActionTask):
    def __init__(self, component) -> None:
        super().__init__()
        self.target_componet = component
        # TODO check if it's necessary that we input the plan of all components
        self.__DEFAULT_CONFIG_ACTION_SYSTEM_PROMPT = """
Your task is to write the config of target component in Qlib(Dataset, Model, Record, Strategy, Backtest).

Config means the yaml file in Qlib. You can find the default config in qlib/contrib/config_template. You can also find the config in Qlib document.

The user has provided the requirements and made plan and reason to each component. You should strictly follow user's plan and you should provide the reason of your hyperparameter choices if exist and some suggestion if user wants to finetune the hyperparameters after the config. Default means you should only use classes in Qlib without any other new code while Personized has no such restriction. class in Qlib means Qlib has implemented the class and you can find it in Qlib document or source code.

Config, Reason and Improve suggestion should always be provided.

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

        self.__CONFIG_ACTION_SYSTEM_PROMPT_TEMPLATE = """
user requirement: {{user_requirement}}
user plan:
- Dataset: ({{dataset_decision}}) {{dataset_plan}}
- Model: ({{model_decision}}) {{model_plan}}
- Record: ({{record_decision}}) {{record_plan}}
- Strategy: ({{strategy_decision}}) {{strategy_plan}}
- Backtest: ({{backtest_decision}}) {{backtest_plan}}
target component: {{target_component}}
"""

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
            target_component=self.target_componet,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            config_prompt, self.__DEFAULT_CONFIG_ACTION_SYSTEM_PROMPT
        )
        self.save_chat_history_to_context_manager(config_prompt, response, self.__DEFAULT_CONFIG_ACTION_SYSTEM_PROMPT)
        res = re.search(r"Config:(.*)Reason:(.*)Improve suggestion:(.*)", response, re.S)
        assert (
            res is not None and len(res.groups()) == 3
        ), "The response of config action task is not in the correct format"

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
    def __init__(self, target_component) -> None:
        super().__init__()
        self.target_component = target_component
        self.__DEFAULT_IMPLEMENT_ACTION_SYSTEM_PROMPT = """
Your task is to write python code and give some reasonable explanation. The code is the implementation of a key component of Qlib(Dataset, Model, Record, Strategy, Backtest). 

The user has provided the requirements and made plan and reason to each component. You should strictly follow user's plan. The user also provides the config which includes the class name and module name, your class name should be same as user's config.

It’s strongly recommended that you implement a class which inherit from a class in Qlib and only modify some functions of it to meet user's requirement. After the code, you should write the explanation of your code. It contains the core idea of your code. Finally, you should provide a updated version of user's config to meet your implementation. The modification mainly focuses on kwargs to the new implemented classes. You can output same config as user input is nothing needs to change.

You response should always contain "Code", "Explanation", "Modified config".

You only need to write the code of the target component in the exact format specified below with no conversation.

Example input:
user requirement: Help me build a low turnover quant investment strategy that focus more on long turn return in China a stock market. I have some data in csv format and I want to merge them with the data in Qlib.
user plan:
- Dataset: (Personized) I will implement a CustomDataset imherited from qlib.data.dataset and exposed a api to load user's csv file. I will check the format of user's data and align them with Qlib data. Because it is a suitable dataset to get a long turn return in China A stock market.
- Model: (Default) I will use LGBModel in qlib.contrib.model.gbdt and choose more robust hyperparameters to focus on long-term return. Because tree model is more stable than NN models and is more unlikely to be over converged.
- Record: (Default) I will use SignalRecord in qlib.workflow.record_temp and SigAnaRecord in qlib.workflow.record_temp to save all the signals and the analysis results. Because user needs to check the metrics to determine whether the system meets the requirements.
- Strategy: (Default) I will use TopkDropoutStrategy in qlib.contrib.strategy. Because it is a more robust strategy which saves turnover fee.
- Backtest: (Default) I will use the default backtest module in Qlib. Because it can tell the user a more real performance result of the model we build.
User config:
```yaml
dataset:  
  class: CustomDataset  
  module_path: path.to.your.custom_dataset_module  
  kwargs:  
    handler:  
      class: CSVMergeHandler  
      module_path: path.to.your.csv_merge_handler_module  
      kwargs:  
        csv_path: path/to/your/csv/data  
```
target component: Dataset
Example output:
Code:
```python
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class CSVMergeHandler(DataHandlerLP):
    def __init__(self, csv_path, **kwargs):
        super().__init__(**kwargs)
        self.csv_data = pd.read_csv(csv_path)

    def load_all(self):
        qlib_data = super().load_all()
        merged_data = qlib_data.merge(self.csv_data, on=["date", "instrument"], how="left")
        return merged_data


class CustomDataset(DatasetH):
    def __init__(self, handler):
        super().__init__(handler)
```
Explanation:
In this implementation, the CSVMergeHandler class inherits from DataHandlerLP and overrides the load_all method to merge the csv data with Qlib data. The CustomDataset class inherits from DatasetH and takes the handler as an argument.
Modified config:
```yaml
dataset:  
  class: CustomDataset  
  module_path: custom_dataset  
  kwargs:  
    handler:  
      class: CSVMergeHandler  
      module_path: custom_dataset  
      kwargs:  
        csv_path: path/to/your/csv/data
```
    """
        self.__DEFAULT_IMPLEMENT_ACTION_USER_PROMPT = """
user requirement: {{user_requirement}}
user plan:
- Dataset: ({{dataset_decision}}) {{dataset_plan}}
- Model: ({{model_decision}}) {{model_plan}}
- Record: ({{record_decision}}) {{record_plan}}
- Strategy: ({{strategy_decision}}) {{strategy_plan}}
- Backtest: ({{backtest_decision}}) {{backtest_plan}}
User config:
```yaml
{{user_config}}
```
target component: {{target_component}}
        """

    def execute(self):
        """
        return a list of interested tasks
        Copy the template project maybe a part of the task
        """

        user_prompt = self._context_manager.get_context("user_prompt")
        component_list = ["Dataset", "Model", "Record", "Strategy", "Backtest"]
        prompt_element_dict = dict()
        for component in component_list:
            prompt_element_dict[f"{component}_decision"] = self._context_manager.get_context(f"{component}_decision")
            prompt_element_dict[f"{component}_plan"] = self._context_manager.get_context(f"{component}_plan")

        assert None not in prompt_element_dict.values(), "Some decision or plan is not set by plan maker"
        config = self._context_manager.get_context(f"{self.target_component}_config")

        implement_prompt = Template(self.__DEFAULT_IMPLEMENT_ACTION_USER_PROMPT).render(
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
            target_component=self.target_component,
            user_config=config,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            implement_prompt, self.__DEFAULT_IMPLEMENT_ACTION_SYSTEM_PROMPT
        )
        self.save_chat_history_to_context_manager(
            implement_prompt, response, self.__DEFAULT_IMPLEMENT_ACTION_SYSTEM_PROMPT
        )

        res = re.search(r"Code:(.*)Explanation:(.*)Modified config:(.*)", response, re.S)
        assert (
            res is not None and len(res.groups()) == 3
        ), f"The response of implement action task of component {self.target_component} is not in the correct format"

        code = re.search(r"```python(.*)```", res.group(1), re.S)
        assert code is not None, "The code part of implementation action task response is not in the correct format"
        code = code.group(1)
        explanation = res.group(2)
        modified_config = re.search(r"```yaml(.*)```", res.group(3), re.S)
        assert (
            modified_config is not None
        ), "The modified config part of implementation action task response is not in the correct format"
        modified_config = modified_config.group(1)

        self._context_manager.set_context(f"{self.target_component}_code", code)
        self._context_manager.set_context(f"{self.target_component}_code_explanation", explanation)
        self._context_manager.set_context(f"{self.target_component}_modified_config", modified_config)

        return []


class YamlEditTask(ActionTask):
    """This yaml edit task will replace a specific component directly"""

    def __init__(self, file: Union[str, Path], module_path: str, updated_content: str):
        """

        Parameters
        ----------
        file
            a target file that needs to be modified
        module_path
            the path to the section that needs to be replaced with `updated_content`
        updated_content
            The content to replace the original content in `module_path`
        """
        self.p = Path(file)
        self.module_path = module_path
        self.updated_content = updated_content

    def execute(self):
        # 1) read original and new content
        with self.p.open("r") as f:
            config = yaml.safe_load(f)
        update_config = yaml.safe_load(io.StringIO(self.updated_content))

        # 2) locate the module
        focus = config
        module_list = self.module_path.split(".")
        for k in module_list[:-1]:
            focus = focus[k]

        # 3) replace the module and save
        focus[module_list[-1]] = update_config
        with self.p.open("w") as f:
            yaml.dump(config, f)


class SummarizeTask(Task):
    __DEFAULT_WORKSPACE = "./"

    __DEFAULT_WORKFLOW_SYSTEM_PROMPT = """
    You are an expert in quant domain.
    Your task is to help user to analysis the output of qlib, your main focus is on the backtesting metrics of 
    user strategies. Warnings reported during runtime can be ignored if deemed appropriate.
    your information including the strategy's backtest log and runtime log. 
    You may receive some scripts of the codes as well, you can use them to analysis the output.
    At the same time, you can also use your knowledge of the Microsoft/Qlib project and finance to complete your tasks.
    If there are any abnormal areas in the log or scripts, please also point them out.
    
    Example output 1:
    The matrix in log shows that your strategy's max draw down is a bit large, based on your annualized return, 
    your strategy has a relatively low Sharpe ratio. Here are a few suggestions:
    You can try diversifying your positions across different assets.
    
    Example output 2:
    The output log shows the result of running `qlib` with `LinearModel` strategy on the Chinese stock market CSI 300 
    from 2008-01-01 to 2020-08-01, based on the Alpha158 data handler from 2015-01-01. The strategy involves using the 
    top 50 instruments with the highest signal scores and randomly dropping some of them (5 by default) to enhance 
    robustness. The backtesting result is shown in the table below:
        
        | Metrics | Value |
        | ------- | ----- |
        | IC | 0.040 |
        | ICIR | 0.312 |
        | Long-Avg Ann Return | 0.093 |
        | Long-Avg Ann Sharpe | 0.462 |
        | Long-Short Ann Return | 0.245 |
        | Long-Short Ann Sharpe | 4.098 |
        | Rank IC | 0.048 |
        | Rank ICIR | 0.370 |


    It should be emphasized that:
    You should output a report, the format of your report is Markdown format.
    Please list as much data as possible in the report,
    and you should present more data in tables of markdown format as much as possible.
    The numbers in the report do not need to have too many significant figures.
    You can add subheadings and paragraphs in Markdown for readability.
    You can bold or use other formatting options to highlight keywords in the main text.
    """
    __DEFAULT_WORKFLOW_USER_PROMPT = "Here is my information: '{{information}}'\n{{user_prompt}}"
    __DEFAULT_USER_PROMPT = "Please summarize them and give me some advice."

    # TODO: 2048 is close to exceed GPT token limit
    __MAX_LENGTH_OF_FILE = 2048
    __DEFAULT_REPORT_NAME = "finCoReport.md"

    def __init__(self):
        super().__init__()

    def execute(self) -> Any:
        user_prompt = self._context_manager.get_context("user_prompt")
        user_prompt = user_prompt if user_prompt is not None else self.__DEFAULT_USER_PROMPT
        system_prompt = self.__DEFAULT_WORKFLOW_SYSTEM_PROMPT
        workspace = self._context_manager.get_context("workspace")
        workspace = workspace if workspace is not None else self.__DEFAULT_WORKSPACE
        file_info = self.get_info_from_file(workspace)
        context_info = self.get_info_from_context()

        information = context_info + file_info
        prompt_workflow_selection = Template(self.__DEFAULT_WORKFLOW_USER_PROMPT).render(
            information=information, user_prompt=user_prompt
        )

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=prompt_workflow_selection, system_prompt=system_prompt
        )
        self.save_markdown(content=response)
        return []

    def summarize(self) -> str:
        return ""

    def interact(self) -> Any:
        return

    def get_info_from_file(self, path) -> List:
        """
        read specific type of files under path
        """
        file_list = []
        path = Path.cwd().joinpath(path).resolve()
        for root, dirs, files in os.walk(path):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_list.append(file_path)

        result = []
        for file in file_list:
            postfix = file.split(".")[-1]
            if postfix in ["py", "log", "yaml"]:
                with open(file) as f:
                    content = f.read()
                    self.logger.info(f"file to summarize: {file}")
                    # in case of too large file
                    # TODO: Perhaps summarization method instead of truncation would be a better approach
                    result.append({"file": file, "content": content[: self.__MAX_LENGTH_OF_FILE]})

        return result

    def get_info_from_context(self):
        context = []
        # TODO: get all keys from context?
        for key in [
            "user_prompt",
            "chat_history",
            "Dataset_plan",
            "Model_plan",
            "Record_plan",
            "Strategy_plan",
            "Backtest_plan",
        ]:
            c = self._context_manager.get_context(key=key)
            if c is not None:
                c = str(c)
                context.append({key: c[: self.__MAX_LENGTH_OF_FILE]})
        return context

    def save_markdown(self, content: str):
        workspace = self._context_manager.get_context("workspace")
        workspace = workspace if workspace is not None else self.__DEFAULT_WORKSPACE
        with open(Path.joinpath(workspace, self.__DEFAULT_REPORT_NAME), "w") as f:
            f.write(content)
        self.logger.info(f"report has saved to {self.__DEFAULT_REPORT_NAME}")

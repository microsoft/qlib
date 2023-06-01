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
        prompt_workflow_selection = Template(
            self.__DEFAULT_WORKFLOW_USER_PROMPT
        ).render(user_prompt=user_prompt)
        response = build_messages_and_create_chat_completion(prompt_workflow_selection, self.__DEFAULT_WORKFLOW_SYSTEM_PROMPT)
        # TODO: use the above line instead of the following line before release!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # response = 'workflow: supervised learning'
        self.save_chat_history_to_context_manager(prompt_workflow_selection, response, self.__DEFAULT_WORKFLOW_SYSTEM_PROMPT)
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
        prompt_plan_all = Template(
            self.__DEFAULT_WORKFLOW_USER_PROMPT
        ).render(user_prompt=user_prompt)
        response = build_messages_and_create_chat_completion(prompt_plan_all, self.__DEFAULT_WORKFLOW_SYSTEM_PROMPT)
        # TODO: use upper lines instead of the following line before release!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # response = "components:\n- Dataset: (Default) I will use the default dataset module in Qlib. Because it is a suitable dataset to get a long-term return in China A stock market.\n\n- Model: (Personized) I will implement a CustomTransformerModel inherited from qlib.model.base and use the transformer model with 10 MLP layers before the head. Because it meets the user's requirement of using transformer model with 10 MLP layers before the head to achieve better long-term return.\n\n- Record: (Default) I will use SignalRecord in qlib.workflow.record_temp and SigAnaRecord in qlib.workflow.record_temp to save all the signals and the analysis results. Because user needs to check the metrics to determine whether the system meets the requirements.\n\n- Strategy: (Default) I will use TopkDropoutStrategy in qlib.contrib.strategy. Because it is a more robust strategy which saves turnover fee.\n\n- Backtest: (Default) I will use the default backtest module in Qlib. Because it can tell the user a more real performance result of the model we build."

        self.save_chat_history_to_context_manager(prompt_plan_all, response, self.__DEFAULT_WORKFLOW_SYSTEM_PROMPT)
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
                    new_task.extend([ConfigActionTask(name), ImplementActionTask(name)])
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
        # TODO use the upper lines to replace the following lines after debug
        # if self.target_componet == "Dataset":
        #     response = 'Config:\n```yaml\ndataset:\n    class: FeatureDataset\n    module_path: qlib.data.dataset\n    kwargs:\n        handler_config:\n            class: Alpha360\n            module_path: qlib.contrib.data.handler\n        segment:\n            train:\n                start_time: 2010-01-01\n                end_time: 2017-12-31\n            valid:\n                start_time: 2018-01-01\n                end_time: 2018-12-31\n            test:\n                start_time: 2019-01-01\n                end_time: 2019-12-31\n        scenario: Alpha360\n        feature:\n            label: \n                class: RegressionLabel\n                module_path: qlib.contrib.data.label\n                kwargs:\n                    label_name: f01\n                    direction: 1\n            transform:\n                class: Sequence\n                module_path: qlib.contrib.transform\n                kwargs:\n                    len_seq: 10\n                    step: 5\n                    mode: slide\n                    group: single\n            custom:\n                transform:\n                    class: CustomTransform\n                    module_path: qlib.contrib.data.transform\n                    kwargs:\n                        # customized feature processing for your features\n                        ma_windows:[5,10,20]\n                        macd_windows:[5,10,20]\n                        rsi_windows:[5,10]\n        train_loader:\n            class: PyTorchLoader\n            module_path: qlib.data.dataloader\n            kwargs:\n                batch_size: 4096\n                num_workers: 1\n        valid_loader:\n            class: PyTorchLoader\n            module_path: qlib.data.dataloader\n            kwargs:\n                batch_size: 4096\n                num_workers: 1   \n```   \nReason: I use the FeatureDataset class because the user did not provide the csv data format and we can assume the data is in the default format of Qlib. The FeatureDataset loads data by segment and we can choose the segment of train, valid and test for our model. In addition, I use the customized transform to process the features. With the customized transform, we could add more feature engineering procedures according to our own requirements.\n\nImprove suggestion: Since you are trying to build a low turnover strategy, you may consider using the subseries selection method to improve the efficiency of your model. Moreover, you can fine-tune the parameters in the customized transform according to your own requirements.'
        # elif self.target_componet == "Model":
        #     response = "Config:\n```yaml\nmodel:\nclass: CustomTransformerModel\nmodule_path: user_module.CustomModel \nkwargs:\n    transformer_stack_kwargs:\n        num_heads: 8\n        num_layers: 6\n        hidden_dim: 256\n        dropout: 0.1\n    mlp_last_dim: 128\n    mlp_activation: relu\n    mlp_layers: 10\n```\n`user_module` should be replaced with your own module path where you define the `CustomTransformerModel`.\n\nReason: \n- `CustomTransformerModel` is used to meet the user's requirement of using a transformer model with 10 MLP layers before the head, which might capture long-term signals more effectively.\n- `num_heads` and `num_layers` are set to 8 and 6 respectively as they are commonly used values in transformer model for financial time series, and they allow the model to capture complex patterns in the data with sufficient capacity.\n- `hidden_dim` is set to 256 to control the model's capacity, balance the computational efficiency, and avoid overfitting to the training data.\n- `mlp_last_dim` is set to 128 to avoid too much dimensionality reduction, and the activation function `relu` is commonly used in financial time series modeling.\n- `mlp_layers` is set to 10 to satisfy the user's requirement of using 10 MLP layers before the head, and it provides the model with enough capacity to capture long-term signals.\n\nImprove suggestion: \n- You can try to adjust the transformer stack hyperparameters, such as `num_heads`, `num_layers`, `hidden_dim`, or the MLP hyperparameters, such as `mlp_last_dim`, `mlp_activation`, `mlp_layers`, to improve the model's performance.\n- You can also try experimenting with different optimizers (`Adam`, `SGD`, etc.) or learning rates to optimize the training procedure.\n- Fine-tuning the hyperparameters may be based on observing the validation metrics and balancing computational efficiency with model performance."
        # elif self.target_componet == "Record":
        #     response = 'Config:\n```yaml\nrecord:\nclass: SignalRecord\nmodule_path: qlib.workflow.record_temp\nkwargs:\n    mode: "train"\n    dump_path: "./dump_dir/"\n    std_q: 10\n    max_keep_days: 365\n    clear_cache: True\n```\nReason: I choose the SignalRecord because it saves all the signals and is suitable for the task of building a low turnover strategy with a focus on long-term return. The std_q and max_keep_days are set to 10 and 365 respectively to enable the system to remove signals that are not performing well and keep signals that are performing well for up to a year. The clear_cache setting is set to True to ensure that the cache is cleared before each new run.\nImprove suggestion: Since the user wants to focus on long-term return, it might be worth considering increasing the max_keep_days value to 730 to keep signals for up to two years. Also, it is worth considering using a more advanced record module such as SigAnaRecord or MetaRecord to get more detailed analysis results.'
        # elif self.target_componet == "Strategy":
        #     response = 'Config:\n```yaml\nstrategy:\nclass: TopkDropoutStrategy\nmodule_path: qlib.contrib.strategy.strategy\nkwargs:\n    topk: 20\n    threshold: 0.025\n    max_orders: 5\n    dropout: 0.25\n    warmup_length: 100\n```\nReason: Since the user wants a low turnover strategy, I choose TopkDropoutStrategy as it saves turnover fee. The hyperparameters are set to default in Qlib, which is suitable for general use.\nImprove suggestion: For a more customized strategy, the user can try adjusting the hyperparameters of TopkDropoutStrategy, such as increasing the topk value to expand the stock pool, or modifying the threshold to adjust the confidence level of the model. Additionally, the user can explore other strategies in `qlib.contrib.strategy` module to achieve better long-term return.'
        # elif self.target_componet == "Backtest":
        #     response = 'Config:\n```yaml\nbacktest:\nclass: NormalBacktest\nmodule_path: qlib.backtest.backtest\nkwargs:\n    start_time: "2008-01-01"\n    end_time: "2021-12-31"\n    rebalance_period: "month"\n    benchmark: "SH000300"\n    trade_cost: 0.0015\n    min_cost: 5\n    return_mean: false\n```\nReason: I choose the default NormalBacktest in Qlib because it is a straightforward backtest method for evaluating long-term investment strategies. It has a reasonable trade cost and can perform monthly rebalance, making it suitable for evaluating the low turnover strategy that the user requires.\n\nImprove suggestion: If the user wants to fine-tune the performance of the backtest, they can consider adjusting the trade cost and minimum cost (min_cost) parameters to better reflect their trading environment. Additionally, they may experiment with other backtest methods in Qlib, such as LongShortBacktest or DelayEvalBacktest, to see if they can further improve the performance of their strategy.'
        
        self.save_chat_history_to_context_manager(config_prompt, response, self.__DEFAULT_CONFIG_ACTION_SYSTEM_PROMPT)
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
        self.__DEFAULT_IMPLEMENT_ACTION_USER_PROMPT = ("""
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
        """)
    
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
            user_config=config
        )
        response = build_messages_and_create_chat_completion(implement_prompt, self.__DEFAULT_IMPLEMENT_ACTION_SYSTEM_PROMPT)
        self.save_chat_history_to_context_manager(implement_prompt, response, self.__DEFAULT_IMPLEMENT_ACTION_SYSTEM_PROMPT)

        res = re.search(r"Code:(.*)Explanation:(.*)Modified config:(.*)", response, re.S)
        assert res is not None and len(res.groups()) == 3, f"The response of implement action task of component {self.target_component} is not in the correct format"

        code = re.search(r"```python(.*)```", res.group(1), re.S)
        assert code is not None, "The code part of implementation action task response is not in the correct format"
        code = code.group(1)
        explanation = res.group(2)
        modified_config = re.search(r"```yaml(.*)```", res.group(3), re.S)
        assert modified_config is not None, "The modified config part of implementation action task response is not in the correct format"
        modified_config = modified_config.group(1)

        self._context_manager.set_context(f"{self.target_component}_code", code)
        self._context_manager.set_context(f"{self.target_component}_code_explanation", explanation)
        self._context_manager.set_context(f"{self.target_component}_modified_config", modified_config)

        return []

class SummarizeTask(Task):
    __DEFAULT_OUTPUT_PATH = "./"

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

    __MAX_LENGTH_OF_FILE = 9192
    __DEFAULT_REPORT_NAME = 'finCoReport.md'

    def __init__(self):
        super().__init__()

    def execute(self) -> Any:
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
        self.save_markdown(content=response)
        return response

    def summarize(self) -> str:
        return ''

    def interact(self) -> Any:
        return

    def parse2txt(self, path) -> List:
        """
        read specific type of files under path
        """
        file_list = []
        path = Path.cwd().joinpath(path).resolve()
        for root, dirs, files in os.walk(path):
            for filename in files:
                file_path = os.path.join(root, filename)
                print(f"file to summarize: {file_path}")
                file_list.append(file_path)

        result = []
        for file in file_list:
            postfix = file.split('.')[-1]
            if postfix in ['txt', 'py', 'log', 'yaml']:
                with open(file) as f:
                    content = f.read()
                    # in case of too large file
                    # TODO: Perhaps summarization method instead of truncation would be a better approach
                    result.append({'postfix': postfix, 'content': content[:self.__MAX_LENGTH_OF_FILE]})
        print(result)
        return result

    def save_markdown(self, content: str):
        with open(self.__DEFAULT_REPORT_NAME, "w") as f:
            f.write(content)
        print(f"report has saved to {self.__DEFAULT_REPORT_NAME}")

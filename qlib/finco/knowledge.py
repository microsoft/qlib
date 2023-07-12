from pathlib import Path
from jinja2 import Template
from typing import List, Union
import pickle
import yaml
import inspect

from qlib.workflow import R
from qlib.finco.log import FinCoLog
from qlib.finco.llm import APIBackend
from qlib.finco.utils import similarity


class Storage:
    """
    This class is responsible for storage and loading of Knowledge related data.

    """

    def __init__(self, path: Union[str, Path]):
        self.path = path if isinstance(path, Path) else Path(path)
        self.logger = FinCoLog()
        self.source = None

        # todo: get document by key
        self.documents = []

    def add(self, documents: List):
        self.documents.extend(documents)
        self.save()

    def load(self, **kwargs):
        raise NotImplementedError(f"Please implement the `load` method.")

    def save(self, **kwargs):
        raise NotImplementedError(f"Please implement the `save` method.")


class PickleStorage(Storage):
    """
    This class is responsible for storage and loading of Knowledge related data in pickle format.

    """

    def __init__(self, path: Union[str, Path]):
        super().__init__(path)

    @classmethod
    def load(cls, path: Union[str, Path]):
        """use pickle as the default load method"""
        path = path if isinstance(path, Path) else Path(path)
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, **kwargs):
        """use pickle as the default save method"""
        with open(self.path, "wb") as f:
            pickle.dump(self, f)


class YamlStorage(Storage):
    """
    This class is responsible for storage and loading of Knowledge related data in yaml format.

    """

    def __init__(self, path: Union[str, Path]):
        super().__init__(path)
        self.load()

    def load(self):
        """load data from yaml format file"""
        try:
            self.documents = yaml.load(open(self.path, "r"), Loader=yaml.FullLoader)
        except FileNotFoundError:
            self.logger.warning(f"YamlStorage: file {self.path} doesn't exist.")

    def save(self, **kwargs):
        """use pickle as the default save method"""
        with open(self.path, 'w') as f:
            yaml.dump(self.documents, f)


class ExperimentStorage(Storage):
    """
    This class is responsible for storage and loading of mlflow related data.

    """

    def __init__(self, exp_name, path=None):
        super().__init__(path=path)
        self.exp_name = exp_name
        self.exp = None
        self.recs = []
        self.docs = []

    def load(self, exp_name, rec_id=None):
        recs = []
        self.exp = R.get_exp(experiment_name=exp_name)
        for r in self.exp.list_recorders(rtype=self.exp.RT_L):
            if rec_id is not None and r.id != rec_id:
                continue
            recs.append(r)
        self.recs.extend(recs)


class Knowledge:
    """
    Use to handle knowledge in finCo such as experiment and outside domain information
    """

    def __init__(self, storage: Storage):
        self.logger = FinCoLog()
        self.storage = storage
        self.knowledge = []

    def summarize(self, **kwargs):
        """
        summarize storage data to knowledge, default knowledge is storage.documents

        Parameters
        ----------

        Return
        ------
        """

        self.knowledge = self.storage.documents

    def load(self, **kwargs):
        """
        Load knowledge in memory

        Parameters
        ----------

        Return
        ------
        """
        raise NotImplementedError(f"Please implement the `load` method.")

    def brief(self, **kwargs):
        """
        Return a brief summary of knowledge

        Parameters
        ----------

        Return
        ------
        """
        raise NotImplementedError(f"Please implement the `load` method.")

    def save(self, **kwargs):
        """save knowledge persistently"""
        self.storage.save(**kwargs)


class ExperimentKnowledge(Knowledge):
    """
    Handle knowledge from experiments
    """

    def __init__(self, storage: ExperimentStorage):
        super().__init__(storage=storage)
        self.storage = storage

    def brief(self):
        docs = []
        for recorder in self.storage.recs:
            docs.append({"exp_name": self.storage.exp.name, "record_info": recorder.info,
                         "config": recorder.load_object("config"),
                         "context_summary": recorder.load_object("context_summary")})
        return docs


class PracticeKnowledge(Knowledge):
    """
    some template sentence for now
    """

    def __init__(self, storage: YamlStorage):
        super().__init__(storage=storage)

        self.summarize()

    def add(self, docs: List):
        self.storage.add(docs)
        self.summarize()

        self.save()


class FinanceKnowledge(Knowledge):
    """
    Knowledge from articles
    """

    def __init__(self, storage: YamlStorage):
        super().__init__(storage=storage)
        if len(self.storage.documents) == 0:
            docs = self.read_files_in_directory(self.storage.path)
            self.add(docs)
        self.summarize()

    def add(self, docs: List):
        self.storage.add(docs)
        self.summarize()

        self.save()

    @staticmethod
    def read_files_in_directory(directory):
        """
        read all .txt files under directory
        """
        # todo: split article in trunks
        file_contents = []
        for file_path in Path(directory).rglob("*.txt"):
            if file_path.is_file():
                file_content = file_path.read_text(encoding="utf-8")
                file_contents.append(file_content)
        return file_contents


class ExecuteKnowledge(Knowledge):
    """
    Config and associate execution result(pass or error message). We can regard the example in prompt as pass execution
    """

    def __init__(self, storage: YamlStorage):
        super().__init__(storage=storage)
        self.summarize()

    def add(self, docs: List):
        self.storage.add(docs)
        self.summarize()

        self.save()


class InfrastructureKnowledge(Knowledge):
    """
    Knowledge from sentences, docstring, and code
    """

    def __init__(self, storage: YamlStorage):
        super().__init__(storage=storage)

        if len(self.storage.documents) == 0:
            # todo: change the path to qlib root path
            docs = self.get_functions_and_docstrings(Path.cwd().parent)
            self.add(docs)

    def add(self, docs: List):
        self.storage.add(docs)
        self.summarize()

        self.save()

    @staticmethod
    def get_functions_and_docstrings(directory):
        """
        get all method and docstring in .py files under directory
        """
        functions = []
        for file_path in Path(directory).rglob("*.py"):
            with file_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("def "):
                        function_name = line.split("(")[0][4:].strip()
                        function_docstring = inspect.getdoc(eval(function_name))
                        functions.append({"function_name": function_name, "docstring": function_docstring})
        return functions


class Topic:

    def __init__(self, name: str, describe: Template):
        self.name = name
        self.describe = describe
        self.docs = []
        self.knowledge = None
        self.logger = FinCoLog()

    def summarize(self, docs: list):
        self.logger.info(f"Summarize topic: \nname: {self.name}\ndescribe: {self.describe.module}")
        prompt_workflow_selection = self.describe.render(docs=docs)
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=prompt_workflow_selection
        )

        self.knowledge = response
        self.docs = docs


class KnowledgeBase:
    """
    Load knowledge, offer brief information of knowledge and common handle interfaces
    """

    KT_EXECUTE = "execute"
    KT_PRACTICE = "practice"
    KT_FINANCE = "finance"

    def __init__(self, init_path=None):
        self.logger = FinCoLog()
        self.init_path = Path(init_path) if init_path else Path.cwd()

        if not self.init_path.exists():
            self.logger.warning(f"{self.init_path} not exist, create empty directory.")
            Path.mkdir(self.init_path)

        self.practice_knowledge = self.load_practice_knowledge(self.init_path)
        self.execute_knowledge = self.load_execute_knowledge(self.init_path)
        self.finance_knowledge = self.load_finance_knowledge(self.init_path)

    def load_experiment_knowledge(self, path) -> List:
        # similar to practice knowledge, not use for now
        if isinstance(path, str):
            path = Path(path)

        knowledge = []
        path = path if path.name == "mlruns" else path.joinpath("mlruns")
        # todo: check the influence of set uri
        R.set_uri(path.as_uri())
        for exp_name in R.list_experiments():
            knowledge.append(ExperimentKnowledge(storage=ExperimentStorage(exp_name=exp_name)))

        self.logger.plain_info(f"Load knowledge from: {path} finished.")
        return knowledge

    def load_practice_knowledge(self, path: Path) -> PracticeKnowledge:
        self.practice_knowledge = PracticeKnowledge(YamlStorage(path.joinpath("practice_knowledge.yaml")))
        return self.practice_knowledge

    def load_execute_knowledge(self, path: Path) -> ExecuteKnowledge:
        self.execute_knowledge = ExecuteKnowledge(YamlStorage(path.joinpath("execute_knowledge.yaml")))
        return self.execute_knowledge

    def load_finance_knowledge(self, path: Path) -> FinanceKnowledge:
        self.finance_knowledge = FinanceKnowledge(YamlStorage(path.joinpath("finance_knowledge.yaml")))
        return self.finance_knowledge

    def knowledge(self, knowledge_type: str = None):
        if knowledge_type == self.KT_EXECUTE:
            knowledge = self.execute_knowledge
        elif knowledge_type == self.KT_PRACTICE:
            knowledge = self.practice_knowledge
        elif knowledge_type == self.KT_FINANCE:
            knowledge = self.finance_knowledge
        else:
            knowledge = self.execute_knowledge.knowledge + self.practice_knowledge.knowledge \
                        + self.finance_knowledge.knowledge
        return knowledge

    def query(self, knowledge_type: str = None, content: str = None, n: int = 5):
        """

        @param knowledge_type: self.KT_EXECUTE, self.KT_PRACTICE or self.KT_FINANCE
        @param content: content to query KnowledgeBase
        @param n: top n knowledge to ask ChatGPT
        @return:
        """
        # todo: replace list with persistent storage strategy such as ES/pinecone to enable
        # literal search/semantic search

        knowledge = self.knowledge(knowledge_type=knowledge_type)
        scores = []
        for k in knowledge:
            scores.append(similarity(str(k), content))
        sorted_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        similar_n_indexes = sorted_indexes[:n]
        similar_n_docs = [knowledge[i] for i in similar_n_indexes]

        prompt = Template("""summarize this information: '{{docs}}'""")
        prompt_workflow_selection = prompt.render(docs=similar_n_docs)
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=prompt_workflow_selection
        )

        return response

from pathlib import Path
from jinja2 import Template
from typing import List, Union
import pickle
import yaml

from qlib.workflow import R
from qlib.finco.log import FinCoLog
from qlib.finco.llm import APIBackend
from qlib.finco.utils import similarity, random_string, SingletonBaseClass

logger = FinCoLog()


class Storage:
    """
    This class is responsible for storage and loading of Knowledge related data.

    """

    def __init__(self, path: Union[str, Path], name: str = None):
        self.path = path if isinstance(path, Path) else Path(path)
        self.name = name if name else self.path.name
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
        Path.mkdir(self.path.parent, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self, f)


class YamlStorage(Storage):
    """
    This class is responsible for storage and loading of Knowledge related data in yaml format.

    """

    DEFAULT_NAME = "storage.yml"

    def __init__(self, path: Union[str, Path]):
        super().__init__(path)
        assert self.path.name, "Yaml storage should specify file name."
        self.load()

    def load(self):
        """load data from yaml format file"""
        try:
            self.documents = yaml.load(open(self.path, "r"), Loader=yaml.FullLoader)
        except FileNotFoundError:
            logger.warning(f"YamlStorage: file {self.path} doesn't exist.")

    def save(self, **kwargs):
        """use pickle as the default save method"""
        Path.mkdir(self.path.parent, exist_ok=True)
        with open(self.path, "w") as f:
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

    def __init__(self, storages: Union[List[Storage], Storage], name: str = None):
        self.name = name if name else random_string()
        self.workdir = Path.cwd().joinpath("knowledge")
        self.storages = [storages] if isinstance(storages, Storage) else storages
        self.knowledge = []

    def get_storage(self, name: str):
        """
        return first storage matched given name, else return None
        """
        for storage in self.storages:
            if storage.name == name:
                return storage
        return None

    def summarize(self, **kwargs):
        """
        summarize storage data to knowledge, default knowledge is storage.documents

        Parameters
        ----------

        Return
        ------
        """
        knowledge = []
        for storage in self.storages:
            knowledge.extend(storage.documents)
        self.knowledge = knowledge

    @classmethod
    def load(cls, path: Union[str, Path]):
        """
        Load knowledge in memory
        use pickle as the default file type
        Parameters
        ----------

        Return
        ------
        """
        """"""
        path = path if isinstance(path, Path) else Path(path)
        with open(path, "rb") as f:
            return pickle.load(f)

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
        # todo: storages save index only
        Path.mkdir(self.workdir.joinpath(self.name), exist_ok=True)
        with open(self.workdir.joinpath(self.name).joinpath("knowledge.pkl"), "wb") as f:
            pickle.dump(self, f)


class ExperimentKnowledge(Knowledge):
    """
    Handle knowledge from experiments
    """

    def __init__(self, storages: Union[List[ExperimentStorage], ExperimentStorage]):
        super().__init__(storages=storages)
        self.storage = storages

    def brief(self):
        docs = []
        for recorder in self.storage.recs:
            docs.append(
                {
                    "exp_name": self.storage.exp.name,
                    "record_info": recorder.info,
                    "config": recorder.load_object("config"),
                    "context_summary": recorder.load_object("context_summary"),
                }
            )
        return docs


class PracticeKnowledge(Knowledge):
    """
    some template sentence for now
    """

    def __init__(self, storages: Union[List[YamlStorage], YamlStorage]):
        super().__init__(storages=storages, name="practice")

        self.summarize()

    def add(self, docs: List, storage_name: str = YamlStorage.DEFAULT_NAME):
        storage = self.get_storage(storage_name)
        if storage is None:
            storage = YamlStorage(path=self.workdir.joinpath(self.name).joinpath(storage_name))
            storage.add(documents=docs)
            self.storages.append(storage)
        else:
            storage.add(documents=docs)

        self.summarize()
        self.save()


class FinanceKnowledge(Knowledge):
    """
    Knowledge from articles
    """

    def __init__(self, storages: Union[List[YamlStorage], YamlStorage]):
        super().__init__(storages=storages, name="finance")

        storage = self.get_storage(YamlStorage.DEFAULT_NAME)
        if len(storage.documents) == 0:
            docs = self.read_files_in_directory(self.workdir.joinpath(self.name))
            docs.extend([
                {"content": "[Success]: XXXX, the results looks reasonable  # Keywords: supervised learning, data"},
                {"content": "[Fail]: XXXX, it raise memory error due to  YYYYY  "
                            "# Keywords: supervised learning, data"}])
            self.add(docs)
        self.summarize()

    def add(self, docs: List, storage_name: str = YamlStorage.DEFAULT_NAME):
        storage = self.get_storage(storage_name)
        if storage is None:
            storage = YamlStorage(path=self.workdir.joinpath(self.name).joinpath(storage_name))
            storage.add(documents=docs)
            self.storages.append(storage)
        else:
            storage.add(documents=docs)

        self.summarize()
        self.save()

    @staticmethod
    def read_files_in_directory(directory) -> List:
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

    def __init__(self, storages: Union[List[YamlStorage], YamlStorage]):
        super().__init__(storages=storages, name="execute")
        self.summarize()

        storage = self.get_storage(YamlStorage.DEFAULT_NAME)
        if len(storage.documents) == 0:
            docs = [{"content": "[Success]: XXXX, the results looks reasonable  # Keywords: supervised learning, data"},
                    {"content": "[Fail]: XXXX, it raise memory error due to  YYYYY  "
                                "# Keywords: supervised learning, data"}]
            self.add(docs)
        self.summarize()

    def add(self, docs: List, storage_name: str = YamlStorage.DEFAULT_NAME):
        storage = self.get_storage(storage_name)
        if storage is None:
            storage = YamlStorage(path=self.workdir.joinpath(self.name).joinpath(storage_name))
            storage.add(documents=docs)
            self.storages.append(storage)
        else:
            storage.add(documents=docs)

        self.summarize()
        self.save()


class InfrastructureKnowledge(Knowledge):
    """
    Knowledge from sentences, docstring, and code
    """

    def __init__(self, storages: Union[List[YamlStorage], YamlStorage]):
        super().__init__(storages=storages, name="infrastructure")

        storage = self.get_storage(YamlStorage.DEFAULT_NAME)
        if len(storage.documents) == 0:
            docs = self.get_functions_and_docstrings(Path(__file__).parent.parent.parent)
            docs.extend([{"docstring": "All the models can be import from `qlib.contrib.models`  "
                                       "# Keywords: supervised learning"},
                         {"docstring": "The API to run rolling models can be found in …   #Keywords: control"},
                         {"docstring": "Here are a list of Qlib’s available analyzers.    #KEYWORDS: analysis"}])
            self.add(docs)
        self.summarize()

    def add(self, docs: List, storage_name: str = YamlStorage.DEFAULT_NAME):
        storage = self.get_storage(storage_name)
        if storage is None:
            storage = YamlStorage(path=self.workdir.joinpath(self.name).joinpath(storage_name))
            storage.add(documents=docs)
            self.storages.append(storage)
        else:
            storage.add(documents=docs)

        self.summarize()
        self.save()

    def get_functions_and_docstrings(self, directory) -> List:
        """
        get all method and docstring in .py files under directory

        """
        functions = []
        for py_file_path in Path(directory).rglob("*.py"):
            for _functions in self.get_functions_with_docstrings(py_file_path):
                functions.append(_functions)

        return functions

    @staticmethod
    def get_functions_with_docstrings(file_path):
        """
        Extract method name and docstring using string matching method
        """
        with open(file_path, "r", encoding="utf8") as f:
            lines = f.readlines()

        functions = []
        current_func = None
        docstring = None
        for line in lines:
            if line.strip().startswith("def ") or line.strip().startswith("class "):
                func = line.strip().split(" ")[1].split("(")[0]
                if func.startswith("__"):
                    continue
                if current_func is not None:
                    docstring = docstring.replace('"""', "") if docstring else docstring
                    functions.append({"function": current_func, "docstring": docstring})
                current_func = f"{file_path.name.split('.')[0]}.{func}"
                docstring = None
            elif current_func is not None and docstring is None and line.strip().startswith('"""'):
                docstring = line
            elif current_func is not None and docstring is not None:
                docstring += line.strip()
                if line.strip().endswith('"""'):
                    docstring = docstring.replace('"""', "") if docstring else docstring
                    functions.append({"function": current_func, "docstring": docstring})
                    current_func = None
                    docstring = None

        return functions


class Topic:
    def __init__(self, name: str, describe: Template):
        self.name = name
        self.describe = describe
        self.docs = []
        self.knowledge = None
        self.logger = FinCoLog()

    def summarize(self, docs: list):
        self.logger.info(f"Summarize Topic \nname: {self.name}\ndescribe: {self.describe.module}")
        prompt_workflow_selection = self.describe.render(docs=docs)
        response = APIBackend().build_messages_and_create_chat_completion(user_prompt=prompt_workflow_selection)

        self.knowledge = response
        self.docs = docs
        self.logger.info(f"Summary of {self.name}:\n{self.knowledge}")


class KnowledgeBase(SingletonBaseClass):
    """
    Load knowledge, offer brief information of knowledge and common handle interfaces
    """

    KT_EXECUTE = "execute"
    KT_PRACTICE = "practice"
    KT_FINANCE = "finance"
    KT_INFRASTRUCTURE = "infrastructure"

    def __init__(self, workdir=None):
        self.logger = FinCoLog()
        self.workdir = Path(workdir) if workdir else Path.cwd()

        if not self.workdir.exists():
            self.logger.warning(f"{self.workdir} not exist, create empty directory.")
            Path.mkdir(self.workdir)

        self.practice_knowledge = self.load_practice_knowledge(self.workdir)
        self.execute_knowledge = self.load_execute_knowledge(self.workdir)
        self.finance_knowledge = self.load_finance_knowledge(self.workdir)
        self.infrastructure_knowledge = self.load_infrastructure_knowledge(self.workdir)

    def load_experiment_knowledge(self, path) -> List:
        # similar to practice knowledge, not use for now
        if isinstance(path, str):
            path = Path(path)

        knowledge = []
        path = path if path.name == "mlruns" else path.joinpath("mlruns")
        # todo: check the influence of set uri
        R.set_uri(path.as_uri())
        for exp_name in R.list_experiments():
            knowledge.append(ExperimentKnowledge(storages=ExperimentStorage(exp_name=exp_name)))

        self.logger.plain_info(f"Load knowledge from: {path} finished.")
        return knowledge

    def load_practice_knowledge(self, path: Path) -> PracticeKnowledge:
        self.practice_knowledge = PracticeKnowledge(
            YamlStorage(path.joinpath(f"{self.KT_PRACTICE}/{YamlStorage.DEFAULT_NAME}"))
        )
        return self.practice_knowledge

    def load_execute_knowledge(self, path: Path) -> ExecuteKnowledge:
        self.execute_knowledge = ExecuteKnowledge(
            YamlStorage(path.joinpath(f"{self.KT_EXECUTE}/{YamlStorage.DEFAULT_NAME}"))
        )
        return self.execute_knowledge

    def load_finance_knowledge(self, path: Path) -> FinanceKnowledge:
        self.finance_knowledge = FinanceKnowledge(
            YamlStorage(path.joinpath(f"{self.KT_FINANCE}/{YamlStorage.DEFAULT_NAME}"))
        )
        return self.finance_knowledge

    def load_infrastructure_knowledge(self, path: Path) -> InfrastructureKnowledge:
        self.infrastructure_knowledge = InfrastructureKnowledge(
            YamlStorage(path.joinpath(f"{self.KT_INFRASTRUCTURE}/{YamlStorage.DEFAULT_NAME}"))
        )
        return self.infrastructure_knowledge

    def get_knowledge(self, knowledge_type: str = None):
        if knowledge_type == self.KT_EXECUTE:
            knowledge = self.execute_knowledge.knowledge
        elif knowledge_type == self.KT_PRACTICE:
            knowledge = self.practice_knowledge.knowledge
        elif knowledge_type == self.KT_FINANCE:
            knowledge = self.finance_knowledge.knowledge
        elif knowledge_type == self.KT_INFRASTRUCTURE:
            knowledge = self.infrastructure_knowledge.knowledge
        else:
            knowledge = (
                    self.execute_knowledge.knowledge
                    + self.practice_knowledge.knowledge
                    + self.finance_knowledge.knowledge
                    + self.infrastructure_knowledge.knowledge
            )
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

        knowledge = self.get_knowledge(knowledge_type=knowledge_type)
        if len(knowledge) == 0:
            return ""

        scores = []
        for k in knowledge:
            scores.append(similarity(str(k), content))
        sorted_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        similar_n_indexes = sorted_indexes[:n]
        similar_n_docs = [knowledge[i] for i in similar_n_indexes]

        prompt = Template(
            """find the most relevant doc with this query: '{{content}}' 
            from docs='{{docs}}. Just return the most relevant item I provided, no more explain. 
            For example: 
            user: find the most relevant doc with this query: ab \n from docs = {abc, xyz, lmn}.
            response: abc
            """
        )
        prompt_workflow_selection = prompt.render(content=content, docs=similar_n_docs)
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=prompt_workflow_selection, system_prompt="You are an excellent assistant."
        )

        return response


# perhaps init KnowledgeBase in other place
KnowledgeBase(workdir=Path.cwd().joinpath('knowledge'))

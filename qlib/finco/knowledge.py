from pathlib import Path
from jinja2 import Template
from typing import List

from qlib.workflow import R
from qlib.finco.log import FinCoLog
from qlib.finco.llm import APIBackend


class Knowledge:
    """
    Use to handle knowledge in finCo such as experiment and outside domain information
    """

    def __init__(self):
        self.logger = FinCoLog()

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


class KnowledgeExperiment(Knowledge):
    """
    Handle knowledge from experiments
    """

    def __init__(self, exp_name, rec_id=None):
        super().__init__()
        self.exp_name = exp_name
        self.exp = None
        self.recs = []

        self.load(exp_name=exp_name, rec_id=rec_id)

    def load(self, exp_name, rec_id=None):
        recs = []
        self.exp = R.get_exp(experiment_name=exp_name)
        for r in self.exp.list_recorders(rtype=self.exp.RT_L):
            if rec_id is not None and r.id != rec_id:
                continue
            recs.append(r)
        self.recs.extend(recs)

    def brief(self):
        docs = []
        for recorder in self.recs:
            docs.append(
                {
                    "exp_name": self.exp.name,
                    "record_info": recorder.info,
                    "config": recorder.load_object("config"),
                    "context_summary": recorder.load_object("context_summary"),
                }
            )

        return docs


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
        response = APIBackend().build_messages_and_create_chat_completion(user_prompt=prompt_workflow_selection)

        self.knowledge = response
        self.docs = docs


class KnowledgeBase:
    """
    Load knowledge, offer brief information of knowledge and common handle interfaces
    """

    def __init__(self, init_path=None, topics: List[Topic] = None):
        self.logger = FinCoLog()
        init_path = init_path if init_path else Path.cwd()

        if not init_path.exists():
            self.logger.warning(f"{init_path} not exist, create empty directory.")
            Path.mkdir(init_path)

        self.knowledge = self.load(path=init_path)

        # todo: replace list with persistent storage strategy such as ES/pinecone to enable
        # literal search/semantic search
        self.docs = self.brief(knowledge=self.knowledge)

        self.topics = topics if topics else []

    def load(self, path) -> List:
        if isinstance(path, str):
            path = Path(path)

        knowledge = []
        path = path if path.name == "mlruns" else path.joinpath("mlruns")
        R.set_uri(path.as_uri())
        for exp_name in R.list_experiments():
            knowledge.append(KnowledgeExperiment(exp_name=exp_name))

        self.logger.plain_info(f"Load knowledge from: {path} finished.")
        return knowledge

    def update(self, path):
        # note: only update new knowledge in future
        knowledge = self.load(path)
        self.knowledge = knowledge
        self.docs = self.brief(self.knowledge)
        self.logger.plain_info(f"Update knowledge finished.")

    def brief(self, knowledge: List[Knowledge]) -> List:
        docs = []
        for k in knowledge:
            docs.extend(k.brief())

        self.logger.plain_info(f"Generate brief knowledge summary finished.")
        return docs

    def query(self, content: str = None):
        # todo: query by DSL
        return self.docs

    def query_topics(self):
        knowledge_of_topics = []
        for topic in self.topics:
            knowledge_of_topics.append({topic.name: topic.knowledge})
        return knowledge_of_topics

    def summarize_by_topic(self):
        for topic in self.topics:
            topic.summarize(self.docs)

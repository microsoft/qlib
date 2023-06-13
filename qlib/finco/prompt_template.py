from jinja2 import Template
from qlib.finco.utils import Singleton
import yaml

class PormptTemplate(Singleton):
    def __init__(self) -> None:
        super().__init__()
        _template = yaml.load(open("./prompt_template.yaml", "r"), Loader=yaml.FullLoader)
        for k, v in _template.items():
            self.__setattr__(k, Template(v))

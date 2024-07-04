from typing import Union
from pathlib import Path
from jinja2 import Template
import yaml

from qlib.finco.utils import SingletonBaseClass
from qlib.finco import get_finco_path


class PromptTemplate(SingletonBaseClass):
    def __init__(self) -> None:
        super().__init__()
        _template = yaml.load(
            open(Path.joinpath(get_finco_path(), "prompt_template.yaml"), "r"), Loader=yaml.FullLoader
        )
        for k, v in _template.items():
            if k == "mods":
                continue
            self.__setattr__(k, Template(v))

    def get(self, key: str):
        return self.__dict__.get(key, Template(""))

    def update(self, key: str, value):
        self.__setattr__(key, value)

    def save(self, file_path: Union[str, Path]):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        Path.mkdir(file_path.parent, exist_ok=True)

        with open(file_path, "w") as f:
            yaml.dump(self.__dict__, f)

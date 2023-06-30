from typing import Union
from pathlib import Path
from jinja2 import Template
import yaml

from qlib.finco.utils import Singleton
from qlib.finco import get_finco_path


class PromptTemplate(Singleton):
    def __init__(self) -> None:
        super().__init__()
        _template = yaml.load(open(Path.joinpath(get_finco_path(), "prompt_template.yaml"), "r"),
                              Loader=yaml.FullLoader)
        for k, v in _template.items():
            if k == "mods":
                continue
            self.__setattr__(k, Template(v))

        for target_name, module_to_render_params in _template["mods"].items():
            for module_name, params in module_to_render_params.items():
                self.__setattr__(f"{target_name}_{module_name}",
                                 Template(self.__getattribute__(target_name).render(**params)))

    def get(self, key: str):
        return self.__dict__.get(key, Template(""))

    def update(self, key: str, value):
        self.__setattr__(key, value)

    def save(self, file_path: Union[str, Path]):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        Path.mkdir(file_path.parent, exist_ok=True)

        with open(file_path, 'w') as f:
            yaml.dump(self.__dict__, f)

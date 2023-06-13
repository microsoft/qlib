from jinja2 import Template
from qlib.finco.utils import Singleton
from qlib.finco import get_finco_path
import yaml
import os

class PormptTemplate(Singleton):
    def __init__(self) -> None:
        super().__init__()
        _template = yaml.load(open(os.path.join(get_finco_path(), "prompt_template.yaml"), "r"), Loader=yaml.FullLoader)
        for k, v in _template.items():
            if k == "mods":
                continue
            self.__setattr__(k, Template(v))
        
        for target_name, module_to_render_params in _template["mods"].items():
            for module_name, params in module_to_render_params.items():
                self.__setattr__(f"{target_name}_{module_name}", Template(self.__getattribute__(target_name).render(**params)))

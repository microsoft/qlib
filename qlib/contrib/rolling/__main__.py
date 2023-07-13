import fire
from qlib import auto_init
from qlib.contrib.rolling.base import Rolling
from qlib.utils.mod import find_all_classes

if __name__ == "__main__":
    sub_commands = {}
    for cls in find_all_classes("qlib.contrib.rolling", Rolling):
        sub_commands[cls.__module__.split(".")[-1]] = cls
    # The sub_commands will be like
    # {'base': <class 'qlib.contrib.rolling.base.Rolling'>, ...}
    # So the you can run it with commands like command below
    # - `python -m qlib.contrib.rolling base --conf_path <path to the yaml> run`
    # - base can be replace with other module names
    auto_init()
    fire.Fire(sub_commands)

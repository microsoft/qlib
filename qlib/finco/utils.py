import json
import string
import random

from typing import List
from pathlib import Path
from fuzzywuzzy import fuzz


class SingletonMeta(type):
    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instance


class SingletonBaseClass(metaclass=SingletonMeta):
    """
    Because we try to support defining Singleton with `class A(SingletonBaseClass)` instead of `A(metaclass=SingletonMeta)`
    This class becomes necessary

    """

    # TODO: Add move this class to Qlib's general utils.


def parse_json(response):
    try:
        return json.loads(response)
    except json.decoder.JSONDecodeError:
        pass

    raise Exception(f"Failed to parse response: {response}, please report it or help us to fix it.")


def similarity(text1, text2):
    text1 = text1 if isinstance(text1, str) else ""
    text2 = text2 if isinstance(text2, str) else ""

    # Maybe we can use other similarity algorithm such as tfidf
    return fuzz.ratio(text1, text2)


def random_string(length=10):
    letters = string.ascii_letters + string.digits
    return "".join(random.choice(letters) for i in range(length))


def directory_tree(root_dif, max_depth=None):

    def _directory_tree(root_dir, padding="", deep=1, max_d=None) -> List:
        _output = []
        if max_d and deep > max_d:
            return _output

        files = sorted(root_dir.iterdir())
        for i, file in enumerate(files):
            if i == len(files) - 1:
                _output.append(padding + '└── ' + file.name)
                if file.is_dir():
                    _output.extend(_directory_tree(file, padding + "    ", deep=deep + 1, max_d=max_d))
            else:
                _output.append(padding + '├── ' + file.name)
                if file.is_dir():
                    _output.extend(_directory_tree(file, padding + "│   ", deep=deep + 1, max_d=max_d))
        return _output

    output = _directory_tree(root_dif, max_d=max_depth)
    return '\n'.join(output)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


class Faculty:
    def __init__(self):
        self.__dict__["_faculty"] = dict()

    def __getitem__(self, key):
        return self.__dict__["_faculty"][key]

    def __getattr__(self, attr):
        if attr in self.__dict__["_faculty"]:
            return self.__dict__["_faculty"][attr]

        raise AttributeError(f"No such {attr} in self._faculty")

    def __setitem__(self, key, value):
        self.__dict__["_faculty"][key] = value

    def __setattr__(self, attr, value):
        self.__dict__["_faculty"][attr] = value

    def update(self, *args, **kwargs):
        self.__dict__["_faculty"].update(*args, **kwargs)


common_faculty = Faculty()

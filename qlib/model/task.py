import abc
import typing


class TaskGen(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> typing.List[dict]:
        """
        generate

        Parameters
        ----------
        args, kwargs:
            The info for generating tasks
            Example 1):
                input: a specific task template
                output: rolling version of the tasks
            Example 2):
                input: a specific task template
                output: a set of tasks with different losses

        Returns
        -------
        typing.List[dict]:
            A list of tasks
        """
        pass

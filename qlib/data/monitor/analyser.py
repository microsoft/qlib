from abc import abstractmethod


class Analyser:
    """
    Analyser is supposed to process the output MetricExt and produce a analysis result
    - The results could be a report or plot.

    We suppose the Analyser doesn't need much computing resource (The heavy computation should be done in MetricExt)
    """

    @abstractmethod
    def analyse(self, *args, **kwargs):
        ...

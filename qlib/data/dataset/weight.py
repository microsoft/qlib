# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


class Reweighter:
    def __init__(self, *args, **kwargs):
        """
        To initialize the Reweighter, users should provide specific methods to let reweighter do the reweighting (such as sample-wise, rule-based).
        """
        raise NotImplementedError()

    def reweight(self, data: object) -> object:
        """
        Get weights for data

        Parameters
        ----------
        data : object
            The input data.
            The first dimension is the index of samples

        Returns
        -------
        object:
            the weights info for the data
        """
        raise NotImplementedError(f"This type of input is not supported")

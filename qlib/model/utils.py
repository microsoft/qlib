# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class IndexSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def __getitem__(self, i: int):
        return self.sampler[i], i

    def __len__(self):
        return len(self.sampler)

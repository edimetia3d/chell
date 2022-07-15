from typing import List, Tuple

import numpy as np


class DataSet:

    def __getitem__(self, item: int):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def split(self, ratio: float = 0.8, shuffle=False) -> Tuple["SplitedDataSet", "SplitedDataSet"]:
        if ratio < 0 or ratio > 1:
            raise ValueError("ratio must be between 0 and 1")

        full_len = len(self)
        if shuffle:
            full_index = np.random.permutation(full_len)
        else:
            full_index = np.arange(full_len)
        first_part_len = int(full_len * ratio)
        first_part = full_index[:first_part_len]
        second_part = full_index[first_part_len:]
        return SplitedDataSet(self, first_part), SplitedDataSet(self, second_part)


class SplitedDataSet(DataSet):

    def __init__(self, raw_set: DataSet, valid_indices: List[int]):
        self.raw_set = raw_set
        self.valid_indices = valid_indices

    def __getitem__(self, item):
        return self.raw_set[self.valid_indices[item]]

    def __len__(self):
        return len(self.valid_indices)

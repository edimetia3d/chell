import numpy as np

from chell import dataset


class DataIter:

    def __init__(self, dset: dataset.DataSet, shuffle: bool, batch_size: int):
        self._dset = dset
        safe_len = int(len(dset) // batch_size * batch_size)
        if shuffle:
            indices = np.random.permutation(safe_len).reshape(-1, batch_size)
        else:
            indices = np.arange(0, safe_len).reshape(-1, batch_size)

        self._indice_iter = iter(indices)

    def __iter__(self):
        return self

    def __next__(self):
        indices = next(self._indice_iter)
        raw_list = []
        for i in indices:
            raw_list.append(self._dset[i])
        return zip(*raw_list)


class DataLoader:

    def __init__(self, dset: dataset.DataSet, *, shuffle: bool = False, batch_size: int = 1):
        self.data_set: dataset.DataSet = dset
        self.shuffle: bool = shuffle
        self.batch_size: int = batch_size

    def __iter__(self):
        return DataIter(self.data_set, self.shuffle, self.batch_size)

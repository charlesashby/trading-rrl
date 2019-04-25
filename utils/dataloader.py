import pickle
import glob
import numpy as np
import datetime

def load_data(data_path):
    with open(data_path, 'rb') as f:
        return pickle.load(f)


class DataLoader(object):

    def __init__(self, train_size, test_size, file):
        self.data = load_data(file)
        self.train_size = train_size
        self.test_size = test_size

    def iterate(self):
        """ len(self.data) = 11, self.train_size = 3, self.test_size = 2

            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            [_, _, _, t, t]                        [test_size * i: test_size * i + (test_size + train_size)]
                  [_, _, _, t, t]                  [2 * 1: 2 * 1 + (2 + 3)]
                        [_, _, _, t, t]            [2 * 2: 2 * 2 + (2 + 3)]
                              [_, _, _, t, t]      [2 * 3: 2 * 3 + (2 + 3)]

            The data fetched is: log_returns, year, month, day, weekday, hour, minute, second
            we normalize the dates and return a matrix for the training data and a matrix
            for the testing data
        """
        n_batch = (len(self.data) - self.train_size) // self.test_size
        for i in range(n_batch):
            data = self.data[self.test_size * i: self.test_size * i + (self.test_size + self.train_size)]
            # batch = data[:, list(np.arange(0, 100)) + [103, 104, 105]]

            # batch[:, 103] = batch[:, 103] / 6.
            # batch[:, 104] = batch[:, 104] / 24.
            # batch[:, 105] = batch[:, 105] / 60.
            # yield batch[: self.train_size], batch[self.train_size:]
            yield data[: self.train_size], data[self.train_size:]


class DataLoaderUSDJPY(object):

    def __init__(self, train_size, test_size, file):
        self.data = load_data(file)
        self.train_size = train_size
        self.test_size = test_size

    def iterate(self):
        """ len(self.data) = 11, self.train_size = 3, self.test_size = 2

            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            [_, _, _, t, t]                        [test_size * i: test_size * i + (test_size + train_size)]
                  [_, _, _, t, t]                  [2 * 1: 2 * 1 + (2 + 3)]
                        [_, _, _, t, t]            [2 * 2: 2 * 2 + (2 + 3)]
                              [_, _, _, t, t]      [2 * 3: 2 * 3 + (2 + 3)]

            The data fetched is: log_returns, year, month, day, weekday, hour, minute, second
            we normalize the dates and return a matrix for the training data and a matrix
            for the testing data
        """
        n_batch = (len(self.data) - self.train_size) // self.test_size
        for i in range(n_batch):
            data = self.data[self.test_size * i: self.test_size * i + (self.test_size + self.train_size)]
            # batch = data[:, list(np.arange(0, 100)) + [103, 104, 105]]

            # batch[:, 103] = batch[:, 103] / 6.
            # batch[:, 104] = batch[:, 104] / 24.
            # batch[:, 105] = batch[:, 105] / 60.
            # yield batch[: self.train_size], batch[self.train_size:]
            yield data[: self.train_size], data[self.train_size:]




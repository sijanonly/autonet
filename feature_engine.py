import numpy as np
import pandas as pd

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i : i + tw]
        train_label = input_data[i + tw : i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def sliding_window(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length):
        _x = data[i : (i + seq_length)]
        _y = data[i + seq_length : i + seq_length + 1]
        x.append(_x)
        y.append(_y)
    # x_arr = np.array(x).reshape(-1, seq_length)
    # y_arr = np.array(y).reshape(-1, seq_length)
    return (
        torch.from_numpy(np.array(x)).float(),
        torch.from_numpy(np.array(y)).float().view(-1, 1),
    )


def transform_data(arr, seq_len):
    x, y = [], []
    for i in range(len(arr) - seq_len):
        x_i = arr[i : i + seq_len]
        y_i = arr[i + 1 : i + seq_len + 1]
        x.append(x_i)
        y.append(y_i)
    x_arr = np.array(x).reshape(-1, seq_len)
    y_arr = np.array(y).reshape(-1, seq_len)
    x_var = torch.from_numpy(x_arr).float()
    y_var = torch.from_numpy(y_arr).float()
    return x_var, y_var


def prepare_train_test():
    data = pd.read_csv("data/A5M.csv", names=["hourly"])

    hour_data = data["hourly"].values.astype(float)
    # The first 1200 records will be used to train the model and the last 31 records will be used as a test set.
    test_data_size = 31
    test_frac = 0.1  # 10% for test
    val_frac = 0.25  # 25% for validation
    test_idx = int(len(hour_data) * (1 - test_frac))
    data, test_data = hour_data[:test_idx], hour_data[test_idx:]
    val_idx = int(len(data) * (1 - val_frac))
    train_data, val_data = (data[:val_idx], data[val_idx:])

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_norm = scaler.fit_transform(train_data.reshape(-1, 1))
    val_data_norm = scaler.transform(val_data.reshape(-1, 1))
    test_data_norm = scaler.transform(test_data.reshape(-1, 1))

    return train_data_norm, val_data_norm, test_data_norm


class Data(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = pd.read_csv(data_path)
        hour_data = data["hourly"].values.astype(float)
        self.transform = transform

    def __getitem__(self, index):
        # This method should return only 1 sample and label
        # (according to "index"), not the whole dataset
        # So probably something like this for you:
        pixel_sequence = self.data["pixels"][index]
        face = [int(pixel) for pixel in pixel_sequence.split(" ")]
        face = np.asarray(face).reshape(self.width, self.height)
        face = cv2.resize(face.astype("uint8"), (self.width, self.height))
        label = self.labels[index]

        return face, label

    def __len__(self):
        return len(self.labels)


class MyDataset(Dataset):
    def __init__(self, data, window):
        self.data = torch.Tensor(data)
        self.window = window
        # self.target_cols = target_cols
        self.shape = self.__getshape__()
        self.size = self.__getsize__()

    def __getitem__(self, index):
        # x = self.data[index : index + self.window]
        # y = self.data[index + self.window, 0:target_cols]
        _x = self.data[index : (index + self.window)]
        _y = self.data[index + self.window]
        return _x, _y

    def __len__(self):
        return len(self.data) - self.window

    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)

    def __getsize__(self):
        return self.__len__()


def get_batches(arr, batch_size, seq_length):
    """Create a generator that returns batches of size
       batch_size x seq_length from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    """
    #     print(len(arr))
    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr) // batch_size_total

    # Keep only enough characters to make full batches
    arr = arr[: n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    #     print(arr)
    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n : n + seq_length]
        # The targets, shifted by one (we want to predict next sequence given previous sequence)
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


def one_hot_encode(arr, n_labels):

    # Initialize the the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.0

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot


def generate_batch_data(x, y, batch_size):
    for batch, i in enumerate(range(0, len(x) - batch_size, batch_size)):
        x_batch = x[i : i + batch_size]
        y_batch = y[i : i + batch_size]
        yield x_batch, y_batch, batch


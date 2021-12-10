from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def load_dataset(name, batch_size):
    train = np.load("datasets/" + name + "_train.npy")
    mean = np.mean(np.concatenate(train, axis=0), axis=0)
    std = np.std(np.concatenate(train, axis=0), axis=0)
    train = (train - mean) / std

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test = np.load("datasets/" + name + "_test.npy")
    test = (test - mean) / std

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, mean, std
